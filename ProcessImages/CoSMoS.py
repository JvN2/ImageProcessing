import fnmatch
import warnings
from datetime import timedelta, datetime

import h5py
import numpy as np
import pandas as pd
from imaris_ims_file_reader.ims import ims
from scipy import ndimage
from tqdm import tqdm

import ProcessImages.ImageIO as iio
import ProcessImages.TraceAnalysis as ta

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def ims_read_header(filename, data=None):
    def time_string_to_seconds(time_string):
        dt, _, ms = time_string.partition('.')
        return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp() + float(ms) / 1000

    header = {}
    with h5py.File(filename, 'r') as hdf:
        header['Pixel (nm)'] = 1000 * np.abs(float(hdf[rf'DataSetInfo/Image'].attrs['ExtMax0'].tobytes().decode()) \
                                             - float(hdf[rf'DataSetInfo/Image'].attrs['ExtMin0'].tobytes().decode())) / \
                               float(hdf[rf'DataSetInfo/Image'].attrs['X'].tobytes().decode())
        channels = []
        for channel in range(6):
            try:
                channels.append(
                    hdf[rf'DataSetInfo/Channel {channel}'].attrs['LSMExcitationWavelength'][:3].tobytes().decode())
            except KeyError:
                header['Colors'] = channels

        time_attrs = [f'TimePoint{i + 1}' for i in
                      range(int(hdf[rf'DataSetInfo/TimeInfo'].attrs['FileTimePoints'].tobytes().decode()))]
        time = np.asarray(
            [time_string_to_seconds(hdf[rf'DataSetInfo/TimeInfo'].attrs[t].tobytes().decode()) for t in time_attrs])
        header['Start (s)'] = np.min(time)
        df = pd.DataFrame(time - header['Start (s)'], columns=['Time (s)'])
        if data is not None:
            data.traces['Time (s)'] = df['Time (s)'].values
            data.globs = pd.Series(header)
    return header, df


if __name__ == '__main__':
    # open data file
    filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.ims'
    filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.ims'
    filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV13_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.29.35.ims'

    image_stack = ims(filename, squeeze_output=True, ResolutionLevelLock=0, )
    data = ta.Traces(filename)

    if 'Pixel (nm)' not in data.globs.index:
        ims_read_header(filename, data)

    highpass = 100
    lowpass = 5

    # Correct drift
    if 'Drift x (nm)' not in data.traces.columns:
        drift = iio.DriftCorrection()
        for frame in tqdm(data.traces.index, postfix='Drift correction'):
            rgb_image = [iio.filter_image(image_stack[frame, data.globs['Colors'].index(c)], highpass=highpass) for c in
                         data.globs['Colors']]
            drift.calc_drift(np.sum(rgb_image, axis=0), persistence=0.5)
        data.traces = pd.concat([data.traces, drift.get_df(data.globs['Pixel (nm)'])], axis=1)

    # find peaks
    if 'X (pix)' not in data.pars.columns:
        summed_image = np.zeros_like(image_stack[0, 0]).astype(float)
        shift = np.asarray([data.traces['Drift x (nm)'], data.traces['Drift y (nm)']]).T / data.globs['Pixel (nm)']
        for frame, s in enumerate(tqdm(shift[:20], postfix='Sum images')):
            summed_image += ndimage.shift(image_stack[frame, data.globs['Colors'].index('637')], s)

        summed_image = iio.filter_image(summed_image, highpass=highpass, lowpass=lowpass, remove_outliers=True)

        data.globs['Radius (nm)'] = 250
        peaks = iio.find_peaks(summed_image, 4 * data.globs['Radius (nm)'] / data.globs['Pixel (nm)'], treshold_sd=2.0)
        data.pars = pd.concat([data.pars, peaks], axis=1)
        data.to_file()

    if len(fnmatch.filter(data.traces.columns, '*: I * (a.u.)')) == 0:
        # get traces
        trace_extraction = iio.TraceExtraction()
        coords = np.asarray(data.pars[['X (pix)', 'Y (pix)']])
        data.globs['Radius (nm)'] = 250
        trace_extraction.set_coords(coords, len(data.traces.index),
                                    data.globs['Radius (nm)'] / data.globs['Pixel (nm)'])
        for frame, _ in enumerate(tqdm(data.traces['Time (s)'], postfix='Extract traces')):
            for color, label in enumerate(data.globs['Colors']):
                image = iio.filter_image(image_stack[frame, color], highpass=highpass, lowpass=lowpass)
                trace_extraction.extract_intensities(image, frame, label=label)
        for col in trace_extraction.df.columns:
            data.traces[col] = trace_extraction.df[col]
        data.to_file()

    # save movie
    shift = np.asarray([data.traces['Drift x (nm)'], data.traces['Drift y (nm)']]).T / data.globs['Pixel (nm)']
    movie = iio.Movie()
    with movie(filename.replace('.ims', '.mp4'), 4):
        movie.set_range(red=[0, 30], green=[0, 30], blue=[0, 30])
        empty_image = np.zeros_like(image_stack[0, 0,]) - 1e6
        movie.set_circles(np.asarray(data.pars[['X (pix)', 'Y (pix)']]),
                          data.globs['Radius (nm)'] / data.globs['Pixel (nm)'])
        for frame, s in enumerate(tqdm(shift, postfix='Add frames to movie')):
            label = f'T = {timedelta(seconds=int(data.traces["Time (s)"][frame]))}'
            rgb_image = [
                iio.filter_image(image_stack[frame, data.globs['Colors'].index(c)], highpass=highpass, lowpass=lowpass)
                if c in data.globs['Colors'] else empty_image for c in ['637', '561', '488']]
            rgb_image = ndimage.shift(rgb_image, [0, s[0], s[1]])
            movie.add_frame(red=rgb_image[0], green=rgb_image[1], blue=rgb_image[2], label=label)
