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

import matplotlib.pyplot as plt

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
    ref_filename = r'C:\Users\noort\Downloads\Slide2_Chann1_LigA+DNA_FOV3_50Expboth_50Pwboth_40rep_2022-07-08_488-637_Zyla_17.59.46.ims'
    filename = r'C:\Users\noort\Downloads\Slide2_Chann1_LigA_FOV3_50Expboth_50Pwboth_500rep_2022-07-08_488_Zyla_18.00.37.ims'

    ref_filename = r'D:\2022-07-08\Slide1_Chann1_Pol1FL+DNA_FOV4_50Expboth_50Pwboth_40rep_2022-07-08_565+637_Zyla_16.27.09.ims'
    filename =     r'D:\2022-07-08\Slide1_Chann1_Pol1FL_FOV4_50Expboth_50Pwboth_500rep_2022-07-08_565_Zyla_16.28.52.ims'


    image_stack = ims(filename, squeeze_output=True, ResolutionLevelLock=0, )
    data = ta.Traces(filename)

    if ref_filename is not None:
        ref_image_stack = ims(ref_filename, squeeze_output=True, ResolutionLevelLock=0, )
        ref_globs, ref_df = ims_read_header(ref_filename)

    if 'Pixel (nm)' not in data.globs.index:
        ims_read_header(filename, data)

    highpass = 5
    lowpass = 100
    anchor_peaks = '637'

    # Correct drift
    # if False:
    if 'Drift x (nm)' not in data.traces.columns:
        drift = iio.DriftCorrection()
        for frame in tqdm(data.traces.index, postfix='Drift correction'):
            ref_image = [iio.filter_image(image_stack[frame, data.globs['Colors'].index(c)], highpass=highpass) for c in
                         data.globs['Colors']]
            drift.calc_drift(np.sum(ref_image, axis=0), persistence=0.5)
        data.traces = pd.concat([data.traces, drift.get_df(data.globs['Pixel (nm)'])], axis=1)

        if ref_filename is not None:
            data.globs['reference filename'] = ref_filename
            image = image_stack[0, 0]
            image = iio.filter_image(image, highpass=highpass)

            ref_image = ref_image_stack[
                ref_image_stack.shape[0] - 1, ref_globs['Colors'].index(data.globs['Colors'][0])]
            ref_image = iio.filter_image(image, highpass=highpass)
            shift = drift.calc_drift(image, ref_image=ref_image)
            for s, c in zip(shift, ['x', 'y']):
                data.traces[f'Drift {c} (nm)'] -= s

        data.to_file()

    # find peaks
    if 'X (pix)' not in data.pars.columns:
        if ref_filename is None:
            mean_image = np.zeros_like(image_stack[0, 0]).astype(float)
            shift = np.asarray([data.traces['Drift x (nm)'], data.traces['Drift y (nm)']]).T / data.globs['Pixel (nm)']
            mean_length = 40
            for frame, s in enumerate(tqdm(shift[:mean_length], postfix='Sum images')):
                mean_image += ndimage.shift(image_stack[frame, data.globs['Colors'].index(anchor_peaks)], s)
            mean_image /= mean_length
        else:
            df = pd.read_excel(ref_filename.replace('.ims', '.xlsx'), 'traces')
            for a in ['x', 'y']:
                df[f'Drift {a} (nm)'] -= df[f'Drift {a} (nm)'].iloc[-1]
            mean_image = np.zeros_like(ref_image_stack[0, 0]).astype(float)
            shift = np.asarray([df['Drift x (nm)'], df['Drift y (nm)']]).T / data.globs['Pixel (nm)']
            for frame, s in enumerate(tqdm(shift, postfix='Sum images')):
                mean_image += ndimage.shift(ref_image_stack[frame, ref_globs['Colors'].index(anchor_peaks)], s)
            mean_image /= len(shift)
        mean_image = iio.filter_image(mean_image, highpass=highpass, lowpass=lowpass, remove_outliers=True)
        data.globs['Radius (nm)'] = 250
        peaks = iio.find_peaks(mean_image, 4 * data.globs['Radius (nm)'] / data.globs['Pixel (nm)'], treshold_sd=5.0)
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
    if True:
        shift = np.asarray([data.traces['Drift x (nm)'], data.traces['Drift y (nm)']]).T / data.globs['Pixel (nm)']
        # movie = iio.Movie()
        # with movie(filename.replace('.ims', '.mp4'), 4):
        movie = iio.Movie(filename.replace('.ims', '.mp4'), 4)
        movie.set_range(red=[0, 30], green=[0, 30], blue=[0, 30])
        empty_image = np.zeros_like(image_stack[0, 0,]) - 1e6
        movie.set_circles(np.asarray(data.pars[['X (pix)', 'Y (pix)']]),
                          data.globs['Radius (nm)'] / data.globs['Pixel (nm)'])
        for frame, s in enumerate(tqdm(shift, postfix='Add frames to movie')):
            label = f'T = {timedelta(seconds=int(data.traces["Time (s)"][frame]))}'
            ref_image = [
                iio.filter_image(image_stack[frame, data.globs['Colors'].index(c)], highpass=highpass, lowpass=lowpass)
                if c in data.globs['Colors'] else empty_image for c in ['637', '561', '488']]
            ref_image = ndimage.shift(ref_image, [0, s[0], s[1]])

            # if ref_filename is not None:
            #     ref_image[['637', '561', '488'].index(anchor_peaks)] = mean_image

            movie.add_frame(red=ref_image[0], green=ref_image[1], blue=ref_image[2], label=label)
