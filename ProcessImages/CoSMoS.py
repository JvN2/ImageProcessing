import warnings
from datetime import timedelta, datetime

import h5py
import numpy as np
import pandas as pd
from imaris_ims_file_reader.ims import ims
from scipy import ndimage
from tqdm import tqdm

import ProcessImages.ImageIO as iio
import ProcessImages.trace_analysis as ta

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def ims_read_header(filename):
    def time_string_to_seconds(time_string):
        dt, _, ms = time_string.partition('.')
        return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp() + float(ms) / 1000

    header = {}
    with h5py.File(filename, 'r') as hdf:
        channels = []
        for channel in range(6):
            try:
                channels.append(
                    hdf[rf'DataSetInfo/Channel {channel}'].attrs['LSMExcitationWavelength'][:3].tobytes().decode())
            except KeyError:
                header['Colors'] = channels
        header['Pixel (nm)'] = 1000 * np.abs(float(hdf[rf'DataSetInfo/Image'].attrs['ExtMax0'].tobytes().decode()) \
                                             - float(hdf[rf'DataSetInfo/Image'].attrs['ExtMin0'].tobytes().decode())) / \
                               float(hdf[rf'DataSetInfo/Image'].attrs['X'].tobytes().decode())

        time_attrs = [f'TimePoint{i + 1}' for i in
                      range(int(hdf[rf'DataSetInfo/TimeInfo'].attrs['FileTimePoints'].tobytes().decode()))]
        time = np.asarray(
            [time_string_to_seconds(hdf[rf'DataSetInfo/TimeInfo'].attrs[t].tobytes().decode()) for t in time_attrs])
        header['Start (s)'] = np.min(time)
        df = pd.DataFrame(time - header['Start (s)'], columns=['Time (s)'])
    return header, df


if __name__ == '__main__':
    # open data file
    filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.ims'
    filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.ims'
    # filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV13_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.29.35.ims'
    image_stack = ims(filename, squeeze_output=True, ResolutionLevelLock=0, )

    globs, traces = ims_read_header(filename)

    highpass = 100
    lowpass = 5

    # Correct drift
    try:
        traces, _, _ = ta.read_hdf(filename)
    except FileNotFoundError:
        drift = iio.DriftCorrection()
        for frame in tqdm(traces.index, postfix='Drift correction'):
            rgb_image = [iio.filter_image(image_stack[frame, globs['Colors'].index(c)], highpass=highpass) for c in
                         globs['Colors']]
            drift.calc_drift(np.sum(rgb_image, axis=0), persistence=0.5)
        traces = pd.concat([traces, drift.get_df(globs['Pixel (nm)'])], axis=1)

    # find peaks
    summed_image = np.zeros_like(image_stack[0, 0]).astype(float)
    shift = np.asarray([traces['Drift x (nm)'], traces['Drift y (nm)']]).T / globs['Pixel (nm)']
    for frame, s in enumerate(tqdm(shift[:20], postfix='Sum images')):
        summed_image += ndimage.shift(image_stack[frame, globs['Colors'].index('637')], s)

    summed_image = iio.filter_image(summed_image, highpass=highpass, lowpass=lowpass, remove_outliers=True)

    globs['Radius (nm)'] = 250
    peaks = iio.find_peaks(summed_image, 4 * globs['Radius (nm)'] / globs['Pixel (nm)'], treshold_sd=2.0)

    # get traces
    trace_extraction = iio.TraceExtraction()
    trace_extraction.set_coords(np.asarray(peaks), len(traces['Time (s)']), globs['Radius (nm)'] / globs['Pixel (nm)'])
    for frame, _ in enumerate(tqdm(traces['Time (s)'], postfix='Extract traces')):
        for color, label in enumerate(globs['Colors']):
            image = iio.filter_image(image_stack[frame, color], highpass=highpass, lowpass=lowpass)
            trace_extraction.extract_intensities(image, frame, label=label)
    try:
        traces = traces.join(trace_extraction.df)
    except ValueError:
        traces.update(trace_extraction.df)

    # save all data
    ta.save_hdf(filename, traces=traces, pars=peaks, globs=globs)

    # save movie
    shift = np.asarray([traces['Drift x (nm)'], traces['Drift y (nm)']]).T / globs['Pixel (nm)']
    movie = iio.Movie()
    with movie(filename.replace('.ims', '.mp4'), 4):
        movie.set_range(red=[0, 30], green=[0, 30], blue=[0, 30])
        movie.set_circles(np.asarray(peaks), globs['Radius (nm)'] / globs['Pixel (nm)'])
        for frame, s in enumerate(tqdm(shift, postfix='Add frames to movie')):
            label = f'T = {timedelta(seconds=int(traces["Time (s)"][frame]))}'
            rgb_image = [
                iio.filter_image(image_stack[frame, globs['Colors'].index(c)], highpass=highpass, lowpass=lowpass)
                for c in globs['Colors']]
            rgb_image = ndimage.shift(rgb_image, [0, s[0], s[1]])
            movie.add_frame(red=rgb_image[globs['Colors'].index('637')], green=rgb_image[globs['Colors'].index('561')],
                            blue=rgb_image[globs['Colors'].index('488')], label=label)
