from datetime import timedelta, datetime

import h5py
import numpy as np
import pandas as pd
from imaris_ims_file_reader.ims import ims
from scipy import ndimage
from tqdm import tqdm

import ProcessImages.ImageIO3 as im3


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
                header['colors'] = channels
        header['nm_pix'] = 1000 * np.abs(float(hdf[rf'DataSetInfo/Image'].attrs['ExtMax0'].tobytes().decode()) \
                                         - float(hdf[rf'DataSetInfo/Image'].attrs['ExtMin0'].tobytes().decode())) / \
                           float(hdf[rf'DataSetInfo/Image'].attrs['X'].tobytes().decode())

        time_attrs = [f'TimePoint{i + 1}' for i in
                      range(int(hdf[rf'DataSetInfo/TimeInfo'].attrs['FileTimePoints'].tobytes().decode()))]
        time = np.asarray(
            [time_string_to_seconds(hdf[rf'DataSetInfo/TimeInfo'].attrs[t].tobytes().decode()) for t in time_attrs])
        header['time'] = time - time[0]

        df = pd.DataFrame(header['time'], columns=['Time (s)'])
    return header, df


if __name__ == '__main__':
    # open data file and df
    filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.ims'
    image_stack = ims(filename, squeeze_output=True, ResolutionLevelLock=0)
    header, df = ims_read_header(filename)

    try:
        df = pd.read_csv(filename.replace('.ims', '.csv'))
    except FileNotFoundError:
        # Correct drift
        drift = im3.DriftCorrection()
        for i in tqdm(df.index, 'Drift correction'):
            rgb_image = [im3.filter_image(image_stack[i, header['colors'].index(c)], highpass=0) for c in
                         header['colors']]
            drift.calc_drift(np.sum(rgb_image, axis=0), persistence=0.5)
        df = pd.concat([df, drift.get_df(header['nm_pix'])], axis=1)
        df.to_csv(filename.replace('.ims', '.csv'), index=False)

    # find peaks
    summed_image = np.zeros_like(image_stack[0, 0]).astype(float)
    shift = np.asarray([df['Drift x (nm)'], df['Drift y (nm)']]).T / header['nm_pix']
    for i, s in enumerate(tqdm(shift[:20], postfix='Sum images')):
        summed_image += ndimage.shift(im3.filter_image(image_stack[i, header['colors'].index('637')], 0), s)
    radius = 125 / header['nm_pix']
    peaks = im3.find_peaks(summed_image, radius * 4, treshold_sd=2.0)

    # get traces
    traces = im3.TraceExtraction()
    traces.set_coords(peaks, radius)
    for i, s in enumerate(tqdm(shift, postfix='Extract traces')):
        for label in ['561', '637']:
            image = im3.filter_image(image_stack[i, header['colors'].index(label)], highpass=0)
            traces.extract_intensities(image, label=label)
    traces.df.to_csv(filename.replace('.ims', '_traces.csv'))
    # df = pd.concat([df, traces.df], axis=1, join='outer')

    # save movie
    shift = np.asarray([df['Drift x (nm)'], df['Drift y (nm)']]).T / header['nm_pix']
    movie = im3.Movie()
    with movie(filename.replace('.ims', '_test.mp4'), 4):
        movie.set_range(red=[0, 30], green=[0, 30])
        movie.set_circles(peaks, radius)
        for i, s in enumerate(tqdm(shift, postfix='Add frames to movie')):
            label = f'T = {timedelta(seconds=int(header["time"][i]))}'
            rgb_image = [im3.filter_image(image_stack[i, header['colors'].index(c)], highpass=0) for c in
                         ['637', '561']]
            rgb_image = ndimage.shift(rgb_image, [0, s[0], s[1]])
            movie.add_frame(red=rgb_image[0], green=rgb_image[1], label=label)
