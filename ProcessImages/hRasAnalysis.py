from datetime import timedelta
from pathlib import Path

import sklearn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

import ProcessImages.ImageIO as iio
import ProcessImages.TraceAnalysis as ta

if __name__ == "__main__":
    _READ_FROM_FILE_ = True
    _CORRECT_DRIFT_ = True
    _MAX_FRAMES_ = True
    _MAKE_MOVIE_ = True


    filename = r'D:\Data\Radek\22020725\Used_for_bleaching\data_031.dat'
    filename = r'D:\Data\Radek\22020725\fixed fish2\data_047.dat'
    # filename = r'D:\Data\Radek\22020725\fixed fish1\data_035.dat'
    # filename = r'D:\Data\Radek\22020725\data_035X\data_035.dat'

    data = ta.Traces(filename)
    if data.traces.empty:
        # read and process imaging parameters
        data.traces = pd.read_csv(Path(filename).with_suffix('.dat'), sep='\t')
        tiff_names = [str(Path(filename).with_name(f'image{image_nr:g}.tiff')) for image_nr in
                      data.traces['Filenr'].values]
        data.traces.insert(loc=0, column='Filename', value=tiff_names)
        data.traces['Time (s)'] -= data.traces['Time (s)'].min()
        print(f'Opened file: {filename}')
        print(f'Duration of experiment = {timedelta(seconds=int(data.traces["Time (s)"].max()))}.')
        data.read_log()
        data.set_glob('highpass', 3, section='Image processing')
        data.set_glob('lowpass', 25, section='Image processing')
        data.to_file()

    if 'Drift x (pix)' not in data.traces.columns:
        drift = iio.DriftCorrection()
        for tiff_file in tqdm(data.traces['Filename'], postfix='Drift correction'):
            image = iio.filter_image(iio.read_tiff(tiff_file), highpass=data.get_glob('highpass'),
                                     lowpass=data.get_glob('lowpass'))
            drift.calc_drift(image, persistence=0.9)
        data.traces = pd.concat([data.traces, drift.get_df()], axis=1)

    # data.traces = data.traces[::10]

    if 'X (pix)' not in data.pars.columns:
        for frame, tiff_file in enumerate(tqdm(data.traces['Filename'], postfix='Sum frames')):
            image = iio.filter_image(iio.read_tiff(tiff_file), highpass=data.get_glob('highpass'),
                                     lowpass=data.get_glob('lowpass'))
            image = iio.filter_image(iio.read_tiff(tiff_file), highpass=0)
            shifted_image = ndimage.shift(image, [data.traces.iloc[frame]['Drift x (pix)'],
                                                  data.traces.iloc[frame]['Drift y (pix)']])
            if frame == 0:
                color_range = (np.asarray([-0.5, 1.5]) * np.percentile(shifted_image, 99.5)).astype(int)
                max_image = shifted_image
            else:
                max_image = np.max(np.asarray([max_image, shifted_image]), axis=0)

        im = [iio.scale_u8(max_image, color_range) / 255, np.zeros_like(max_image), np.zeros_like(max_image)]
        im = np.moveaxis(im, 0, 2)

        plt.imsave(Path(filename).with_suffix('.png'), im)

        data.set_glob('Radius (nm)', 250, 'Trace extraction')
        radius_pix = 0.001 * data.get_glob('Radius (nm)') / data.get_glob('Pixel size (um)')
        peaks = iio.find_peaks(max_image, radius_pix, treshold_sd=4.0)
        # print(sklearn.metrics.pairwise_distances(peaks))
        data.pars = pd.concat([data.pars, peaks], axis=1)
        data.to_file()

    if _MAKE_MOVIE_:
        movie = iio.Movie(filename.replace('.dat', '.mp4'), 4)
        radius_pix = 0.001 * data.get_glob('Radius (nm)') / data.get_glob('Pixel size (um)')
        movie.set_circles(np.asarray(data.pars[['X (pix)', 'Y (pix)']]), radius_pix)
        for frame, tiff_file in enumerate(tqdm(data.traces['Filename'], postfix='Add frames to movie')):
            image = iio.filter_image(iio.read_tiff(tiff_file), highpass=data.get_glob('highpass'),
                                     lowpass=data.get_glob('lowpass'))
            shifted_image = ndimage.shift(image, [data.traces.iloc[frame]['Drift x (pix)'],
                                                  data.traces.iloc[frame]['Drift y (pix)']])

            if frame == 0:
                color_range = (np.asarray([-0.5, 1.5]) * np.percentile(shifted_image, 99.5)).astype(int)
                movie.set_range(red=color_range, green=[0, 30], blue=[0, 30])

            label = f'I = {color_range}, T = {data.traces.iloc[frame]["Time (s)"]:.2f} s'
            movie.add_frame(red=shifted_image, label=label)


