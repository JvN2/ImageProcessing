from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

import ProcessImages.ImageIO as iio
import ProcessImages.TraceAnalysis as ta

if __name__ == "__main__":
    filename = r'D:\Data\Radek\22020725\Used_for_bleaching\data_031.dat'
    # filename = r'D:\Data\Radek\22020725\fixed fish2\data_047.dat'
    # filename = r'D:\Data\Radek\22020725\fixed fish1\data_035.dat'
    # filename = r'D:\Data\Radek\22020725\data_035X\data_035.dat'

    data = ta.Traces(filename)

    try:
        data.traces
    except:
        # read and process imaging parameters
        data.traces = pd.read_csv(filename, sep='\t')
        tiff_names = [str(Path(filename).with_name(f'image{image_nr:g}.tiff')) for image_nr in
                      data.traces['Filenr'].values]
        data.traces.insert(loc=0, column='Filename', value=tiff_names)
        data.traces['Time (s)'] -= data.traces['Time (s)'].min()
        print(f'Opened file: {filename}')
        print(f'Duration of experiment = {timedelta(seconds=int(data.traces["Time (s)"].max()))}.')

        # Correct drift
        drift = iio.DriftCorrection()
        for tiff_file in tqdm(data.traces['Filename'], postfix='Drift correction'):
            image = iio.filter_image(iio.read_tiff(tiff_file), highpass=data.get_glob('highpass'), lowpass=data.get_glob('lowpass'))
            drift.calc_drift(image, persistence=0.9)
        data.traces = pd.concat([data.traces, drift.get_df()], axis=1)

    data.read_log()

    data.set_glob('highpass', 3, section='Image processing')
    data.set_glob('lowpass', 25, section='Image processing')
    data.to_file()

    # data.traces = data.traces[::20]

    # process images and save as mp4 movie

    movie = iio.Movie(filename.replace('.dat', '.mp4'), 4)
    image = iio.filter_image(iio.read_tiff(data.traces['Filename'][0]), highpass=data.get_glob('highpass'), lowpass=data.get_glob('lowpass'))
    color_range = 1.2*np.asarray([np.percentile(image, 5), np.percentile(image, 99.9)])
    color_range = color_range.astype(int)
    movie.set_range(red=color_range, green=[0, 30], blue=[0, 30])
    # movie.set_circles(np.asarray([[256.0, 256.0]]),5)
    for frame, tiff_file in enumerate(tqdm(data.traces['Filename'], postfix='Add frames to movie')):
        image = iio.filter_image(iio.read_tiff(tiff_file), highpass=data.get_glob('highpass'), lowpass=data.get_glob('lowpass'))
        shifted_image = ndimage.shift(image, [data.traces.iloc[frame]['Drift x (pix)'],
                                              data.traces.iloc[frame]['Drift y (pix)']])
        label = f'I = {color_range}, T = {data.traces.iloc[frame]["Time (s)"]:.2f} s'
        movie.add_frame(red=shifted_image, label=label)
