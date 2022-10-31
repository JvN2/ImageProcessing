from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import ProcessImages.ImageIO as iio
import ProcessImages.TraceAnalysis as ta


if __name__ == '__main__':
    folder = r'D:\Data\Pyseq\20220804\data_015'

    _MAKE_MOVIE_ = True

    data = ta.Traces(folder)

    if data.traces.empty:
        filenames = [str(filename.with_suffix('.txt')).replace('c558_', 'meta_') for filename in
                     Path(folder).glob('c558*.tiff')]
        data.traces = pd.DataFrame(filenames, columns=['Filename'])

        for index, row in data.traces.iterrows():
            with open(row['Filename']) as f:
                for line in f:
                    item, *values = line.split(' ')
                    if item == 'time':
                        date = datetime.strptime(values[0][:-1], "%Y%m%d_%H%M%S")
                        data.traces.at[index, 'Timestamp'] = date.timestamp()
                    if item == 'x':
                        data.traces.at[index, 'X (um)'] = float(values[0])
                    if item == 'y':
                        data.traces.at[index, 'Y (um)'] = float(values[0])
                    if item == 'laser1':
                        data.traces.at[index, 'I532nm (mW)'] = float(values[0])
                    if item == 'laser2':
                        data.traces.at[index, 'I660nm (mW)'] = float(values[0])
        data.set_glob('Date',datetime.fromtimestamp(data.traces['Timestamp'].min()).strftime("%m/%d/%Y"), 'Aquisition')
        data.set_glob('Time',datetime.fromtimestamp(data.traces['Timestamp'].min()).strftime("%H:%M:%S"), 'Aquisition')
        data.set_glob('Timestamp', data.traces['Timestamp'].min(), 'Aquisition')
        data.traces.insert(loc=1, column='Time (s)', value=data.traces['Timestamp'] - data.traces['Timestamp'].min())
        data.traces.sort_values(by=['Time (s)'], inplace=True)
        data.traces.drop('Timestamp', axis= 1, inplace=True)
        data.to_file()

    if _MAKE_MOVIE_:
        movie = iio.Movie(str(Path(data.filename).with_suffix('.mp4')), 4)
        # radius_pix = 0.001 * data.get_glob('Radius (nm)') / data.get_glob('Pixel size (um)')
        # movie.set_circles(np.asarray(data.pars[['X (pix)', 'Y (pix)']]), radius_pix)

        colors = {'red': 'c558'}
        color_ranges = {'red': 0, 'green': 0, 'blue': 0}

        for frame, filename in enumerate(tqdm(data.traces['Filename'], postfix='Add frames to movie')):
            for color in colors:
                image = {'red': None, 'green': None, 'blue': None}
                filename = filename.replace('meta', colors[color]).replace('txt', 'tiff')
                image[color] = np.asarray(cv2.imread(filename, cv2.IMREAD_ANYDEPTH)).astype(float).T

                for i, _ in enumerate(image[color]):
                    image[color][i] -= np.percentile(image[color][i], 2)

                if frame == 0:
                    color_ranges[color] = np.asarray([-0.05, 0.95]) * np.percentile(image[color], 95)
                    movie.set_range(red=color_ranges['red'], green=color_ranges['green'], blue=color_ranges['blue'])

            label = f'Channel = {colors["red"]}, T = {data.traces.iloc[frame]["Time (s)"]:.0f} s'
            movie.add_frame(red=image['red'], green=image['green'], blue=image['blue'], label=label)
