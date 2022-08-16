from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import ProcessImages.ImageIO as iio
import ProcessImages.TraceAnalysis as ta


def read_image(filename):
    try:
        im = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(float).T
    except AttributeError:
        return
    # plt.plot(im)
    # plt.show()
    # return
    for i, _ in enumerate(im):
        im[i] -= np.percentile(im[i], 2)

    # scale intensity
    # i_range = [np.percentile(im, 2), np.percentile(im, 95)]
    # im -= i_range[0]
    # im /= i_range[1] -i_range[0]
    # im *= 255

    plt.imshow(im, cmap='Greys_r', vmin=-0.1 * np.percentile(im, 90), vmax=1.2 * np.percentile(im, 90))
    plt.colorbar()
    plt.show()
    return

    # resize image
    scale_percent = 30
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

    im_scaled = np.clip((im - range[0]) / (range[1] - range[0]), 0, 1)
    cv2.imshow(Path(filename).name, im_scaled)


def process_folder(foldername, make_movie=False):
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
        data.set_glob('Date', datetime.fromtimestamp(data.traces['Timestamp'].min()).strftime("%m/%d/%Y"), 'Aquisition')
        data.set_glob('Time', datetime.fromtimestamp(data.traces['Timestamp'].min()).strftime("%H:%M:%S"), 'Aquisition')
        data.set_glob('Timestamp', data.traces['Timestamp'].min(), 'Aquisition')
        data.traces.insert(loc=1, column='Time (s)', value=data.traces['Timestamp'] - data.traces['Timestamp'].min())
        data.traces.sort_values(by=['Time (s)'], inplace=True)
        data.traces.drop('Timestamp', axis=1, inplace=True)
        data.to_file()

    if make_movie:
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


if __name__ == '__main__':
    folder = r'D:\Users\lion\20220804\data_015'
    process_folder(folder, make_movie=True)
