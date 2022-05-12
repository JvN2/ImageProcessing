from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import ProcessImages.ImageIO3 as im3

if __name__ == "__main__":
    filename = r'\\data02\pi-vannoort\Noort\Data\Alex GR\data_004\data_004.dat'

    # read and process imaging parameters
    df = pd.read_csv(filename, sep='\t')
    tiff_names = [str(Path(filename).with_name(f'image{image_nr:g}.tiff')) for image_nr in df['Filenr'].values]
    df.insert(loc=0, column='Filename', value=tiff_names)
    df['Time (s)'] -= df['Time (s)'].min()
    print(f'Opened file: {filename}')
    print(f'Duration of experiment = {timedelta(seconds=int(df["Time (s)"].max()))}.')

    # process images and save as mp4 movie
    movie = im3.Movie()
    with movie(filename.replace('.dat', '_test.mp4'), 4):
        movie.set_range(grey=[-10, 70])
        frames = df['Frame'].unique()
        for i, frame in enumerate(tqdm(frames, 'Processing frames')):
            stack = [im3.read_tiff(tiff_file) for tiff_file in df[(df['Frame'] == frame) & (df['Slice'] != 0)]['Filename']]

            projection = np.max(np.asarray(stack), axis=0)
            projection = im3.filter_image(projection, 0, 10)

            label = f'T = {timedelta(seconds=int(df[(df["Frame"] == frame) & (df["Slice"] != 0)]["Time (s)"].min()))}'
            movie.add_frame(grey=projection, label=label)
