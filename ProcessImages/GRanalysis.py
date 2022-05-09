from datetime import timedelta, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import ProcessImages.CoSMoS as cms


def read_tiff(filename):
    return np.asarray(plt.imread(Path(filename)).astype(float))


def read_tiffs_from_df(filename, df):
    tiff_names = [str(Path(filename).with_name(f'image{image_nr:g}.tiff')) for image_nr in df['Filenr'].values]

    time = list(set(df['Frame']))
    transmission_image = df.iloc[0]['PiezoZ (um)'] > df.iloc[1]['PiezoZ (um)']
    color = [0, 1] if transmission_image else [0]
    z_position = list(set(df['PiezoZ (um)']))[1:] if transmission_image else set(df['PiezoZ (um)'])
    image = read_tiff(tiff_names[0])

    ims = np.zeros([len(time), 3, len(z_position), len(image[0, :]), len(image[:, 0])], dtype='u8')
    for _, row in tqdm(df.iterrows(), 'Opening files'):
        im = read_tiff(Path(filename).with_name(f'image{row["Filenr"]:g}.tiff'))
        ims[int(time.index(row['Frame'])), int(row['Slice'] != 0),
        int(row['Slice']) - 1 if int(row['Slice'] != 0) else 0, :, :] = cms.scale_image_u8(im, z_range=[0, 255])

    return ims


if __name__ == "__main__":
    filename = r'D:\Alex\Internship 2022\TPMM data\220428\data_004\data_004.dat'

    df = pd.read_csv(filename, sep='\t')

    print(f'Opened dataframe: {filename}')
    print(f'Duration of experiment = {timedelta(seconds=int(df["Time (s)"].max() - df["Time (s)"].min()))}.')

    frames = list(range(0,100, 10))
    print(frames)

    df2 = df[df['Frame'].isin(frames)]
    image_stack = read_tiffs_from_df(filename, df2)
    for frame_nr, im in enumerate(image_stack):
        image_stack[frame_nr, 2, 0,] = np.max(im[1,], axis=0)

    color_ranges = [[100,200],[100,200],[100,200]]
    cms.save_image_stack(filename.replace('.dat', '.mp4'), image_stack, [2, 2, 2], color_ranges=color_ranges)

    # projection = np.max(ims[0,1,], axis=0)
    # plt.imshow(projection, cmap = 'gray', vmin = 100, vmax = 200)
    # plt.show()

    print('Done!')
