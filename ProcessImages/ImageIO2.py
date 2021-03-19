from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def read_dat(filename):
    if '.dat' not in filename:
        filename = filename.split('.')[0] + '.dat'
    df = pd.read_csv(filename, delimiter='\t')
    return df


def read_log(filename):
    if '.log' not in filename:
        filename = filename.split('.')[0] + '.log'
    keys = ['LED (V)', 'Pixel size (um)', 'Z step (um)', 'Magnification', 'Exposure (s)', 'Wavelength (nm)']
    settings = {}
    with open(filename) as reader:
        for line in reader:
            for key in keys:
                if key in line:
                    settings[key] = float(line.split('=')[-1])
    return settings


def select_frames(df, slice=None, serie=None, tile=None):
    if slice is not None:
        df = df[df['Slice'] == slice]
    if serie is not None:
        df = df[df['Serie'] == serie]
    if tile is not None:
        df = df[df['Tile'] == tile]
    return df


def get_filenames(dat_file, df):
    filenames = []
    for _, row in df.iterrows():
        filenames.append(str(Path(f'{Path(dat_file).parent}/image{int(row["Filenr"])}.tiff')))
    return filenames


def frames_to_gif(filename_out, filenames_in, fps=5, z_range=None):
    filename_out = str(Path(f'{Path(filenames_in[0]).parent}/{filename_out}'))
    ims = []
    for filename in filenames_in:
        if Path(filename).is_file():
            frame = np.asarray(plt.imread(filename)).T.astype(float)
            if z_range is None:
                z_range = [np.percentile(frame, 5), 1.5 * np.percentile(frame, 95)]
            frame = np.uint8(np.clip((255 * (frame - z_range[0]) / (z_range[1] - z_range[0])), 0, 255))
            ims.append(Image.fromarray(frame))
    ims[0].save(filename_out, save_all=True, append_images=ims, duration=1000 / fps, loop=0)


filename = r'C:\Data\noort\210316\data_023\data_023.dat'
filename = r'C:\tmp\data_023\data_023.dat'

df = read_dat(filename)
# plt.plot(df['Time (s)'], df['Tile'], 'o-', fillstyle=None)
# plt.show()
# print(df.columns)

settings = read_log(filename)
# for key in settings:
#     print(f'{key} = {settings[key]}')

print(df.columns)
df = select_frames(df, tile=1)
frame_files = get_filenames(filename, df)
frames_to_gif('test.gif', frame_files)
