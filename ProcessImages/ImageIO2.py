from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype('arial', 25)


def read_dat(filename):
    if '.dat' not in filename:
        filename = filename.split('.')[0] + '.dat'
    df = pd.read_csv(filename, delimiter='\t')
    df['Filenr'] = [rf'{Path(filename).parent}\Image{int(i)}.tiff' for i in df['Filenr']]
    df.rename(columns={'Filenr': 'Filename'}, inplace=True)
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
        if isinstance(slice, int):
            slice = [slice]
        df = df[df['Slice'].isin(slice)]
    if serie is not None:
        if isinstance(serie, int):
            serie = [serie]
        df = df[df['Serie'].isin(serie)]
    if tile is not None:
        if isinstance(tile, int):
            tile = [tile]
        df = df[df['Tile'].isin(tile)]
    return df


def get_filenames(dat_file, df):
    filenames = []
    for _, row in df.iterrows():
        filenames.append(str(Path(f'{Path(dat_file).parent}/image{int(row["Filenr"])}.tiff')))
    return filenames


def add_time_bar(img, i, n, progress_text=None, progress_step=None):

    if img.mode == 'RGB':
        bg_color = (255, 255, 255)
    else:
        bg_color = 255

    size_pix = np.asarray(img.size)
    img_draw = ImageDraw.Draw(img)

    box = (5, size_pix[0] - 5, size_pix[0] - 5, size_pix[1] - 10)
    bar = (5, size_pix[0] - 5, 5 + (size_pix[0] - 10) * i / (n - 1), size_pix[1] - 10)
    img_draw.rectangle(box, outline=bg_color)
    img_draw.rectangle(bar, outline=bg_color, fill=bg_color)

    if progress_text is not None:
        img_draw.text((5, size_pix[1] - 40), f'{progress_text} = {progress_step * i:4.1f}', fill=bg_color, font=font)
    return


def add_scale_bar(img, pix_um, scale=1, barsize_um=5):
    if img.mode == 'RGB':
        bg_color = (255, 255, 255)
    else:
        bg_color = 255

    size_pix = np.asarray(img.size)
    pix_um /= scale
    img_draw = ImageDraw.Draw(img)
    if pix_um is not None:
        bar = (size_pix[0] - 5 - barsize_um / pix_um, size_pix[1] - 20, size_pix[0] - 5, size_pix[1] - 25)
        img_draw.rectangle(bar, fill=bg_color)
        img_draw.text((size_pix[0] - 5 - barsize_um / pix_um - 85, size_pix[1] - 40),
                      f'{barsize_um:3d} um', fill=bg_color, font=font)
    return


def frames_to_gif(filename_out, df, fps=5, z_range=None, max_intensity=False):
    filename_out = str(Path(f'{Path(df["Filename"].iloc[0]).parent}/{filename_out}'))

    ims = []
    stack = []
    last_slice = df['Slice'].max()
    tmp = select_frames(df, slice=int(last_slice))
    i = 0
    for c, row in tmp.iterrows():
        print(i, row['Slice'],row['Filename'])
        i+=1

    n_frames = len(tmp)
    i = 0

    for _, row in df.iterrows():
        filename = row['Filename']
        if Path(filename).is_file():
            frame = np.asarray(plt.imread(filename)).T.astype(float)

            if max_intensity:
                stack.append(frame)
                if row['Slice'] == last_slice:
                    frame = np.max(np.asarray(stack), axis=0)
                    stack = []
                else:
                    frame = None

            if frame is not None:
                if z_range is None:
                    z_range = [np.percentile(frame, 2), 1.5 * np.percentile(frame, 99)]
                frame = np.uint8(np.clip((255 * (frame - z_range[0]) / (z_range[1] - z_range[0])), 0, 255))
                image = Image.fromarray(frame)
                add_scale_bar(image, 0.1)
                add_time_bar(image, i, n_frames,progress_text='t (s)', progress_step=row['Time (s)'])
                ims.append(image)
                frame = None
                i += 1

    ims[0].save(filename_out, save_all=True, append_images=ims, duration=1000 / fps, loop=0)


filename = r'C:\Data\noort\210316\data_023\data_023.dat'
filename = r'C:\tmp\data_023\data_023.dat'
filename = r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_019\data_019.dat'

df = read_dat(filename)
# plt.plot(df['Time (s)'], df['Tile'], 'o-', fillstyle=None)
# plt.show()
# print(df.columns)

settings = read_log(filename)
# for key in settings:
#     print(f'{key} = {settings[key]}')


df = select_frames(df, tile=1, slice=range(2, 20))

frames_to_gif('test.gif', df, max_intensity=True)
