from datetime import timedelta, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype('arial', 25)


def read_dat(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]

    df = pd.DataFrame()
    for filename in filenames:
        if '.dat' not in filename:
            filename = filename.split('.')[0] + '.dat'
        new_df = pd.read_csv(filename, delimiter='\t')
        new_df['Filenr'] = [rf'{Path(filename).parent}\Image{int(i)}.tiff' for i in new_df['Filenr']]
        new_df.rename(columns={'Filenr': 'Filename'}, inplace=True)

        settings = read_log(filename)
        new_df['Time (s)'] += settings['T0 (ms)']
        utc_time = datetime(1904, 1, 1) + timedelta(seconds=settings['T0 (ms)'])
        df = df.append(new_df)

    # corrections
    df = df[[Path(row['Filename']).exists() for _, row in df.iterrows()]]
    df.sort_values(by=['Time (s)'], inplace=True)
    df['Time (s)'] -= df['Time (s)'].min()

    return df.reset_index()


def read_log(filename):
    if '.log' not in filename:
        filename = filename.split('.')[0] + '.log'
    keys = ['LED (V)', 'Pixel size (um)', 'Z step (um)', 'Magnification', 'Exposure (s)', 'Wavelength (nm)', 'T0 (ms)']
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


def add_time_bar(img, t, t_max, progress_text=None):
    if img.mode == 'RGB':
        bg_color = (255, 255, 255)
    else:
        bg_color = 255

    size_pix = np.asarray(img.size)
    img_draw = ImageDraw.Draw(img)

    box = (5, size_pix[0] - 5, size_pix[0] - 5, size_pix[1] - 10)
    bar = (5, size_pix[0] - 5, 5 + (size_pix[0] - 10) * t / t_max, size_pix[1] - 10)
    img_draw.rectangle(box, outline=bg_color)
    img_draw.rectangle(bar, outline=bg_color, fill=bg_color)

    if progress_text is not None:
        img_draw.text((5, size_pix[1] - 40), f'{progress_text} = {timedelta(seconds=t//1)}', fill=bg_color, font=font)
    return


def add_scale_bar(img, pix_um, scale=1, barsize_um=10):
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
    # filename_out = str(Path(filename_out))

    ims = []
    stack = []
    last_slice = df['Slice'].max()
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
                add_scale_bar(image, 0.2)
                add_time_bar(image, row['Time (s)'] - df['Time (s)'].min(), df['Time (s)'].max() - df['Time (s)'].min(),
                             progress_text='t')
                ims.append(image)
                i += 1

    ims[0].save(filename_out, save_all=True, append_images=ims, duration=1000 / fps, loop=0)
    return filename_out


filename = r'C:\Data\noort\210316\data_023\data_023.dat'
filename = r'C:\tmp\data_023\data_023.dat'
filenames = [
    r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_018\data_018.dat'
    # ,r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_017\data_017.dat'
    , r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_019\data_019.dat'
    # , r'D:\Data\Noort\2photon\210325 time_lapse_pollen\data_020\data_020.dat'
]

df = read_dat(filenames)
# plt.plot(df['Time (s)'], df['Tile'], 'o-', fillstyle=None)
# plt.show()
# print(df.columns)


# settings = read_log(filename)
# for key in settings:
#     print(f'{key} = {settings[key]}')


foldername = Path(df['Filename'].iloc[0]).parent.parent
for tile in set(df['Tile'].astype(int)):
    filename_out = (rf'{foldername}\tile{tile}.gif')
    print(frames_to_gif(filename_out, select_frames(df, tile=tile, slice=range(2, 20)), max_intensity=True))
