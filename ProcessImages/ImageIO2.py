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

        try:
            settings = read_log(filename)
            new_df['Time (s)'] += settings['T0 (ms)']
            utc_time = datetime(1904, 1, 1) + timedelta(seconds=settings['T0 (ms)'])
        except KeyError:
            pass
        df = df.append(new_df)

    # corrections
    df = df[[Path(row['Filename']).exists() for _, row in df.iterrows()]]
    df.sort_values(by=['Time (s)'], inplace=True)
    df['Time (s)'] -= df['Time (s)'].min()

    return df.reset_index(drop=True)


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


def select_frames(df, slice=None, serie=None, tile=None, frame=None):
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
    if frame is not None:
        if isinstance(frame, int):
            frame = [frame]
        df = df[df['Frame'].isin(frame)]
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
        img_draw.text((5, size_pix[1] - 40), f'{progress_text} = {timedelta(seconds=t // 1)}', fill=bg_color, font=font)
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


def scale_u8(array, z_range=None):
    if z_range is None:
        z_range = np.asarray([np.percentile(array, 5), 1.5 * np.percentile(array, 95)])

    return np.uint8(np.clip((255 * (array - z_range[0]) / (z_range[1] - z_range[0])), 0, 255))


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
                image = Image.fromarray(scale_u8(frame))
                add_scale_bar(image, 0.2)
                add_time_bar(image, row['Time (s)'] - df['Time (s)'].min(), df['Time (s)'].max() - df['Time (s)'].min(),
                             progress_text='t')
                ims.append(image)
                i += 1

    ims[0].save(filename_out, save_all=True, append_images=ims, duration=1000 / fps, loop=0)
    return filename_out


def replace_roi(image, roi, center, max_intensity=True):
    x = ll = 0
    y = ur = 1

    def to_corners(center, width):
        corners = [[int(c - w // 2), int(c - w // 2) + int(w)] for c, w in zip(center, width)]
        return corners

    def clip_edges(image, corners, roi):
        if corners[x][0] < 0:
            roi = roi[-corners[x][0]:, :]
            corners[x][0] = 0
        if corners[y][0] < 0:
            roi = roi[:, -corners[y][0]:]
            corners[y][0] = 0
        image_size = np.shape(image)
        if corners[x][1] > image_size[x]:
            roi = roi[0:image_size[x] - corners[x][1]:, :]
            corners[x][1] = image_size[x]
        if corners[y][1] > image_size[y]:
            roi = roi[:, 0:image_size[y] - corners[y][1]]
            corners[y][1] = image_size[y]
        return corners, roi

    def get_roi(image, corners):
        return image[corners[x][ll]:corners[x][ur], corners[y][ll]:corners[y][ur]]

    corners = to_corners(center, np.shape(roi))
    corners, roi = clip_edges(image, corners, roi)

    if max_intensity:
        old_roi = get_roi(image, corners)
        roi[old_roi > roi] = old_roi[old_roi > roi]

    image[corners[x][ll]:corners[x][ur], corners[y][ll]:corners[y][ur]] = roi
    return image


def stitch_mosaic(df, zrange=None):
    pixel_size = read_log(filenames[0])["Pixel size (um)"]
    image_size = np.asarray(np.shape(plt.imread(Path(df['Filename'].iloc[0]))))

    for i, axe in enumerate(['x', 'y']):
        df[f'{axe} (pix)'] = df[f'Stepper {axe} (um)'] / pixel_size
        df[f'{axe} (pix)'] -= df[f'{axe} (pix)'].min()
        df[f'{axe} (pix)'] += image_size[i] / 2

    mosaic = np.zeros(np.asarray((df['x (pix)'].max(), df['y (pix)'].max())).astype(int) + image_size // 2,
                      dtype=np.uint8)

    print(f'Stitching tiles ...')
    print(f'n tiles  {df.shape[0]}')
    print(f'pixel size (um) {pixel_size}')
    print(f'mosaic_size (pix) {np.shape(mosaic)}')

    range = pixel_size * np.asarray(np.shape(mosaic))
    extent = [df['Stepper x (um)'].min() - pixel_size * image_size[0] / 2,
              df['Stepper x (um)'].max() + pixel_size * image_size[0] / 2,
              df['Stepper y (um)'].min() - pixel_size * image_size[1] / 2,
              df['Stepper y (um)'].max() + pixel_size * image_size[1] / 2]
    extent = np.asarray(extent)

    for i, row in df.iterrows():
        if zrange is None:
            image = plt.imread(Path(row['Filename']))
            zrange = [np.percentile(image, 5), 2 * np.percentile(image, 90)]
        roi = scale_u8(plt.imread(Path(row['Filename'])), zrange)
        mosaic = replace_roi(mosaic, roi, [row['x (pix)'], row['y (pix)']])

    return mosaic, extent


filename = r'C:\Data\noort\210316\data_023\data_023.dat'
filename = r'C:\tmp\data_023\data_023.dat'
filenames = [
    r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_018\data_018.dat'
    # ,r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_017\data_017.dat'
    , r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_019\data_019.dat'
    # , r'D:\Data\Noort\2photon\210325 time_lapse_pollen\data_020\data_020.dat'
]

filenames = [r'C:\Users\jvann\Downloads\data_002.dat']

filenr = '001'
filenames = [r'D:\Data\Noort\2photon\210406_grid_test\data_002\data_002.dat']
filenames = [rf'C:\Data\noort\210422\data_{filenr}\data_{filenr}.dat']



df = read_dat(filenames)

if 0:
    # read settings
    settings = read_log(filenames[0])
    for key in settings:
        print(f'{key} = {settings[key]}')
if 0:
    # make gif
    foldername = Path(df['Filename'].iloc[0]).parent.parent
    for tile in set(df['Tile'].astype(int)):
        filename_out = (rf'{foldername}\tile{tile}.gif')
        print(frames_to_gif(filename_out, select_frames(df, tile=tile, slice=range(2, 20)), max_intensity=True))

if 1:
    # stitch mosaic
    mosaic, extent = stitch_mosaic(select_frames(df, frame=0, slice=0))
    if mosaic is not None:
        plt.imshow(mosaic.T, origin='lower', cmap='gray', extent=extent)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        plt.show()
