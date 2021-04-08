from datetime import timedelta, datetime
from pathlib import Path

from microscopestitching import stitch

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
                # frame = np.uint8(np.clip((255 * (frame - z_range[0]) / (z_range[1] - z_range[0])), 0, 255))
                image = Image.fromarray(scale_u8(frame))
                add_scale_bar(image, 0.2)
                add_time_bar(image, row['Time (s)'] - df['Time (s)'].min(), df['Time (s)'].max() - df['Time (s)'].min(),
                             progress_text='t')
                ims.append(image)
                i += 1

    ims[0].save(filename_out, save_all=True, append_images=ims, duration=1000 / fps, loop=0)
    return filename_out


def replace_roi(image, roi, center):
    def _check_boundaries(i_size, i_coords, r_coords):
        for axis in [0, 1]:
            if i_coords[axis][0] < 0:
                r_coords[axis][0] -= i_coords[axis][0]
                i_coords[axis][0] = 0
            if i_coords[axis][1] > i_size[axis]:
                r_coords[axis][1] -= i_coords[axis][1] - i_size[axis]
                i_coords[axis][1] = i_size[axis] - 1
        return i_coords.astype(int), r_coords.astype(int)

    def _get_roi(image, r_coords):
        roi = image[r_coords[0][0]:r_coords[0][1], r_coords[1][0]:r_coords[1][1]]
        return roi

    center = np.asarray(center)
    roi_size = np.asarray(np.shape(roi))

    r_coords = np.asarray([np.zeros(2), np.shape(roi)]).T
    i_coords = np.asarray(center - roi_size / 2)
    i_coords = np.append(i_coords, i_coords + roi_size)
    i_coords = np.reshape(i_coords, (2, 2)).T

    i_coords, r_coords = _check_boundaries(np.shape(image), i_coords, r_coords)
    try:
        image[i_coords[0][0]:i_coords[0][1], i_coords[1][0]:i_coords[1][1]] = \
            roi[r_coords[0][0]:r_coords[0][1], r_coords[1][0]:r_coords[1][1]]
    except ValueError:
        print(f'center: {center}, imagesize: {np.shape(image)}, roisize: {np.shape(roi)}')
        # print(f'r_coords: {r_coords}')
        # print(f'i_coords: {i_coords}')
        print(f'roi size: {np.shape(_get_roi(roi, r_coords))}')
        print(f'image size: {np.shape(_get_roi(image, i_coords))}')

    return image


def stitch_mosacic(df, z_range=None):
    correction = 2
    pixel_size = read_log(filenames[0])["Pixel size (um)"]
    image_size = np.asarray(np.shape(plt.imread(Path(df['Filename'].iloc[0]))))

    xy_range = np.asarray([df["Stepper x (um)"].max() - df["Stepper x (um)"].min(), \
                           df["Stepper y (um)"].max() - df["Stepper y (um)"].min()])
    xy_range *= correction
    xy_range = np.asarray(xy_range) + pixel_size * image_size
    xy_range = np.uint16(xy_range / pixel_size)

    origin = np.asarray([df["Stepper x (um)"].min(), df["Stepper y (um)"].min()])
    origin *= correction

    print(f'xy_range (pix) = {xy_range}')
    print(f'image_size (pix) = {image_size}')


    return 0

    mosaic = np.zeros(2 * xy_range, dtype=np.uint8)
    for _, row in df.iterrows():
        if Path(row['Filename']).is_file():
            tile = np.asarray(plt.imread(row['Filename'])).T.astype(float)
            if z_range is None:
                z_range = np.asarray([0.5 * np.percentile(tile, 5), 1.5 * np.percentile(tile, 95)])
            tile = np.transpose(scale_u8(tile, z_range=z_range))

            offset = np.asarray((row["Stepper x (um)"], row["Stepper y (um)"]))
            offset -= origin
            print(offset)
            mosaic = replace_roi(mosaic, tile, offset / pixel_size)

    return mosaic


filename = r'C:\Data\noort\210316\data_023\data_023.dat'
filename = r'C:\tmp\data_023\data_023.dat'
filenames = [
    r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_018\data_018.dat'
    # ,r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_017\data_017.dat'
    , r'D:\Data\Noort\2photon\210324 time_lapse_pollen\data_019\data_019.dat'
    # , r'D:\Data\Noort\2photon\210325 time_lapse_pollen\data_020\data_020.dat'
]

filenames = [r'C:\Users\jvann\Downloads\data_002.dat']

df = read_dat(filenames)
# plt.plot(df['Time (s)'], df['Tile'], 'o-', fillstyle=None)
# plt.show()
# print(df.columns)
df = df[df['Frame'] == 0]
df = df[df['Stepper y (um)'].between(-1800, -1700)]
# df = df[:2]

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
    mosaic = stitch_mosacic(df)
    if mosaic:
        plt.imshow(mosaic.T, origin='lower', cmap='Greys')
        plt.show()

if 0:
    image = replace_roi(np.zeros((100, 100)), np.ones((10, 20)), [35, 50])
    plt.imshow(image.T, origin='lower', cmap='Greys')
    plt.show()
