from datetime import timedelta, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as sfft
import tqdm, cv2
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp

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
        df = df.append(new_df, sort=False)

    # corrections
    df = df[[Path(row['Filename']).exists() for _, row in df.iterrows()]]  # only existing files
    df.sort_values(by=['Time (s)'], inplace=True)  # ensure chronological order
    df['Time (s)'] -= df['Time (s)'].min()  # set t = 0 for first frame
    offset = (df['Frame'] + 1) * (np.diff(df['Frame'], append=0) < 0)
    offset = np.cumsum(offset)
    offset = np.roll(offset, 1)
    offset[0] = 0
    df['Frame'] += offset  # Correct for duplicate frame numbers in different files

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


def read_image(filename):
    return plt.imread(Path(filename)).astype(float).T


def select_frames(df, slice=None, serie=None, tile=None, frame=None, folder=None):
    if slice is not None:
        if not isinstance(slice, list):
            slice = [int(slice)]
        df = df[df['Slice'].isin(slice)]
    if serie is not None:
        if not isinstance(serie, list):
            serie = [int(serie)]
        df = df[df['Serie'].isin(serie)]
    if tile is not None:
        if not isinstance(tile, list):
            tile = [int(tile)]
        df = df[df['Tile'].isin(tile)]
    if frame is not None:
        if not isinstance(frame, list):
            frame = [int(frame)]
        df = df[df['Frame'].isin(frame)]
    if folder is not None:
        if not isinstance(folder, list):
            folder = [int(folder)]
        selected = np.zeros(len(df))
        for f in folder:
            selected = np.logical_or(selected, df['Filename'].str.contains(fr'data_{f:03d}'))
        df = df[selected]
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


def add_text(img, txt):
    if img.mode == 'RGB':
        bg_color = (255, 255, 255)
    else:
        bg_color = 255
    img_draw = ImageDraw.Draw(img)
    img_draw.text((5, 5), f'{txt}', fill=bg_color, font=font)


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


def stacks_to_gif(filename, df, tile, max_intensity_range=None, transmission_range=None, fps=5):
    ims = []
    for frame in tqdm.tqdm(df['Frame'].unique(), desc=filename):
        df_stack = select_frames(df, slice=0, tile=tile, frame=frame)
        if not df_stack.empty:
            max_intensity = get_max_intensity_image(df, frame=frame, tile=tile)
            transmission = read_image(df_stack.iloc[0]['Filename'])

            if max_intensity_range is None:
                max_intensity_range = [np.percentile(max_intensity, 2), 2 * np.percentile(max_intensity, 99)]
            if transmission_range is None:
                transmission_range = [0.3*np.percentile(transmission, 2), 3*np.percentile(transmission, 60)]

            transmission = scale_u8(transmission, transmission_range)
            max_intensity = scale_u8(max_intensity, max_intensity_range)

            im = merge_rgb_images(transmission, red=max_intensity)

            # add_text(im, f"Wavelength = {df_stack.iloc[0]['Wavelength (nm)']:.0f} nm")
            text = df_stack.iloc[0]['Filename']
            add_text(im, f"{text[len(str(Path(text).parent.parent.parent)):]}")
            add_scale_bar(im, 0.13)
            add_time_bar(im, df_stack.iloc[0]['Time (s)'], df['Time (s)'].max(), progress_text='t')

            ims.append(im)

    ims[0].save(filename, save_all=True, append_images=ims, duration=1000 / fps, loop=0)
    return filename


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


def merge_rgb_images(grey, red=None, green=None, blue=None):
    if red is None:
        red = grey
    else:
        red = scale_u8(red.astype(int) + grey.astype(int), z_range=[0, 255])

    if green is None:
        green = grey
    else:
        green = scale_u8(green.astype(int) + grey.astype(int), z_range=[0, 255])

    if blue is None:
        blue = grey
    else:
        blue = scale_u8(blue.astype(int) + grey.astype(int), z_range=[0, 255])

    im = Image.merge('RGB', (Image.fromarray(red), Image.fromarray(green), Image.fromarray(blue)))

    return im


def get_max_intensity_image(df, frame, tile=0, serie=0):
    df = select_frames(df, serie=serie, tile=tile, frame=frame)
    if df['Photodiode (mW)'].min() < 0.5 * df['Photodiode (mW)'].mean():
        df = df[df['Photodiode (mW)'].ne(df['Photodiode (mW)'].min())]
    stack = []
    for _, row in df.iterrows():
        stack.append(read_image(row['Filename']))
    return np.max(np.asarray(stack), axis=0)


def get_spectrum(df, show=False):
    intensity = []
    for _, row in df.iterrows():
        im = read_image(row['Filename'])
        intensity.append(np.percentile(im, 95) - np.percentile(im, 10))

    intensity = np.asarray(intensity)
    intensity -= np.min(intensity)
    # intensity /= np.max(intensity)

    laser = df['Photodiode (mW)'].values
    laser -= np.min(laser)
    laser /= np.max(laser)
    if show:
        plt.plot(df['Wavelength (nm)'].values, intensity)
        plt.plot(df['Wavelength (nm)'].values, laser ** 2)
        plt.ylim((0, 1.2))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(df.iloc[0]['Filename'])
        plt.show()

    return df['Wavelength (nm)'].values, intensity


def get_drift(df, tile, vmax=None):
    def mask(im, min=0, max=None):
        N = np.shape(im)
        X, Y = np.meshgrid(np.linspace(0, N[0] - 1, N[0]) - N[0] / 2, np.linspace(0, N[1] - 1, N[1]) - N[1] / 2)
        R = np.sqrt(X ** 2 + Y ** 2)
        mask = np.asarray(R > min).astype(float)
        if max is not None:
            mask *= np.asarray(R < max).astype(float)
        mask /= np.sum(mask)
        return mask

    for frame in df['Frame'].unique():
        # for frame in [1,2,3,4,5]:
        df_stack = select_frames(df, tile=tile, frame=frame)
        if df_stack['Photodiode (mW)'].min() < 0.5 * df_stack['Photodiode (mW)'].mean():
            df_stack = df_stack[df_stack['Photodiode (mW)'].ne(df_stack['Photodiode (mW)'].min())]

        df_stack["r"] = np.nan
        for _, row in df_stack.iterrows():
            im = read_image(row['Filename'])
            im -= np.median(im)
            mask1 = mask(im, min=30, max=50)
            mask2 = mask(im, min=50, max=100)
            fft_im = sfft.fft2(im)
            fft_im = sfft.fftshift(fft_im)
            fft_im = np.abs(fft_im)
            r = np.abs(np.sum(fft_im * mask1)) / np.abs(np.sum(fft_im * mask2)) - 1
            df_stack.at[row.name, 'r'] = r

        focus = np.sum(df_stack['PiezoZ (um)'] * df_stack['r']) / np.sum(df_stack['r'])

        plt.plot(df_stack['PiezoZ (um)'], df_stack['r'], 'o')
        plt.title(f'frame: {frame}, focus = {focus:.1f} (um)')
        plt.ylim((0, 2))
        plt.show()

        # max_intensity = get_max_intensity_image(df, frame=frame, tile=tile)
        # transmission = read_image(df_stack.iloc[0]['Filename'])
        #
        # if max_intensity_range is None:
        #     max_intensity_range = [np.percentile(max_intensity, 2), 2 * np.percentile(max_intensity, 99)]
        # if transmission_range is None:
        #     transmission_range = [np.percentile(transmission, 2), 2 * np.percentile(transmission, 99)]
        #
        # transmission = scale_u8(transmission, transmission_range)
        # max_intensity = scale_u8(max_intensity, max_intensity_range)
        #
        # im = merge_rgb_images(transmission, red=max_intensity)
        #
        # # add_text(im, f"Wavelength = {df_stack.iloc[0]['Wavelength (nm)']:.0f} nm")
        # add_scale_bar(im, 0.13)
        # add_time_bar(im, df_stack.iloc[0]['Time (s)'], df['Time (s)'].max(), progress_text='t')
        #
        # ims.append(im)

def convert_gif_avi(filename, extension = 'avi'):
    clip = mp.VideoFileClip(filename)
    filename = filename.replace('gif', extension)
    if extension == 'avi':
        clip.write_videofile(filename, codec='libx264')
    elif extension == 'mp4':
        clip.write_videofile(filename)
    clip.close()
    print(f'converted: {filename}')

if __name__ == "__main__":
    dir = Path(r'D:\Data\Noort\2photon\210521 pollen')
    filenames = [str(f) for f in dir.glob('*\data_01[2,3,4,5,6,7,8,9]\data_*.dat')]
    # filenames = ['D:\\Data\\Noort\\2photon\\210521 pollen\\210521\\data_012\\data_012.dat', 'D:\\Data\\Noort\\2photon\\210521 pollen\\210521\\data_013\\data_013.dat']

    nrs = [10]
    # filenames = [rf'D:\Data\Noort\2photon\210423_first dataset pollen\data_{nr:03d}\data_{nr:03d}.dat' for nr in nrs]
    # filenames = [rf'{dir}\data_{nr:03d}\data_{nr:03d}.dat' for nr in nrs]

    filenames.sort()

    # for f in filenames:
    #     print(f)

    df = read_dat(filenames)
    # for i, row in df.iterrows():
    #     print(i, row['Filename'])

    df['Time (s)'] -= df['Time (s)'].min()

    # max_intensity = get_max_intensity_image(df, frame=2, tile=28)
    # plt.imshow(max_intensity, origin='lower')
    # plt.show()


    # i = get_max_intensity_image(df,frame = 0, tile = 10 )
    # plt.imshow(i)
    # plt.show()

    # filename = Path(df.iloc[0]['Filename'])
    # print(filename)
    # tmp = [f for f in filename.iterdir() if f.is_dir()]
    # print(tmp)

    if 0:
        for nr, color in zip(nrs, ['green', 'red']):
            w, s = get_spectrum(select_frames(select_frames(df, slice=1), folder=nr))
            plt.plot(w, s, color=color)
        # plt.ylim((0, 1.2))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend(('LEC1-GFP', 'DR5V2-tdTomato'))
        plt.show()

    if 0:
        # read settings
        settings = read_log(filenames[0])
        for key in settings:
            print(f'{key} = {settings[key]}')
    if 1:
        # make gif

        # df = select_frames(df, folder=[10,11,12,13])
        # for _, row in df.iterrows():
        #     print(row['Filename'])
        df['Time (s)'] -= df['Time (s)'].min()

        foldername = Path(df['Filename'].iloc[0]).parent.parent

        # for tile in set(df['Tile'].astype(int)):
        for tile in [1]:
            filename_out = (rf'{foldername}\Fluorescence_{tile}.gif')
            stacks_to_gif(filename_out, df, tile)

            convert_gif_avi(filename_out)

    if 0:
        # stitch mosaic
        mosaic, extent = stitch_mosaic(select_frames(df, frame=0, slice=0))
        if mosaic is not None:
            plt.imshow(mosaic.T, origin='lower', cmap='gray', extent=extent)
            plt.xlabel('x (um)')
            plt.ylabel('y (um)')
            plt.show()

    if 0:
        df['Time (s)'] -= df['Time (s)'].min()
        for tile in [1]:
            get_drift(df, tile)
