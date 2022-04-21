import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from imaris_ims_file_reader.ims import ims
from scipy import fftpack
from tqdm import tqdm

from ProcessImages.ImageIO import create_circular_mask
from ProcessImages.ImageIO2 import merge_rgb_images, scale_u8


def scale_image_u8(image_array, z_range=None):
    if z_range is None:
        z_range = [-2 ** 15, 2 ** 15]
    image_array = np.uint8(np.clip((255 * (image_array - z_range[0]) / (z_range[1] - z_range[0])), 0, 255))
    return image_array


def find_peaks(image_array, width=20, scale=1, treshold_sd=3, n_traces=20, range=None, file_out="test.png", show=False):
    print('Finding peaks ...')
    if range is None:
        range = [np.median(image_array) - np.std(image_array), np.median(image_array) + 10 * np.std(image_array)]
    if show:
        image_out = Image.fromarray(scale_image_u8(image_array, range))
        image_out = image_out.convert("RGBA")
        image_draw = ImageDraw.Draw(image_out)
        font = ImageFont.truetype('arial', 20 * scale)
        text_position = 0.3 * scale * width * np.ones(2)
        circle_position = scale * np.asarray([-width, -width, width, width]) / 2

    max = np.max(image_array)
    median = np.median(image_array)
    treshold = np.median(image_array) + treshold_sd * np.std(image_array)
    trace_i = 0
    pos = []
    while max > treshold and trace_i < n_traces:
        max_index = np.asarray(np.unravel_index(np.argmax(image_array, axis=None), image_array.shape))
        mask = create_circular_mask(width, np.shape(image_array), max_index)
        image_array = mask * median + (1 - mask) * image_array
        max = np.max(image_array)
        max_index = np.flip(max_index) * scale
        if show:
            image_draw.text(text_position + max_index, f'{trace_i}', fill=(255, 0, 0, 255), font=font)
            image_draw.ellipse(list(np.append(max_index, max_index) + circle_position), outline=(255, 0, 0, 255))
            Image.fromarray(scale_image_u8(image_array, range)).show()
        trace_i += 1
        # print(trace_i)
        pos.append(max_index)
    if show:
        image_out.save(file_out)
    return np.asarray(pos)


def filter_image(image, highpass=None, lowpass=None, show=False):
    size = len(image[0])
    x = np.outer(np.linspace(-size / 2, size / 2, size), np.ones(size)) - 0.5
    y = x.T
    r = (x ** 2 + y ** 2) ** 0.5
    filter = np.ones_like(image).astype(float)

    if highpass is not None:
        filter *= 1 / (1 + 2 ** (6 * (highpass - r)))  # Butterworth filter
    if lowpass is not None:
        filter *= np.exp(-(r / (2 * lowpass)) ** 2)  # Gaussian filter

    im_fft = fftpack.fft2(image)
    im_fft = fftpack.fftshift(im_fft)
    im_fft *= filter

    im_fft = fftpack.fftshift(im_fft)
    image = np.real(fftpack.ifft2(im_fft)).astype(float)

    if show:
        plt.imshow(filter)
        plt.show()

    return image


def filter_image_stack(image_stack, highpass=5, low_pas=None):
    print('Filtering ....')
    shape = np.shape(image_stack)
    image_stack = np.reshape(image_stack, (-1, shape[-2], shape[-1]))
    image_stack = [filter_image(image, highpass) for image in image_stack]
    image_stack = [image - np.percentile(image, 50) for image in image_stack]
    image_stack = np.reshape(np.asarray(image_stack), shape)
    return image_stack


def get_background(image):
    background = image[image < 0]
    background = np.random.choice([-1, 1], len(background))
    std = np.std(background)
    return std


def get_traces(image_stack, coords, radius):
    all_traces = []
    shape = np.shape(image_stack)
    shape = (shape[0], shape[-2], shape[-1])
    for i, c in enumerate(tqdm(coords)):
        mask = create_circular_mask(radius * 2, size=shape[-2:], center=c).T
        traces = []
        for c in range(np.shape(image_stack)[1]):
            traces.append([np.sum(image * mask) for image in np.reshape(image_stack[:, c, :, :, :], shape)])
        all_traces.append(traces)
    return np.asarray(all_traces)


def cut_roi(image_stack, roi_width, center=None):
    if center is None:
        roi_center = np.asarray(image_stack.shape[-2:]) // 2
    roi_corners = np.asarray([roi_center - roi_width // 2])
    roi_corners = np.append(roi_corners, roi_corners + roi_width)
    image_stack = image_stack[:, :, :, roi_corners[0]:roi_corners[2], roi_corners[1]:roi_corners[3]]
    return image_stack


def save_image_stack(filename, image_stack, channels, frames=None, peaks=None, radius=10, numbers=False, fps=2):
    if frames is None:
        frames = range(image_stack.shape[0])
    im_rgb = np.zeros([3, image_stack.shape[-2], image_stack.shape[-1]])
    image_colors = {'637': 0, '561': 1, '488': 2}

    ims = []
    for frame in frames:
        for color, channel in enumerate(channels):
            image = image_stack[frame, color, 0, :, :]
            std = get_background(image)
            im_rgb[image_colors[channel]] = scale_u8(image, [-std, 20 * std])
        im = merge_rgb_images(im_rgb[0] * 0, im_rgb[0], im_rgb[1], im_rgb[2])

        for i, coord in enumerate(peaks):
            bbox = (coord[0] - radius, coord[1] - radius, coord[0] + radius, coord[1] + radius)
            draw = ImageDraw.Draw(im)
            draw.ellipse(bbox, fill=None, outline='white')
            bbox = list(np.clip(coord + radius / np.sqrt(2), 0, np.shape(image_stack)[-1]))
            if frame == frames[0]:
                draw.text(bbox, str(i), color='white')
        del draw
        ims.append(im)
    save_cv2_movie(filename, ims, fps)


def save_cv2_movie(filename, ims, fps):
    ext = filename[-3:]
    frames = range(np.shape(ims)[0])
    if ext == 'jpg':
        for frame, im in zip(frames, ims):
            im.save(filename.replace('.jpg', f'_{frame}.jpg'))
    elif ext == 'gif':
        ims[0].save(filename, save_all=True, append_images=ims, duration=1000 / fps, loop=0)
    elif ext in ['avi', 'mp4']:
        height, width, layers = np.asarray(ims[0]).shape
        size = (width, height)
        if ext == 'avi':
            codec = 'DIVX'
        elif ext == 'mp4':
            codec = 'mp4v'
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec), fps, size)
        for im in ims:
            out.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        out.release()
    else:
        print('Filetype {ext} not supported.')


def save_traces(filename, traces, header):
    ext = filename[-3:]
    if ext in ['mp4', 'jpg']:
        plot_colors = {'637': 'r', '561': 'g', '488': 'b'}
        ims = []
        for trace_nr, trace in enumerate(traces):
            fig = plt.figure()
            for i, channel in enumerate(header['colors']):
                plt.plot(header['time'], trace[i], color=plot_colors[channel], label=f'{channel} nm')
                # plt.scatter(header['time'], trace[i], color="none", edgecolor=plot_colors[channel], label=f'{channel} nm')
            plt.legend(loc="upper right")
            plt.xlabel('Time (s)')
            plt.ylabel('Intensity (a.u.)')
            plt.title(f'Peak {trace_nr} @{peaks[trace_nr][0]}, {peaks[trace_nr][1]}')
            plt.ylim(-200, 2500)
            plt.show()

            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = np.reshape(img, fig.canvas.get_width_height()[::-1] + (3,))
            ims.append(img)
        if ext == 'mp4':
            save_cv2_movie(filename.replace('.ims', '_traces.mp4'), ims, 1)
        else:
            for frame, im in enumerate(ims):
                im.save(filename.replace('.jpg', f'_{frame}.jpg'))
    elif ext == 'csv':
        column_names = []
        for trace_nr, trace in enumerate(traces):
            for i, channel in enumerate(header['colors']):
                column_names.append(f'Peak {trace_nr}: {channel} nm')
        df = pd.DataFrame(np.reshape(traces, (-1, np.shape(traces)[-1])).T, columns=column_names)
        df['Time (s)'] = header['time']
        df.set_index(['Time (s)'], inplace=True)
        df.to_csv(filename.replace('ims', 'csv'))


def read_header(filename):
    header = {}
    with h5py.File(filename, 'r') as hdf:
        channels = []
        for channel in range(6):
            try:
                channels.append(
                    hdf[rf'DataSetInfo/Channel {channel}'].attrs['LSMExcitationWavelength'][:3].tobytes().decode())
            except KeyError:
                header['colors'] = channels
        header['nm_pix'] = 1000 * np.abs(float(hdf[rf'DataSetInfo/Image'].attrs['ExtMax0'].tobytes().decode()) \
                                         - float(hdf[rf'DataSetInfo/Image'].attrs['ExtMin0'].tobytes().decode())) / \
                           float(hdf[rf'DataSetInfo/Image'].attrs['X'].tobytes().decode())
        header['time'] = [t[1] / 1e10 for t in np.asarray(hdf[rf'DataSetTimes/Time'])]
    return header


if __name__ is '__main__':
    # filename = r'C:\Users\noort\Downloads\Plexi_Channel2_FOV4_dsDNAPol1_2022-03-21_Protocol 4_18.56.31.ims'
    # filename = r'C:\Users\noort\Downloads\Slide2_Channel1_FOV2_512_Int100_Exp50_Rep05_Pol1_2022-04-06_Protocol 2_16.32.11.ims'
    filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV13_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.29.35.ims'
    # filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV14_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.38.58.ims'

    header = read_header(filename)
    # print(header['colors'])
    # header['colors'] = ['561', '488', '637'] # overrule header
    # print(header['colors'])

    if True:
        image_stack = ims(filename)
        image_stack = cut_roi(image_stack, roi_width=512)
        image_stack = filter_image_stack(image_stack, highpass=15)

        radius = 300 / header['nm_pix']
        selection_image = np.zeros_like(image_stack[0, 0, 0, :, :])
        for color in ['488', '637']:
            selection_image += np.percentile(image_stack[:, header['colors'].index(color), 0, :, :], 70, axis=0)
        selection_image = filter_image(selection_image, lowpass=40)

        peaks = find_peaks(selection_image, radius * 2, n_traces=10000, treshold_sd=3.5)
        save_image_stack(filename.replace('.ims', '.mp4'), image_stack, header['colors'], peaks=peaks, radius=radius)

        traces = get_traces(image_stack, peaks, radius)
        save_traces(filename.replace('.ims', '.csv'), traces, header)
        save_traces(filename.replace('.ims', '_traces.mp4'), traces, header)
