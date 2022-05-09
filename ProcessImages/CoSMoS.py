import cv2, io
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from imaris_ims_file_reader.ims import ims
from scipy import fftpack, ndimage
from tqdm import tqdm

from ProcessImages.ImageIO import create_circular_mask, get_roi, fit_peak
from ProcessImages.ImageIO2 import merge_rgb_images, scale_u8


def scale_image_u8(image_array, z_range=None):
    if z_range is None:
        z_range = [-2 ** 15, 2 ** 15]
    image_array = np.uint8(np.clip((255 * (image_array - z_range[0]) / (z_range[1] - z_range[0])), 0, 255))
    return image_array


def find_peaks(image_array, width=20, scale=1, treshold_sd=3, n_traces=2000, range=None, file_out="test.png", show=False):
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
    peak_i = 0
    peaks = []
    while max > treshold and peak_i < n_traces:
        max_index = np.asarray(np.unravel_index(np.argmax(image_array, axis=None), image_array.shape))
        mask = create_circular_mask(width, np.shape(image_array), max_index)
        image_array = mask * median + (1 - mask) * image_array
        max = np.max(image_array)
        max_index = np.flip(max_index) * scale
        if show:
            image_draw.text(text_position + max_index, f'{peak_i}', fill=(255, 0, 0, 255), font=font)
            image_draw.ellipse(list(np.append(max_index, max_index) + circle_position), outline=(255, 0, 0, 255))
            Image.fromarray(scale_image_u8(image_array, range)).show()
        peak_i += 1
        peaks.append(max_index)
    peaks = np.asarray(peaks)
    if peaks.any():
        peaks = peaks[peaks[:, 1].argsort()]

    if show:
        image_out.save(file_out)

    return peaks


def filter_image(image, highpass=None, lowpass=None):
    size = len(image[0])
    x = np.outer(np.linspace(-size / 2, size / 2, size), np.ones(size)) - 0.5
    y = x.T
    r = (x ** 2 + y ** 2) ** 0.5
    filter = np.ones_like(image).astype(float)

    if highpass is not None:
        filter *= 1 / (1 + 2 ** (6 * (size / highpass - r)))  # Butterworth filter
    if lowpass is not None:
        filter *= np.exp(-(r / (2 * size / lowpass)) ** 2)  # Gaussian filter

    im_fft = fftpack.fft2(image)
    im_fft = fftpack.fftshift(im_fft)
    im_fft *= filter

    im_fft = fftpack.fftshift(im_fft)
    image = np.real(fftpack.ifft2(im_fft)).astype(float)

    return image


def filter_image_stack(image_stack, highpass=None, lowpass=None, offset=True):
    shape = np.shape(image_stack)
    image_stack = np.reshape(image_stack, (-1, shape[-2], shape[-1]))
    image_stack = [filter_image(image, highpass, lowpass) for image in image_stack]
    if offset:
        image_stack = [image - np.percentile(image, 50) for image in image_stack]
    image_stack = np.reshape(np.asarray(image_stack), shape)
    return image_stack


def get_background(image):
    background = image[image < 0]
    background = np.random.choice([-1, 1], len(background))
    std = np.std(background)
    return std


def get_traces(image_stack, coords, radius, header):
    all_traces = []
    shape = np.shape(image_stack)
    shape = (shape[0], shape[-2], shape[-1])
    for i, c in enumerate(tqdm(coords, 'Getting traces')):
        mask = create_circular_mask(radius * 2, size=shape[-2:], center=c).T
        traces = []
        for c in range(np.shape(image_stack)[1]):
            traces.append([np.sum(image * mask) for image in np.reshape(image_stack[:, c, :, :, :], shape)])
        all_traces.append(traces)

    column_names = []
    for trace_nr, trace in enumerate(all_traces):
        for i, channel in enumerate(header['colors']):
            column_names.append(f'{trace_nr}: I{channel}nm (a.u.)')

    df = pd.DataFrame(np.reshape(all_traces, (-1, np.shape(traces)[-1])).T, columns=column_names)
    df['Time (s)'] = header['time']
    df.set_index(['Time (s)'], inplace=True)
    return df


def stack_cut_roi(image_stack, roi_width, center=None):
    if center is None:
        center = np.asarray(image_stack.shape[-2:]) // 2
    else:
        center = np.asarray(center)
    roi_corners = np.asarray([center - roi_width // 2])
    roi_corners = np.append(roi_corners, roi_corners + roi_width)
    image_stack = image_stack[:, :, :, roi_corners[0]:roi_corners[2], roi_corners[1]:roi_corners[3]]
    return image_stack


def save_image_stack(filename, image_stack, channels, frames=None, peaks=None, radius=10, color_ranges=None, fps=2):
    if frames is None:
        frames = range(image_stack.shape[0])
    im_rgb = np.zeros([3, image_stack.shape[-2], image_stack.shape[-1]])
    image_colors = {'637': 0, '561': 1, '488': 2}

    ims = []

    if color_ranges is None:
        color_ranges = []
        for color, _ in enumerate(channels):
            std = get_background(image_stack[0, color, 0, :, :])
            color_ranges.append([-std, 20 * std])

    for frame in frames:
        for color, (channel, color_range) in enumerate(zip(channels, color_ranges)):
            image = image_stack[frame, color, 0, :, :]

            im_rgb[image_colors[channel]] = scale_u8(image, color_range)
        im = merge_rgb_images(im_rgb[0] * 0, im_rgb[0], im_rgb[1], im_rgb[2])

        if peaks is not None:
            draw = ImageDraw.Draw(im)
            if peaks is not None:

                for i, coord in enumerate(peaks):
                    bbox = (coord[0] - radius, coord[1] - radius, coord[0] + radius, coord[1] + radius)
                    draw.ellipse(bbox, fill=None, outline='white')
                    bbox = list(np.clip(coord + radius / np.sqrt(2), 0, np.shape(image_stack)[-1]))
                    if frame == frames[0]:
                        draw.text(bbox, str(i), color='white')

            draw.text([10, 10], f'Frame {frame}', color='white')
            del draw

        ims.append(im)
    save_movie(filename, ims, fps)


def save_movie(filename, ims, fps):
    codec = {'avi': 'DIVX', 'mp4': 'mp4v'}
    ext = filename[-3:]
    frames = range(np.shape(ims)[0])
    if ext == 'jpg':
        for frame, im in zip(frames, ims):
            im.save(filename.replace('.jpg', f'_{frame}.jpg'))
    elif ext == 'gif':
        ims[0].save(filename, save_all=True, append_images=ims, duration=1000 / fps, loop=0)
    elif ext in codec.keys():
        shape = np.array(ims[0]).shape[:2][::-1]
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec[ext]), fps, shape)
        for im in tqdm(ims, f'Saving {filename}'):
            out.write(cv2.cvtColor(np.array(im).astype(np.float32), cv2.COLOR_RGB2BGR))
        out.release()
    else:
        print('Filetype {ext} not supported.')


def save_traces(filename, traces):
    trace_nrs = list(set([t.split(': ')[0] for t in traces.columns if ': ' in t]))
    colors = list(set([t.split(': I')[1][:3] for t in traces.columns if ': I' in t]))

    ext = filename[-3:]
    if ext in ['mp4', 'jpg']:
        plot_colors = {'637': 'r', '561': 'g', '488': 'b'}
        buffer_format = {'jpg':'jpg', 'mp4': 'png'}
        ims = []
        for trace_nr in trace_nrs:
            fig = plt.figure()
            for color in colors:
                plt.plot(traces.index, traces[f'{trace_nr}: I{color}nm (a.u.)'], color=plot_colors[color], label=f'{color} nm')
            plt.legend(loc="upper right")
            plt.xlabel('Time (s)')
            plt.ylabel('Intensity (a.u.)')
            plt.title(f'Peak {trace_nr}')
            plt.ylim(-200, 2500)
            plt.show()

            # buf = io.BytesIO()
            # fig.savefig(buf, format = buffer_format[ext])
            # buf.seek(0)
            # img = Image.open(buf)

            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = np.reshape(img, fig.canvas.get_width_height()[::-1] + (3,))
            ims.append(img)

        if ext == 'mp4':
            save_movie(filename.replace('.ims', f'_traces.{ext}'), ims, 1)
        else:
            for frame, im in enumerate(ims):
                im.save(filename.replace('.jpg', f'_{frame}.jpg'))
    elif ext == 'csv':
        traces.to_csv(filename.replace('ims', 'csv'))


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


def correct_drift(image_stack, sub_pixel=False, persistence=0.9, colors=None, show = False):
    if colors is None:
        colors = np.arange(np.shape(image_stack)[1])

    shifts = pd.DataFrame([[0, 0]], columns=['x (pix)', 'y (pix)'])
    ref_image = np.sum(image_stack[0, colors, 0, :, :], axis=0)
    for frame in tqdm(range(image_stack.shape[0] - 1), 'Drift correction'):
        ref_image = persistence * ref_image + (1 - persistence) * np.sum(image_stack[frame, colors, 0, :, :], axis=0)
        image = np.sum(image_stack[frame + 1, colors, 0, :, :], axis=0)
        shift = get_drift(image, ref_image, sub_pixel=sub_pixel)
        image_stack[frame + 1,] = ndimage.shift(image_stack[frame + 1,], [0, 0, shift[0], shift[1]])
        shifts.loc[frame + 1] = shift

    if show:
        plt.plot(shifts, )
        plt.xlabel('frame')
        plt.ylabel('drift(pix)')
        plt.legend(['x', 'y'])
        plt.show()

    return image_stack, shifts


def get_drift(image, ref_image, sub_pixel=True):
    shift_image = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(image).conjugate() * np.fft.fft2(ref_image))))
    max = np.asarray(np.unravel_index(np.argmax(shift_image, axis=None), shift_image.shape))
    if sub_pixel:
        _, offset = fit_peak(get_roi(shift_image, max, 10))
        max = max + np.asarray([offset[1].nominal_value, offset[0].nominal_value])
    shift = max - np.asarray(np.shape(shift_image)) / 2.0
    return shift


def generate_test_image(peaks, size=512, width=5):
    image = np.zeros((size, size))
    x = np.linspace(0, size, size)
    x, y = np.meshgrid(x, x)

    for peak in peaks:
        r = ((x - peak[0]) ** 2 + (y - peak[1]) ** 2) ** 0.5
        image += np.exp(-(r / (2 * width)) ** 2)
    return image


def test_drift():
    im_size = 256
    n_colors = 3
    n_peaks = 30
    n_frames = 100
    peaks = np.random.uniform(im_size, size=(n_peaks, 2))
    peaks = peaks[peaks[:, 1].argsort()]
    peaks0 = peaks.copy()
    drift = np.linspace(0, 15, n_frames)
    image_stack = []
    noise = np.random.poisson(150, size=(n_colors, im_size, im_size))
    for d in drift:
        im = 300 * generate_test_image(peaks, im_size, 3)
        image_stack.append(np.asarray([ndimage.shift(im, [d, 0])] * n_colors) + noise)
        peaks[np.random.randint(n_peaks)] = np.random.uniform(im_size, size=2)
    image_stack = np.reshape(image_stack, [-1, n_colors, 1, im_size, im_size])
    image_stack = filter_image_stack(image_stack, highpass=None)  # , lowpass=im_size/4)

    image_stack, shifts = correct_drift(image_stack, sub_pixel=True, persistence=0.95)

    plt.plot(-drift, color='k')
    plt.plot(0 * drift, color='k')
    plt.plot(shifts, marker='o', linestyle='none', markerfacecolor='None')
    plt.show()

    save_image_stack(r'c:\tmp\test.mp4', image_stack, ['637', '561', '488'], peaks=peaks0,
                     color_ranges=[[-10, 300]] * 3)


if __name__ == '__main__':
    # size = 512
    # n = 100
    # peaks = np.random.uniform(0, size, [n, 2])
    # ref_image = generate_test_image(peaks, size, 2.5)
    # shift = [20.4, 53.2]
    # image = ndimage.shift(ref_image, shift)
    # shift = get_drift(image, ref_image)
    #
    # corrected_image = ndimage.shift(image, shift)
    #
    # for im in [image, corrected_image, ref_image]:
    #     plt.imshow(im, origin='lower')
    #     plt.show()
    # breakpoint()

    # filename = r'C:\Users\noort\Downloads\Plexi_Channel2_FOV4_dsDNAPol1_2022-03-21_Protocol 4_18.56.31.ims'
    # filename = r'C:\Users\noort\Downloads\Slide2_Channel1_FOV2_512_Int100_Exp50_Rep05_Pol1_2022-04-06_Protocol 2_16.32.11.ims'
    filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV13_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.29.35.ims'
    # filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV14_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.38.58.ims'
    filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.ims'
    # filename = r'C:\Users\noort\Downloads\Slide1_Chan2_FOV1_512_Exp50o50r_pr%40o70r_Rep100_Int120_T+P+P_2022-04-22_Protocol 5_15.08.02.ims'

    if True:
        roi_width = 512
        header = read_header(filename)
        # print(header['colors'])
        # header['colors'] = ['561', '488', '637'] # overrule header
        # print(header['colors'])
        image_stack = ims(filename)
        # print(image_stack.resolution)
        image_stack = stack_cut_roi(image_stack, roi_width=roi_width)
        # image_stack = image_stack[::5, ]
        image_stack = filter_image_stack(image_stack)
        image_stack, drift = correct_drift(image_stack, sub_pixel=True, persistence=0.95, show=True)

        radius = 125 / header['nm_pix']
        selection_image = np.zeros_like(image_stack[0, 0, 0, :, :]).astype(float)
        for color in ['637']:
            selection_image += np.percentile(image_stack[:10, header['colors'].index(color), 0, :, :], 80, axis=0)
        selection_image = filter_image(selection_image)

        peaks = find_peaks(selection_image, radius * 4, treshold_sd=3.5)
        save_image_stack(filename.replace('.ims', '.mp4'), image_stack, header['colors'], peaks=peaks, radius=radius)

        traces = get_traces(image_stack, peaks, radius, header)
        for d in drift.columns[::-1]:
            traces.insert(0, f'Drift_{d.replace("pix", "nm")}', drift[d].values *header['nm_pix'])

        save_traces(filename.replace('.ims', '.csv'), traces)
        save_traces(filename.replace('.ims', '_traces.mp4'), traces)
    else:
        test_drift()
