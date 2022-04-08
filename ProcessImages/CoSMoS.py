import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageDraw
from imaris_ims_file_reader.ims import ims
from scipy import fftpack
from tqdm import tqdm
import h5py

from ProcessImages.ImageIO import find_peaks, create_circular_mask
from ProcessImages.ImageIO2 import merge_rgb_images, scale_u8


def filter_image(image, highpass=None, lowpass=None):
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
    return image


def get_background(image):
    background = image[image < 0]
    background = np.random.choice([-1, 1], len(background))
    std = np.std(background)
    return std


def plt_circles(coords, radius, color=None, fill=False, show_number=False):
    ax = plt.gca()
    for i, c in enumerate(coords):
        ax.add_patch(plt.Circle(c, radius, color=color, fill=fill))
        if show_number:
            plt.text(*(c + radius), str((i)), color=color)
    plt.axis('equal')


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


def filter(image_stack, highpass=5, low_pas=None):
    shape = np.shape(image_stack)
    image_stack = np.reshape(image_stack, (-1, shape[-2], shape[-1]))
    image_stack = [filter_image(image, highpass) for image in image_stack]
    image_stack = [image - np.percentile(image, 50) for image in image_stack]
    image_stack = np.reshape(np.asarray(image_stack), shape)
    return image_stack


def save_rgb(filename, image_stack, channels, t=0, peaks=None, radius=10, numbers = False):
    im_rgb = np.zeros([3, np.shape(image_stack)[-2], np.shape(image_stack)[-1]])
    image_colors = {'637': 0, '561': 1, '488': 2}
    for color, channel in enumerate(channels):
        image = image_stack[t, color, 0, :, :]
        std = get_background(image)
        im_rgb[image_colors[channel]] = scale_u8(image, [-std, 20 * std])
    im = merge_rgb_images(im_rgb[0] * 0, im_rgb[0], im_rgb[1], im_rgb[2])

    for i, coord in enumerate(peaks):
        bbox = (coord[0] - radius, coord[1] - radius, coord[0] + radius, coord[1] + radius)
        draw = ImageDraw.Draw(im)
        draw.ellipse(bbox, fill=None, outline='white')
        bbox = list(np.clip(coord+radius/np.sqrt(2), 0, np.shape(image_stack)[-1]))
        if numbers:
            draw.text(bbox, str(i), color='white')
    del draw
    im.save(filename.replace('.ims', f'_{t}.jpg'))

def read_channels(filename):
    channels = []
    with h5py.File(filename, 'r') as hdf:
        for channel in range(6):
            try:
                channels.append(hdf[rf'DataSetInfo/Channel {channel}'].attrs['LSMExcitationWavelength'][:3].tobytes().decode())
            except KeyError:
                return channels



filename = r'C:\Users\noort\Downloads\Plexi_Channel2_FOV4_dsDNAPol1_2022-03-21_Protocol 4_18.56.31.ims'
filename = r'C:\Users\noort\Downloads\Slide2_Channel1_FOV2_512_Int100_Exp50_Rep05_Pol1_2022-04-06_Protocol 2_16.32.11.ims'
print(f'Opening file: {filename}')

image_stack = ims(filename)
channels = read_channels(filename)
print(channels)

image_stack = cut_roi(image_stack, roi_width=150)
print('Filtering ....')
image_stack = filter(image_stack, highpass=5)


radius = 5
print('Finding peaks...')
peaks = find_peaks(image_stack[0, channels.index('561'), 0, :, :], radius, n_traces=10000, treshold_sd=2)

for t in range(image_stack.shape[0]):
    save_rgb(filename, image_stack, channels, t, peaks, radius)

traces = get_traces(image_stack, peaks, radius)
plot_colors = {'637': 'r', '561': 'g', '488':'b'}
for trace_nr, trace in enumerate(traces):
    for i, channel in enumerate(channels):
        plt.plot(trace[i], color=plot_colors[channel])
    plt.title(f'peak {trace_nr}')
    plt.ylim(-200, 2500)
    plt.show()


column_names = []
for trace_nr, trace in enumerate(traces):
    for i, channel in enumerate(channels):
       column_names.append(f'{trace_nr}: {channel}nm')
df = pd.DataFrame(np.reshape(traces, (-1, np.shape(traces)[-1])).T, columns=column_names)
df.to_csv(filename.replace('ims', 'csv'))
