from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import fftpack, ndimage
from scipy.optimize import curve_fit
from uncertainties import ufloat


class Movie():
    def __init__(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def __call__(self, filename, fps=1):
        self.filename = filename
        ext = filename.split('.')[-1].lower()
        codec = {'avi': 'DIVX', 'mp4': 'mp4v'}
        self.codec = codec[ext]
        self.out = None

        self.rgb_image = None
        self.fps = fps
        self.range = [None] * 4
        self.frame_nr = -1

        self.circle_radius = None
        self.circle_coords = []
        return self

    def add_frame(self, red=None, green=None, blue=None, grey=None, label=None):
        self.frame_nr += 1
        if self.rgb_image is not None:
            self.rgb_image *= 0
        for i, image in enumerate([red, green, blue, grey]):
            if image is not None:
                if self.out is None:
                    size = np.asarray(image).shape
                    self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*self.codec), self.fps, size)
                    self.rgb_image = np.zeros((size[0], size[1], 3), 'int32')
                if self.range[i] is None:
                    self.range[i] = [np.percentile(image, 10), np.percentile(image, 90)]

                if i == 3:
                    for j in [0, 1, 2]:
                        self.rgb_image[..., j] += scale_u8(image, self.range[i]).astype(np.int32)
                else:
                    self.rgb_image[..., i] += scale_u8(image, self.range[i]).astype(np.int32)

        # Use Pillow to annotate image
        im = Image.fromarray(np.clip(self.rgb_image, 0, 255).astype(np.uint8), mode='RGB')
        draw = ImageDraw.Draw(im)
        if label is not None:
            font_family = "arial.ttf"
            font = ImageFont.truetype(font_family, 15)
            draw.text([10, 10], label, color='white', font=font)

        for i, coords in enumerate(self.circle_coords):
            circle = np.asarray([coords - self.circle_radius, coords + self.circle_radius]).reshape((4)).tolist()
            draw.ellipse(circle, outline='white')
            if self.frame_nr == 0:
                text_position = 0.3 * self.circle_radius * np.ones(2)
                draw.text(text_position + coords, f'{i}', fill='white', font=ImageFont.truetype(font_family, 10))
        del draw

        # im.save(self.filename.replace('.mp4', f'_{self.frame_nr}.jpg'))
        self.out.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        return

    def set_range(self, red=None, green=None, blue=None, grey=None):
        for i, intensity_range in enumerate([red, green, blue, grey]):
            if intensity_range is not None:
                self.range[i] = intensity_range
        return

    def set_circles(self, coords, radius=10):
        self.circle_coords = coords
        self.circle_radius = radius


class DriftCorrection():
    def __init__(self):
        self.ref_image = None
        self.persistence = 0
        self.shift = []
        return

    def calc_drift(self, image, sub_pixel=True, persistence=None):
        if persistence is not None:
            self.persistence = persistence
        if self.ref_image is None:
            self.ref_image = image
            self.shift.append(np.zeros(2))
        else:
            self.shift.append(get_drift(image, self.ref_image, sub_pixel=sub_pixel))
            shifted_image = ndimage.shift(image, self.shift[-1])
            self.ref_image = self.persistence * self.ref_image + (1 - self.persistence) * shifted_image
        return self.shift[-1]

    def get_df(self, nm_px=None):
        if nm_px is None:
            return pd.DataFrame(np.asarray(self.shift), columns=['Drift x (pix)', 'Drift y (pix)'])
        else:
            return pd.DataFrame(np.asarray(self.shift) * nm_px, columns=['Drift x (nm)', 'Drift y (nm)'])


class TraceExtraction():
    def __init__(self):
        self.coords = []
        self.radius = 10
        self.masks = None
        self.df = pd.DataFrame()
        self.frame_nr = -1
        return

    def set_coords(self, coords, shape, radius=None):
        if radius is not None:
            self.radius = radius
        self.masks = [create_circular_mask(radius, shape, c) for c in coords]

    def get_intensities(self, image, coords=None, radius=None, label=''):
        self.frame_nr += 1
        if coords is not None:
            self.set_coords(coords, np.shape(image), radius)

        intensities = {f'{i}: I {label} (a.u.)' : np.sum(m*image) for i, m in enumerate(self.masks)}
        self.df = self.df.append(pd.Series(intensities, name= self.frame_nr))

        return self.df


def scale_u8(image, intensity_range=None):
    if intensity_range is None:
        intensity_range = np.asarray([np.percentile(image, 5), 1.5 * np.percentile(image, 95)])
    return np.uint8(np.clip((255 * (image - intensity_range[0]) / (intensity_range[1] - intensity_range[0])), 1, 255))


def read_tiff(filename):
    return np.asarray(plt.imread(Path(filename)).astype(float))


def filter_image(image, highpass=None, lowpass=None):
    size = len(image[0])
    x = np.outer(np.linspace(-size / 2, size / 2, size), np.ones(size)) - 0.5
    y = x.T
    r = (x ** 2 + y ** 2) ** 0.5
    filter = np.ones_like(image).astype(float)

    if (highpass is not None) and (highpass > 0):
        filter *= 1 / (1 + 2 ** (6 * (size / highpass - r)))  # Butterworth filter
    if (lowpass is not None) and (lowpass > 0):
        filter *= np.exp(-(r / (2 * size / lowpass)) ** 2)  # Gaussian filter

    im_fft = fftpack.fft2(image)
    im_fft = fftpack.fftshift(im_fft)
    im_fft *= filter

    im_fft = fftpack.fftshift(im_fft)
    image = np.real(fftpack.ifft2(im_fft)).astype(float)

    if highpass == 0:
        image -= np.percentile(image, 25)

    return image


def get_drift(image, ref_image, sub_pixel=True):
    shift_image = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(image).conjugate() * np.fft.fft2(ref_image))))
    max = np.asarray(np.unravel_index(np.argmax(shift_image, axis=None), shift_image.shape))
    if sub_pixel:
        _, offset = fit_peak(get_roi(shift_image, max, 10))
        max = max + np.asarray([offset[1].nominal_value, offset[0].nominal_value])
    shift = max - np.asarray(np.shape(shift_image)) / 2.0
    return shift


def get_roi(image_array, pos, width):
    bottom_left = np.asarray(pos) - width // 2
    roi = image_array[bottom_left[0]:bottom_left[0] + width, bottom_left[1]:bottom_left[1] + width]
    return roi


def fit_peak(Z, show=False, center=[0, 0]):
    # Our function to fit is a two-dimensional Gaussian
    def gaussian(x, y, x0, y0, sigma_x, sigma_y, A, offset):
        return A * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2) + offset

    # This is the callable that is passed to curve_fit. M is a (2,N) array
    # where N is the total number of data points in Z, which will be ravelled
    # to one dimension.
    def _gaussian(M, *args):
        x, y = M
        arr = gaussian(x, y, *args)
        return arr

    Z = np.asarray(Z)
    N = len(Z)
    X, Y = np.meshgrid(np.linspace(0, N - 1, N) - N / 2 + center[0],
                       np.linspace(0, N - 1, N) - N / 2 + center[1])
    p = (center[0], center[1], 2, 2, np.max(Z), np.min(Z))

    xdata = np.vstack((X.ravel(), Y.ravel()))
    popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p)
    fit = gaussian(X, Y, *popt)

    pars = []
    for i, p in enumerate(popt):
        pars.append(ufloat(p, np.sqrt(pcov[i, i])))

    if show:
        # Plot the 3D figure of the fitted function and the residuals.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='afmhot')
        ax.contourf(X, Y, Z - fit, zdir='z', offset=-4, cmap='afmhot')
        ax.set_zlim(-4, np.max(fit))
        plt.show()

    return fit, np.asarray(pars)


def create_circular_mask(width, size=None, center=None, steepness=3):
    if size is None:
        size = [width, width]
    if center is None:
        center = -0.5 + np.asarray(size) / 2
    x = np.outer(np.linspace(0, size[0] - 1, size[0]), np.ones(size[1]))
    y = np.outer(np.ones(size[0]), np.linspace(0, size[1] - 1, size[1]))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = 1 - 1 / (1 + np.exp(-(r - width / 2) * steepness))
    return mask


def find_peaks(image, radius=20, treshold_sd=3, n_traces=2000):
    max = np.max(image)
    median = np.median(image)
    treshold = np.median(image) + treshold_sd * np.std(image)
    peaks = []
    while max > treshold and len(peaks) < n_traces:
        max_index = np.asarray(np.unravel_index(np.argmax(image, axis=None), image.shape))
        mask = create_circular_mask(radius, np.shape(image), max_index)
        image = mask * median + (1 - mask) * image
        max = np.max(image)
        peaks.append(np.flip(max_index))
    peaks = np.asarray(peaks)
    if peaks.any():
        peaks = peaks[peaks[:, 1].argsort()]
    return peaks