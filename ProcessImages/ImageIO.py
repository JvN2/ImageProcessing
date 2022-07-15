from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import fftpack, ndimage
from scipy.optimize import curve_fit
from tqdm import tqdm
from uncertainties import ufloat


class Movie():
    # def __init__(self):
    #     return
    #
    # def __enter__(self):
    #     return self
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     return self

    def __init__(self, filename, fps=1):
        self.filename = filename
        ext = filename.split('.')[-1].lower()
        codec = {'avi': 'DIVX', 'mp4': 'mp4v', 'jpg': 'JPEG', 'png': 'PNG'}
        self.codec = codec[ext]
        self.out = None

        self.rgb_image = None
        self.fps = fps
        self.range = [None] * 4
        self.frame_nr = -1

        self.circle_radius = None
        self.circle_coords = []
        return

    def add_plot(self):
        fig = plt.gcf()
        fig.canvas.draw()
        fig.canvas.get_renderer()
        im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = np.reshape(im, fig.canvas.get_width_height()[::-1] + (3,))
        if self.out is None:
            size = np.asarray(im).shape[:2][::-1]
            self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*self.codec), self.fps, size)
        self.out.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        plt.close(fig)
        return

    def add_frame(self, red=None, green=None, blue=None, grey=None, label=None):
        self.frame_nr += 1
        if self.rgb_image is not None:
            self.rgb_image *= 0
        for i, image in enumerate([red, green, blue, grey]):
            if image is not None:
                if self.out is None:
                    self.size = np.asarray(image).shape
                    self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*self.codec), self.fps,
                                               self.size[::-1])
                    self.rgb_image = np.zeros((self.size[0], self.size[1], 3), 'int32')
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

        if self.codec in ['JPEG', 'PNG']:
            path = Path(self.filename[:-4])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            im.save(str(path) + (fr'/frame_{self.frame_nr}.{self.filename.split(".")[-1].lower()}'), self.codec)
            cv2.imshow('test', cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        else:
            frame = cv2.resize(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), self.size[::-1])
            self.out.write(frame)
            # self.out.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
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

    def calc_drift(self, image, sub_pixel=True, persistence=None, ref_image = None):
        if ref_image is not None:
            self.ref_image = ref_image
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
        self.mask = None
        self.traces = []
        return

    def set_coords(self, coords, n_frames, radius=None):
        if radius is not None:
            self.radius = radius
        self.coords = coords
        self.mask = create_circular_mask(int(self.radius * 2))
        self.df = pd.DataFrame(index=range(n_frames))

    def extract_intensities(self, image, frame_nr, coords=None, radius=None, label=''):
        if coords is not None:
            self.set_coords(coords, radius)
        size = np.shape(self.mask)[0]

        col_names = [f'{i}: I {label} (a.u.)' for i, _ in enumerate(self.coords)]
        pd.concat([self.df, pd.DataFrame(columns=col_names)])

        for i, c in enumerate(self.coords):
            self.df.at[frame_nr, f'{i}: I {label} (a.u.)'] = np.sum(get_roi(image, c[::-1], size) * self.mask)

        return self.df


def set_roi(image_array, center, roi):
    width = len(roi[0])
    start = (np.asarray(center) - width // 2).astype(int)
    image_array[start[0]: start[0] + width,
    start[1]: start[1] + width] = roi  # invert axis for numpy array of image
    return


def scale_u8(image, intensity_range=None):
    if intensity_range is None:
        intensity_range = np.asarray([np.percentile(image, 5), 1.5 * np.percentile(image, 95)])
    return np.uint8(np.clip((255 * (image - intensity_range[0]) / (intensity_range[1] - intensity_range[0])), 1, 255))


def read_tiff(filename):
    return np.asarray(plt.imread(Path(filename)).astype(float))


def filter_image(image, highpass=None, lowpass=None, remove_outliers=False):
    size = np.shape(image)
    if len(size) > 2:
        size = size[-2:]

    x = np.outer(np.linspace(-size[0] / 2, size[0] / 2, size[0]), np.ones(size[1])) - 0.5
    y = np.outer(np.ones(size[0]), np.linspace(-size[1] / 2, size[1] / 2, size[1])) - 0.5

    r = (x ** 2 + y ** 2) ** 0.5
    filter = np.ones(size).astype(float)

    if remove_outliers:
        median = np.median(image)
        image[image < median * 0.9] = median

    if (highpass is not None) and (highpass > 0):
        filter *= 1 / (1 + 2 ** (size[0] / highpass - r))  # Butterworth filter
        # filter *= (1- np.exp(-(r / (2 * size / highpass)) ** 2)) # Gaussian filter
        # filter *= r < highpass

    if (lowpass is not None) and (lowpass > 0):
        filter *= 1 - (1 / (1 + 2 ** (size[0] / lowpass - r)))  # Butterworth filter
        # filter *= np.exp(-(r / (2 * size / lowpass)) ** 2)  # Gaussian filter
        # filter *= r >lowpass

    if len(np.shape(image)) == 2:
        im_fft = fftpack.fft2(image)
        im_fft = fftpack.fftshift(im_fft)
        im_fft *= filter

        im_fft = fftpack.fftshift(im_fft)
        image = np.real(fftpack.ifft2(im_fft)).astype(float)

        if highpass == 0:
            image -= np.percentile(image, 25)
    else:
        for i, _ in enumerate(image):
            im_fft = fftpack.fft2(image[i])
            im_fft = fftpack.fftshift(im_fft)
            im_fft *= filter

            im_fft = fftpack.fftshift(im_fft)
            image[i] = np.real(fftpack.ifft2(im_fft)).astype(float)

            if highpass == 0:
                image[i] -= np.percentile(image[i], 25)
    return image


def get_drift(image, ref_image, sub_pixel=True):
    shift_image = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(image).conjugate() * np.fft.fft2(ref_image))))

    max = np.asarray(np.unravel_index(np.argmax(shift_image, axis=None), shift_image.shape))
    if sub_pixel:
        _, offset = fit_peak(get_roi(shift_image, max, 10))
        # max = max + np.asarray([offset[1].nominal_value, offset[0].nominal_value])
        max = max + np.asarray([offset[1], offset[0]])
    return max - np.asarray(np.shape(shift_image)) / 2.0


def get_roi(image, center, width):
    start = (np.asarray(center) - width // 2)
    start = list(np.clip(start, np.zeros(2), np.shape(image) - np.asarray(width)).astype(np.uint16))
    return image[start[0]: start[0] + width, start[1]: start[1] + width]


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
    pars_guess = (center[0], center[1], 2, 2, np.max(Z), np.min(Z))

    bounds = np.asarray(((-np.shape(Z)[0]/2, np.shape(Z)[0]/2), (-np.shape(Z)[1]/2, np.shape(Z)[1]/2), (0.5, 10), (0.5, 10),
                         (np.max(Z) * 0.1, np.max(Z) * 10), (np.min(Z) * 0.1, np.min(Z) * 10))).T

    xdata = np.vstack((X.ravel(), Y.ravel()))

    try:
        popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), pars_guess, bounds=bounds)
        fit = gaussian(X, Y, *popt)

        pars = []
        for i, p in enumerate(popt):
            pars.append(ufloat(p, np.sqrt(pcov[i, i])))
    except:
        pars = []
        for i, p in enumerate(pars_guess):
            pars.append(ufloat(p, np.nan))
        fit = gaussian(X, Y, *pars_guess)

    if show:
        # Plot the 3D figure of the fitted function and the residuals.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='afmhot')
        ax.contourf(X, Y, Z - fit, zdir='z', offset=-4, cmap='afmhot')
        ax.set_zlim(-4, np.max(fit))
        plt.show()

    # return fit, np.asarray(pars)
    return fit, popt


def create_circular_mask(width, size=None, center=None, steepness=3):
    if size is None:
        size = [width, width]
    if center is None:
        center = -0.5 + np.asarray(size) / 2
    x = np.outer(np.linspace(0, size[0] - 1, size[0]), np.ones(size[1]))
    y = np.outer(np.ones(size[0]), np.linspace(0, size[1] - 1, size[1]))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return 1 - 1 / (1 + np.exp(-(r - width / 2) * steepness))


def find_peaks(image, radius=20, treshold_sd=3, n_traces=2000):
    median = np.median(image)
    treshold = np.median(image) + treshold_sd * np.std(image)
    peaks = []
    for _ in tqdm(range(n_traces), postfix='Find peaks'):
        max_index = np.asarray(np.unravel_index(np.argmax(image, axis=None), image.shape))
        mask = create_circular_mask(radius, np.shape(image), max_index)
        image = mask * median + (1 - mask) * image
        max = np.max(image)
        if max < treshold:
            break
        peaks.append(np.flip(max_index))
    peaks = np.asarray(peaks)
    if peaks.any():
        peaks = peaks[peaks[:, 1].argsort()]
    return pd.DataFrame(peaks, columns=['X (pix)', 'Y (pix)'])
