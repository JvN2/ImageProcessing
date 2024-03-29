import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from natsort import natsorted
from scipy import ndimage
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

import ProcessImages.ImageIO as iio


def get_rgb_image(im, page, shift=[0, 0]):
    width = np.shape(im)[1] // 2
    im.seek(page * 2)
    r = np.asarray(im)[:, width:]
    im.seek(page * 2 + 1)
    g = np.asarray(im)[:, :width]
    g = ndimage.shift(g, shift)
    b = np.asarray(im)[:, width:]
    return np.asarray([r, g, b]).astype(float)


# settings
filename = r'c:\tmp\Alex_Fret\test1.tif'
filename = r'c:\tmp\Alex_Fret\test2.tif'

shift = [5, -4]
radius = 3.5
highpass = 50
lowpass = 3
treshold_sd = 4.5
intensity_range = np.asarray([-5, 45])

# open file
im = Image.open(filename)
n_frames = im.n_frames // 2

if True:
    # find peaks
    summed_image = get_rgb_image(im, 0)[0] * 0
    for i in range(n_frames):
        summed_image += np.sum(get_rgb_image(im, i, shift), axis=0)
    peak_coords = iio.find_peaks(summed_image, radius * 3, treshold_sd)

if True:
    # get traces
    trace_extraction = iio.TraceExtraction()
    trace_extraction.set_coords(np.asarray(peak_coords[['X (pix)', 'Y (pix)']]), n_frames, radius)
    for i in tqdm(range(n_frames), postfix='Get traces'):
        rgb = get_rgb_image(im, i, shift)
        rgb = iio.filter_image(rgb, highpass=highpass, lowpass=lowpass)
        for color, label in enumerate(['donor', 'acceptor', 'fret']):
            trace_extraction.extract_intensities(rgb[color], i, label=label)
    traces = trace_extraction.df
    traces = traces.reindex(natsorted(traces.columns), axis=1)
    traces.to_csv(filename.replace('.tif', '.csv'))

if True:
    # make color movie
    movie = iio.Movie()
    movie.open(filename.replace('.tif', '.mp4'), 10)
    movie.set_range(red=intensity_range, green=intensity_range, blue=intensity_range / 2)
    movie.set_circles(np.asarray(peak_coords[['X (pix)', 'Y (pix)']]), radius * 2)
    for i in tqdm(range(n_frames), postfix='Save color movie'):
        rgb = get_rgb_image(im, i, shift)
        rgb = iio.filter_image(rgb, highpass=highpass, lowpass=lowpass)
        movie.add_frame(red=rgb[0], green=rgb[1], blue=rgb[2], label=f'frame\n = {i}')

if True:
    # movie of plots
    data = pd.read_csv(filename.replace('.tif', '.csv'), index_col=0)
    trace_nrs = natsorted(np.unique([name.split(':')[0] for name in data.columns]))
    movie = iio.Movie()
    movie.open(filename.replace('.tif', '_traces.mp4'))
    for trace_nr in tqdm(trace_nrs, postfix='Save plots'):
        for name, color in zip(['donor', 'acceptor', 'fret'], ['g', 'r', 'b']):
            trace_name = f'{trace_nr}: I {name} (a.u.)'
            plt.plot(data.index, data[trace_name], color=color, label=name, linewidth=0.5)
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Intensity (a.u.)')
        plt.ylim((-100, 900))
        plt.title(f'trace {trace_nr}')
        movie.add_plot()
