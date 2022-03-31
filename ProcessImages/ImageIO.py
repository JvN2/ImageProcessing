import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import glob, os, re, sys, configparser
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageSequence, TiffImagePlugin
from scipy.optimize import curve_fit
from uncertainties import ufloat

sys.path.append(r'C:\Users\noort\PycharmProjects\TraceEditor')
import TraceIO as tio

photons_e = 3.0
font = ImageFont.truetype('arial', 25)


def scale_image_u8(image_array, z_range=None):
    if z_range is None:
        z_range = [-2 ** 15, 2 ** 15]
    image_array = np.uint8(np.clip((255 * (image_array - z_range[0]) / (z_range[1] - z_range[0])), 0, 255))
    return image_array


def crop_image(image_array, center=None, w=100):
    if center is None:
        center = np.asarray(np.shape(image_array)) // 2
    return image_array[center[0] - w // 2:center[0] + w // 2, center[1] - w // 2:center[1] + w // 2]


def filter_image(image_array, highpass=None, lowpass=None, SLIM=False):
    size = len(image_array[0])
    x = np.outer(np.linspace(-size / 2, size / 2, size), np.ones(size)) - 0.5
    y = x.T
    r = (x ** 2 + y ** 2) ** 0.5
    filter = np.ones_like(image_array)

    if highpass is not None:
        filter *= 1 / (1 + 2 ** (6 * (highpass - r)))  # Butterworth filter
    if lowpass is not None:
        filter *= np.exp(-(r / (2 * lowpass)) ** 2)  # Gaussian filter

    im_fft = fftpack.fft2(image_array)
    im_fft = fftpack.fftshift(im_fft)
    im_fft *= filter

    if SLIM:
        max_peak = (np.argwhere(np.abs(im_fft.max()) == np.abs(im_fft))) - size / 2
        period = (np.sum(max_peak ** 2)) ** 0.5
        print(f'SLIM period = {period} (pix)')
        filter = np.sqrt((np.cos(np.pi * y / period)) ** 2)
        im_fft *= filter

    im_fft = fftpack.fftshift(im_fft)
    image_array = fftpack.ifft2(im_fft)
    return np.abs(image_array)


def aggregate_images(filenames, type='mean'):
    if type == 'max':
        for filename in filenames:
            try:
                max = np.maximum(plt.imread(filename).T, max)
            except UnboundLocalError:
                max = np.asarray(plt.imread(filename).T).astype(float)
        return max

    for filename in filenames:
        try:
            sum += plt.imread(filename).T
        except UnboundLocalError:
            sum = np.asarray(plt.imread(filename).T).astype(float)

    average_image = np.asarray(sum) / len(filenames)
    if type == 'mean':
        return average_image
    if type == 'sd':
        variance = np.zeros_like(average_image)
        for filename in filenames:
            variance += (np.asarray(plt.imread(filename).T).astype(float) - average_image) ** 2
        return np.sqrt(variance)


def merge_tiffs(filepaths, n_frames=None, SLIM_periods=1):
    fileout = (filepaths.replace('*', ''))
    filenames = list(set(glob.glob(filepaths)))
    filenames.sort(key=tio.natural_keys)

    if n_frames is None:
        frames = len(filenames) // SLIM_periods
        frames = np.max(frames, n_frames)
    else:
        frames = n_frames

    filenames = np.reshape(np.asarray(filenames[:SLIM_periods * frames]), (frames, SLIM_periods))

    im = np.asarray(Image.open(filenames[0, 0]), dtype=float)
    i = 0
    with TiffImagePlugin.AppendingTiffWriter(fileout) as tf:
        for frame in filenames:
            print(i, '/', frames, frame[-1])
            im *= 0
            for period in frame:
                im += np.asarray(Image.open(period), dtype=float)
            selection = crop_image(im, w=200, center=[730, 530])
            selection = filter_image(selection, highpass=10, lowpass=50)
            selection = np.asarray(selection, dtype=np.int16)

            im_out = Image.fromarray(selection)
            im_out.save(tf)
            tf.newFrame()

            i += 1
            if i >= frames:
                break

    print(fileout)
    return fileout


def add_time_bar(img, i, n, progress_text=None, progress_step=None):
    bg_color = (255, 255, 255)
    size_pix = np.asarray(img.size)
    img_draw = ImageDraw.Draw(img)
    if progress_step is not None:
        box = (5, size_pix[0] - 5, size_pix[0] - 5, size_pix[1] - 10)
        bar = (5, size_pix[0] - 5, 5 + (size_pix[0] - 10) * i / (n - 1), size_pix[1] - 10)
        img_draw.rectangle(box, outline=bg_color)
        img_draw.rectangle(bar, outline=bg_color, fill=bg_color)
    if progress_text is not None:
        img_draw.text((5, size_pix[1] - 40), f'{progress_text} = {progress_step * i:4.1f}', fill=bg_color, font=font)
    return


def add_scale_bar(img, pix_um, scale=1, barsize_um=5):
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


def tiff_to_gif(filename, scale=1, Irange=None, n_pages=None, frame_time_s=None, pix_um=None):
    img = Image.open(filename)
    img.load()
    width, height = img.size
    if n_pages is None or n_pages < 0:
        n_pages = img.n_frames
    new_size = (width * scale, height * scale)
    summed = np.zeros([width * scale, height * scale])
    img_out = []
    i = 0
    for page in ImageSequence.Iterator(img):
        page = page.resize(new_size)
        arr = photons_e * np.asarray(page)
        arr = scale_image_u8(arr, Irange)
        img_out.append(Image.fromarray(arr))
        summed += page
        print(i, '/', n_pages)

        add_scale_bar(img_out[i], pix_um, scale)
        add_time_bar(img_out[i], i, frame_time_s, n_pages)

        i += 1
        if i >= n_pages:
            break

    img_out[0].save(filename.replace('tiff', 'gif'), save_all=True, append_images=img_out[1:], duration=100, loop=0)
    img_sum = Image.fromarray(scale_image_u8(photons_e * summed / n_pages, Irange))
    img_sum.save(filename.replace('.tiff', '_sum.gif'), )

    return filename


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


def tiffs_to_traces(filename, n_pages=None, n_traces=None, width=10, scale=2, Irange=None, treshold_sd=3):
    img = Image.open(filename)
    img.load()
    im_width, im_height = img.size
    if n_pages is None or n_pages < 0:
        n_pages = img.n_frames

    summed_img = np.zeros([im_width, im_height])
    i = 0
    for img in ImageSequence.Iterator(img):
        summed_img += img
        i += 1
        if i >= n_pages:
            break
    averaged_img = summed_img / n_pages
    median = np.median(averaged_img)
    averaged_img -= median

    img_out = Image.fromarray(scale_image_u8(averaged_img, Irange))
    new_size = (im_width * scale, im_height * scale)
    img_out = img_out.resize(new_size)

    img_draw = ImageDraw.Draw(img_out)
    font = ImageFont.truetype('arial', 5 * scale)
    text_position = 0.35 * scale * width * np.ones(2)
    circle_position = scale * np.asarray([-width, -width, width, width]) / 2

    treshold = np.median(averaged_img) + treshold_sd * np.std(averaged_img)
    if n_traces is None:
        n_traces = np.inf
    trace_i = 0
    traces = []
    max_intensity = np.max(averaged_img)
    while max_intensity > treshold and trace_i < n_traces:
        max_index = np.asarray(np.unravel_index(np.argmax(averaged_img, axis=None), averaged_img.shape))
        mask = create_circular_mask([im_width, im_height], max_index, width)
        trace = []
        i = 0
        for img in ImageSequence.Iterator(img):
            trace.append(np.sum(mask * img / np.sum(mask)))
            i += 1
            if i >= n_pages:
                break
        averaged_img = mask * median + (1 - mask) * averaged_img

        trace = np.asarray(trace - median) * photons_e
        max_intensity = np.max(averaged_img)
        traces.append(trace)

        trace_i += 1
        print(f'{trace_i}) pos: {np.flip(max_index)}, '
              f'Intensity: {np.mean(trace):.1f} +/- {np.std(trace):.1f} ({np.std(trace) / np.sqrt(np.mean(trace)):.1f})')
        max_index = np.flip(max_index) * scale
        img_draw.text(text_position + max_index, f'{trace_i}', fill=(255, 0, 0, 255), font=font)
        img_draw.ellipse(list(np.append(max_index, max_index) + circle_position), outline=255)

    # add_scale_bar(img_out, 0.105, scale)
    img_out.save(filename.replace('.tiff', '_peak.gif'), )

    plt.plot(np.asarray(traces).T)
    tio.format_plot('frame', 'I (photons)', yrange=Irange)
    return filename


def find_peaks(image_array, width=20, scale=1, treshold_sd=3, n_traces=20, range=None, file_out="test.png", show = False):
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
        mask = create_circular_mask(width, np.shape(image_array), max_index )
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


def get_roi(image_array, center, width):
    start = (np.asarray(center) - width // 2).astype(int)
    roi = image_array[start[0]: start[0] + width, start[1]: start[1] + width]  # invert axis for numpy array of image
    return roi


def set_roi(image_array, center, roi):
    width = len(roi[0])
    start = (np.asarray(center) - width // 2).astype(int)
    image_array[start[0]: start[0] + width, start[1]: start[1] + width] = roi  # invert axis for numpy array of image
    return


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


def find_peaks2(image_array, width=25, scale=1, treshold_sd=3, n_peaks=20, range=None, file_out="test.png"):
    if range is None:
        range = [np.median(image_array) - np.std(image_array), np.median(image_array) + 10 * np.std(image_array)]

    mask = create_circular_mask(width)

    treshold = np.median(image_array) + treshold_sd * np.std(image_array)
    trace_i = 0
    pos = []
    max = np.max(image_array)
    while max > treshold and trace_i < n_peaks:
        max_index = np.asarray(np.unravel_index(np.argmax(image_array, axis=None), image_array.shape))
        roi = get_roi(image_array, max_index, width)

        # plt.imshow(roi, cmap='afmhot', origin='lower')
        # plt.colorbar()
        # plt.show()

        fit, fit_pars = fit_peak(roi, center=max_index)
        print(f'{trace_i}:', [f'{p:P}' for p in fit_pars])

        background = np.sum(roi * ((1 - mask) / np.sum(1 - mask)))
        # set_roi(image_array, max_index,  mask * background + roi*(1-mask))
        set_roi(image_array, max_index, roi - fit + fit_pars[-1].nominal_value)

        # trace_i = n_traces + 1  # Loop only once; for debugging
        trace_i += 1

    plt.imshow(image_array.T, cmap='afmhot', origin='lower')
    plt.colorbar()
    plt.show()
    #     mask = peak_selection_mask(np.shape(image), max_index, width)
    #     image = mask * median + (1 - mask) * image
    #     max = np.max(image)
    #     max_index = np.flip(max_index) * scale
    #     image_draw.text(text_position + max_index, f'{trace_i}', fill=(255, 0, 0, 255), font=font)
    #     image_draw.ellipse(list(np.append(max_index, max_index) + circle_position), outline=(255, 0, 0, 255))
    #     trace_i += 1
    #     # print(trace_i)
    #     pos.append(max_index)
    #     Image.fromarray(scale_image(image, range)).show()
    # image_out.save(file_out)
    return np.asarray(roi)


def get_roi(image_array, pos, width):
    # pos = np.flip(pos)
    bottom_left = pos - width // 2
    roi = image_array[bottom_left[0]:bottom_left[0] + width, bottom_left[1]:bottom_left[1] + width]
    return roi


def image_to_traces(filenames, positions, width=20):
    h5_file = twophoton_dat_to_h5(aggregate_filenames(filenames, ext='dat'), over_write=True)
    tio.convert_log_h5(h5_file)
    channels, labels, _ = tio.h5_contents(h5_file)
    trace = tio.h5_read_trace(h5_file, channels[0])
    traces = np.empty((len(positions), len(trace)))
    traces[:] = np.nan

    mask = create_circular_mask([width, width], [width / 2, width / 2], width=width / 2)
    mask = 1 - mask
    mask /= np.sum(mask)

    for filename in filenames:
        index = int(filename.split('image')[-1].split('.')[0]) - 1

        image = np.asarray(plt.imread(filename)).astype(float)
        background = np.median(image)
        background = 0

        for label, position in enumerate(positions):
            roi = get_roi(image, position, width)

            background = np.sum(mask * roi)

            intensity = (np.sum(roi) - np.product(np.shape(roi)) * background)
            traces[label, index] = intensity
            if label == 0:
                print(filename, index, '/', len(filenames), intensity, background)

        for label, intensity in enumerate(np.asarray(traces)):
            tio.h5_write_trace(intensity, h5_file, 'Intensity (a.u.)', label)

        for label, position in enumerate(positions):
            tio.h5_write_par(h5_file, str(label), ' selected', 1, type='local')
            tio.h5_write_par(h5_file, str(label), 'X0 (pix)', position[0], type='local')
            tio.h5_write_par(h5_file, str(label), 'Y0 (pix)', position[1], type='local')
            tio.h5_write_par(h5_file, str(label), 'Imax (a.u.)', np.max(intensity), type='local')

    return


def aggregate_filenames(filenames, ext='h5'):
    foldername = os.path.dirname(filenames[0])
    filename = fr'{foldername}\{os.path.basename(foldername)}.{ext}'
    return filename


def twophoton_dat_to_h5(filename, check_existing=True, over_write=False):
    file_out = tio.change_extension(filename, 'h5')
    # pd.set_option('io.hdf.default_format', 'table')

    if check_existing or over_write:
        if os.path.isfile(file_out):
            if over_write:
                os.remove(file_out)
            else:
                print(f'{file_out} already exists, no conversion necessary')
                return file_out

    traces = pd.read_csv(filename, sep='\t', header=(0), decimal=',')

    for trace in traces.items():
        label = re.findall(r'\d+', trace[0])
        if label == []:
            label = 'shared'
        else:
            label = label[0]
        channel = trace[0].replace(label, '')
        tio.h5_write_trace(trace[1].values, file_out, channel, label)
    return file_out


def process_2p(filenames):
    im = aggregate_images(filenames, type='mean')
    pos = find_peaks(im, width=10, n_traces=10, treshold_sd=1, file_out=aggregate_filenames(filenames, ext='png'))
    print('Done finding peaks')
    traces = image_to_traces(filenames, pos)
    print('Done')
    return im


def save_gif(filename, frames_array, fps=5, z_range=[120, 200], progress_text='', progress_step=None, um_pix=None):
    filename = tio.change_extension(filename, 'gif')
    cm = plt.get_cmap('hot')
    ims = []
    for i, frame in enumerate(frames_array):
        frame = scale_image_u8(frame, z_range)
        im = Image.fromarray(frame)

        if progress_step is not None:
            add_time_bar(im, i, len(frames_array), progress_text, progress_step)
            # drawObject.text((5, size_pix[1] - 40), f'{progress_text} = {progress_step * i:4.1f}', fill=bg_color, font=font)
        if um_pix is not None:
            add_scale_bar(im, um_pix, barsize_um=10)

        ims.append(im)
    ims[0].save(filename, save_all=True, append_images=ims, duration=1000 / fps, loop=0)


def read_parameter(filename, section, item):
    # ConfigParser ingnores items containing '#'. Replace in log file and item name if neccesary.
    if item[:1] == 'N ':
        item = item.replace('N ', '#')
    if '#' in item:
        print(f'Warning: "#" in file {filename} not compatible with Python ConfigParser. Replaced # with "N ".')
        with open(filename, 'r') as file:
            filedata = file.read()
        filedata = filedata.replace('#', 'N ')
        with open(filename, 'w') as file:
            file.write(filedata)
            print('replaced')
    item = item.replace('#', 'N ')

    config = configparser.ConfigParser()
    config.read(filename)
    value = config.get(section, item)

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def process_folder(foldername):
    filenames = sorted(glob.glob(rf"{foldername}\image*.tiff"), key=tio.natural_keys)
    logfile = rf'{foldername}\{os.path.split(foldername)[-1]}.log'
    file_nr = int(logfile.split('_')[-1].split('.')[0])

    # Create movie of MaxIntensity stacks
    stack_size = int(read_parameter(logfile, 'Step settings', '#z steps'))
    um_pix = 0.26

    LED_on = bool(read_parameter(logfile, 'Excitation', 'LED (V)'))
    dt = read_parameter(logfile, 'Timing', 'Repetition time (s)') * stack_size
    z_range = [100, 300]

    filenames = np.array(filenames).reshape(len(filenames) // stack_size, stack_size)
    filename_out = rf'{foldername}\{os.path.split(foldername)[-1]}_timeseries.gif'
    frames = []
    for stack in filenames:
        print(f'{stack[0]} > {filename_out}')
        if LED_on: stack = stack[1:]
        frames.append(aggregate_images(stack, 'max'))
    save_gif(filename_out, frames, progress_text='t (s)', progress_step=dt, z_range=z_range, um_pix=um_pix)

    # Create movie of z-stack
    filename_out = rf'{foldername}\{os.path.split(foldername)[-1]}_zstack.gif'
    z_step = int(read_parameter(logfile, 'Step settings', 'z step (um)'))
    frames = []
    for filename in filenames[0]:
        print(f'{stack[0]} > {filename_out}')
        frames.append(np.asarray(plt.imread(filename)).T.astype(float))
    save_gif(filename_out, frames, progress_text='z (um)', progress_step=z_step, z_range=z_range, um_pix=um_pix)


def process_multiple_folders(root_folder):
    folders = os.listdir(root_folder)
    for f in folders:
        try:
            process_folder(root_folder + f)
        except:
            pass


def process_file(filename):
    image_array = np.asarray(plt.imread(filename)).astype(float)
    # image_array = filter_image(image_array, highpass = 0, lowpass= 50)
    # f_i = fftpack.fft2(image_array- np.mean(image_array))
    # f_i = np.roll(f_i, np.asarray(np.shape(f_i))//2, (0,1))
    # plt.imshow(np.abs(f_i), origin='lower', cmap='afmhot', vmin = 0, vmax = 150000)

    # image_array = filter_image(image_array, highpass = 0.5, lowpass=150)
    # plt.imshow(image_array, origin='lower', cmap='afmhot',)
    # plt.colorbar()
    # plt.show()

    res = find_peaks2(image_array, n_peaks=40, width=35)

    # z_range = [100, 500]
    # frame = scale_image_u8(image_array, z_range)
    # im = Image.fromarray(frame)
    # add_scale_bar(im, pix_um=0.26)
    # im.save(filename.replace('tiff', 'jpg'))


if __name__ == "__main__":
    # files = r'N:\Redmar\SLIM_data\data_032\image*.tiff'
    foldername = r'C:\MeasurementData\201218\data_008'

    foldername = r'D:\MeasurementData\201222 - Pollen with Charlotte\data_035'
    filename = r'D:\MeasurementData\201218 - 2p excitation spectra with Deep\data_015/image25.tiff'

    process_file(filename)
    # process_folder(foldername)

    if False:
        im = process_2p(filenames)
        print(filenames[-1])
        h5_file = aggregate_filenames(filenames, 'h5')
        # channels, labels, parameters = tio.h5_contents(h5_file)
        # print(h5_file)
        # print(channels)
        # print(labels)
        # print(parameters)

    # roi = get_roi(im, pos[-1], 20)
    # plt.imshow(roi, cmap='gray', vmin=90, vmax=200)
    # plt.colorbar()
    # plt.show()

    # for f in filenames[1:10:2]:
    #     print(f)

    if False:
        if 'im' not in locals():
            im = plt.imread(filenames[nr])
        plt.imshow(im, cmap='gray', vmin=90, vmax=500)
        plt.colorbar()
        plt.title(filenames[nr])
        plt.show()

        mask = peak_selection_mask([100, 100])
        plt.imshow(mask, cmap='gray')
        plt.colorbar()
        plt.title(filenames[nr])
        plt.show()

    if False:
        im = aggregate_images(filenames)
        # plt.imshow(im.T, cmap='gray', origin='lower')
        # plt.title(filenames[nr])
        find_peaks2(im)
        # N = 100
        # im = np.asarray([list(np.arange(0,N))]*N)
        # center = np.asarray((10, 75))
        # roi = get_roi(im, center,10)*circular_mask(10)
        # set_roi(im, center, roi)
        # print(roi)
        # plt.imshow(im.T, cmap='gray', origin = 'lower')
        # plt.colorbar()
        # plt.show()

    # for filename in filenames[:2]:
    #     print(filename)
    #     im = plt.imread(filename)
    #     plt.imshow(im)
    #     plt.show()

    # Irange = [0, 200]
    # tiff_file = merge_tiffs(files, SLIM_periods=6)
    # tiff_to_gif(files.replace('*', ''), scale=2, Irange=Irange, n_pages=-1, frame_time_s=0.3, pix_um=0.105)
    # tiff_to_traces(files.replace('*', ''), width=10, scale=4, Irange=np.asarray(Irange) * 0.4, treshold_sd=5)
