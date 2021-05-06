import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.ndimage
from configparser import ConfigParser


def read_log(filename, show = False):
    if '.log' not in filename:
        filename = filename.split('.')[0] + '.log'
    keys = ['LED (V)', 'Pixel size (um)', 'Z step (um)', 'Magnification', 'Exposure (s)', 'Wavelength (nm)', 'T0 (ms)',
            'Slice (s)', 'DAQ rate (Hz)', 'Transition (s)', 'Sigma', 'A spiral (V)', 'SLIM', 'Spirals', 'Steps',
            'Z step (um)', 'UV (%)', 'Shutter delay (s)', 'xc (V)', 'yc (V)', 'zc (V)', 'x0 (V)', 'y0 (V)', 'z0 (V)']
    settings = {}
    for key in keys:
        settings[key] = 0

    with open(filename) as reader:
        for line in reader:
            for key in keys:
                if key in line:
                    settings[key] = float(line.split('=')[-1])

    settings['zc (V)'] = 0

    if show:
        for key in settings:
            print(f'{key} = {settings[key]}')

    return settings


def add_transition(x_start, x_array, x_end, n, kind='quadratic'):
    if kind == 'zero':
        ynew = x_start * np.ones(int(n))
    else:
        t = np.asarray([-2, -1, n + 1, n + 2])
        y = np.asarray([x_array[-2], x_array[-1], x_end, x_end])
        f = interpolate.interp1d(t, y, kind=kind)
        tnew = np.arange(0, n)
        ynew = f(tnew)

    x_array = np.append(x_array, ynew)

    t = np.asarray([-n - 2, -n - 1, 0, 1])
    y = np.asarray([x_start, x_start, x_array[0], x_array[1]])
    f = interpolate.interp1d(t, y, kind=kind)
    tnew = np.arange(0, n) - n
    ynew = f(tnew)
    x_array = np.append(ynew, x_array)
    return x_array


def check_image(channels, n_pix=300, n_spots=5, d_spots=30, pix_volt=35, frames=None):
    print('Reconstructing excitation image ...')
    w = 10
    x = np.asarray([list(range(n_pix))] * n_pix)
    im = np.exp(-(x - n_pix / 2) ** 2 / w ** 2) * np.exp(-(x.T - n_pix / 2) ** 2 / w ** 2)
    im *= im

    loc = np.zeros((2, n_spots ** 2))
    for i in range(n_spots):
        loc[0][i * n_spots:(i + 1) * n_spots] = np.arange(0, n_spots * d_spots, d_spots) + 0.5 * d_spots * (i % 2)
        loc[1][i * n_spots:(i + 1) * n_spots] = np.ones(n_spots) * d_spots * i * np.sqrt(3) / 2

    for i, _ in enumerate(loc):
        loc[i] = loc[i] - np.mean(loc[i])

    im2 = im * 0
    for i, x in enumerate(loc.T):
        if i == -1:
            im2 += scipy.ndimage.shift(-im, x)
        else:
            im2 += scipy.ndimage.shift(im, x)

    im = im2
    im2 = im * 0

    daq_index = np.asarray(range(len(channels[4])))
    all_frames = np.asarray((
        daq_index[np.diff(channels[4], prepend=0) == 1],
        daq_index[np.diff(channels[4], prepend=0) == -1])).T
    if frames is None:
        frames = range(len(all_frames))

    for i, f in enumerate(all_frames):
        if i in frames:
            print(f'Frame: {i}')
            for x in channels.T[f[0]:f[1]]:
                im2 += scipy.ndimage.shift(im, pix_volt * x[:2])
    plt.imshow(im2.T, origin='lower', cmap='afmhot')
    plt.colorbar()
    plt.show()


def create_signals(filename, show=False):
    pars = read_log(filename, show)
    t = np.arange(0, pars['Exposure (s)'], 1.0 / pars['DAQ rate (Hz)'])
    n_slice = int(pars['Slice (s)'] * pars['DAQ rate (Hz)'])
    n_transition = int(pars['Transition (s)'] * pars['DAQ rate (Hz)'])
    n_active = len(t) + 2 * n_transition
    n_slice = np.max([n_slice, n_active])
    n_pwm = 50

    tau = np.sqrt(t / pars['Exposure (s)']) * np.exp(((t / pars['Exposure (s)']) ** 2 - 1) / (2 * pars['Sigma']))
    if pars['SLIM'] == 0:
        x = 0.455 * pars['A spiral (V)'] * tau * np.sin(np.pi * 2 * pars['Spirals'] * tau) + pars['xc (V)']
        y = 0.455 * pars['A spiral (V)'] * tau * np.cos(np.pi * 2 * pars['Spirals'] * tau) + pars['yc (V)']
        c_block = [add_transition(0, np.ones_like(x), 0, n_transition, kind='zero')]
        x_block = [add_transition(pars['x0 (V)'], x, pars['x0 (V)'], n_transition)]
        y_block = [add_transition(pars['y0 (V)'], y, pars['y0 (V)'], n_transition)]
        z_block = [add_transition(0, np.zeros_like(x), 1, n_transition)]

    # else:
    #     for phase in range(int(pars['SLIM'])):
    #         x = pars['a spiral (V)'] * tau * np.sin(np.pi * 2 * pars['n spirals'] * tau)
    #         # y = 0.15 * pars['a spiral (V)'] * tau * np.cos(np.pi * 2 * pars['n spirals'] * tau)
    #         y = 0.5 * np.ones_like(x) * phase / int(pars['slim'])
    #
    #         a = 0.63 * pars['a spiral (V)']
    #         t0 = np.max(t) / 2
    #         x = np.exp(-(t - t0) ** 2 / (0.5 * t0) ** 2)
    #         x = np.diff(x)
    #         x = np.append(x, 0)
    #         x /= np.max(x)
    #         x *= a
    #         x = np.roll(x, len(t) // 2)
    #         x = np.linspace(a, -a, len(x))
    #
    #         angle = -np.pi / 2
    #
    #         x = np.cos(angle) * x + np.sin(angle) * y
    #         y = np.sin(angle) * x + np.cos(angle) * y
    #
    #         x += pars['xc (V)']
    #         y += pars['yc (V)']
    #
    #         if phase == 0:
    #             c_block = [add_transition(0, np.ones_like(x), 0, n_transition, kind='zero')]
    #             x_block = [add_transition(pars['x0 (V)'], x, pars['x0 (V)'], n_transition)]
    #             y_block = [add_transition(pars['y0 (V)'], y, pars['y0 (V)'], n_transition)]
    #         else:
    #             c_block.append(add_transition(0, np.ones_like(x), 0, n_transition, kind='zero'))
    #             x_block.append(add_transition(pars['x0 (V)'], x, pars['x0 (V)'], n_transition))
    #             y_block.append(add_transition(pars['y0 (V)'], y, pars['y0 (V)'], n_transition))

    ones = np.ones_like(x)
    channels = ['X (V)', 'Y (V)', 'Z (V)', 'Camera', 'Shutter', 'LED', 'UV']

    z_steps = np.linspace(-0.5, 0.5, int(pars['Steps']))* pars['Z step (um)'] * (pars['Steps']-1)
    z_steps /= 10 # um/V
    z_steps += pars['zc (V)']
    z_steps = np.append(z_steps, pars['z0 (V)'])

    for i, z_step in enumerate(z_steps[:-1]):
        for x, y, z, c in zip(x_block, y_block, z_block, c_block):
            new_block = np.zeros([len(channels), n_slice])
            new_block[channels.index('X (V)')][:n_active] = x
            new_block[channels.index('X (V)')][n_active:] = x[0]
            new_block[channels.index('Y (V)')][:n_active] = y
            new_block[channels.index('Y (V)')][n_active:] = y[0]


            new_block[channels.index('Z (V)')][:n_active:] = z_step + z*(z_steps[i+1] -z_step)
            new_block[channels.index('Z (V)')][n_active:] = z_steps[i+1]

            new_block[channels.index('Camera')][:n_active] = c
            pwm = np.asarray(range(len(new_block[0]))) % n_pwm < n_pwm * pars['UV (%)']
            new_block[channels.index('UV')] = pwm
            new_block[channels.index('UV')][:n_active] = 0

            shutter_delay = np.min([n_transition, pars['DAQ rate (Hz)'] * pars['Shutter delay (s)']])
            new_block[channels.index('Shutter')][:n_active] = np.roll(c, -int(shutter_delay))

            try:
                block = np.append(block, new_block, axis=1)
            except UnboundLocalError:
                block = np.asarray(new_block)

    if pars['LED (V)'] > 0:
        if pars['Steps'] == 0:
            z_center = pars['zc (V)']/10
        else:
            z_center = np.mean(z_steps[:-1])

        new_block = np.zeros([len(channels), n_slice])
        new_block[channels.index('X (V)')] = pars['x0 (V)']
        new_block[channels.index('Y (V)')] = pars['y0 (V)']

        new_block[channels.index('Z (V)')][:n_active] = add_transition(
            pars['z0 (V)'],
            ones * z_center,
            z_steps[0], n_transition)
        new_block[channels.index('Z (V)')][n_active:] = z_steps[0]

        new_block[channels.index('Camera')][n_transition: n_transition + len(ones)] = 1
        pwm = np.asarray(range(len(new_block[0]))) % n_pwm < n_pwm * pars['UV (%)']
        new_block[channels.index('UV')] = pwm
        new_block[channels.index('UV')][:n_active] = 0
        pwm = np.asarray(range(len(ones))) % n_pwm < n_pwm * pars['LED (V)'] / 5
        new_block[channels.index('LED')][n_transition: n_transition + len(ones)] = pwm
        try:
            block = np.append(new_block, block, axis=1)
        except (ValueError, UnboundLocalError) as e:
            block = np.asarray(new_block)
    else:
        block[channels.index('Z (V)')][:n_transition] = add_transition(
            pars['z0 (V)'],
            ones * z_steps[0],
            z_steps[0], n_transition)[:n_transition]

    if show:
        colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'cyan']
        fig, axs = plt.subplots(len(block))
        for i, (s, c, l) in enumerate(zip(block, colors, channels)):
            axs[i].plot(s, color=c)
            axs[i].set_ylabel(l)
        plt.show()

    return block, channels

if __name__ == '__main__':
    filenr = '001'
    filename = rf'C:\Data\noort\210422\data_{filenr}\data_{filenr}.dat'

    filename = 'data_011.log'

    # settings = read_log(filename, True)

    channels, _ = create_signals(filename, show=True)

    # check_image(channels, frames=[2])
    # print(LV_create_scan_pattern(r'C:\Users\noort\PycharmProjects\2Photon-microscope\Scan pattern\test.ini'))
    # x = []
    # x = safe_append(x, np.linspace(5, 10,5))
    # print(x)
    # x = safe_append(x, np.linspace(5, 10, 5))
    # print(x)
