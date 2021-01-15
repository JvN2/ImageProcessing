import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.ndimage
from configparser import ConfigParser
import ProcessImages.ImageIO as ioi

def read_all_parameters(filename):
    parser = ConfigParser()
    parser.read(filename)
    # confdict = {section: dict(parser.items(section)) for section in parser.sections()}
    confdict = {}
    for section in parser.sections():
        for item in parser.items(section):
            confdict[item[0]] = float(item[1])
    return confdict

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


def check_image(channels, n_pix=300, n_spots=5, d_spots=30, pix_volt=70, frames=None):
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
    # plt.plot(im2)
    plt.show()


def create_signals(pars, show=False):
    t = np.arange(0, pars['exposure (s)'], 1.0 / pars['daq rate (hz)'])
    n_slice = int(pars['slice (s)'] * pars['daq rate (hz)'])
    n_transition = int(pars['ramp time (s)'] * pars['daq rate (hz)'])
    n_active = len(t) + 2 * n_transition
    n_slice = np.max([n_slice, n_active])
    n_pwm = 50

    tau = np.sqrt(t / pars['exposure (s)']) * np.exp(((t / pars['exposure (s)']) ** 2 - 1) / (2 * pars['sigma']))
    if pars['slim'] == 0:
        x = 0.455*pars['a spiral (v)'] * tau * np.sin(np.pi * 2 * pars['n spirals'] * tau) + pars['xc (v)']
        y = 0.455*pars['a spiral (v)'] * tau * np.cos(np.pi * 2 * pars['n spirals'] * tau) + pars['yc (v)']
        c_block = [add_transition(0, np.ones_like(x), 0, n_transition, kind='zero')]
        x_block = [add_transition(pars['x0 (v)'], x, pars['x0 (v)'], n_transition)]
        y_block = [add_transition(pars['y0 (v)'], y, pars['y0 (v)'], n_transition)]
    else:
        for phase in range(int(pars['slim'])):
            x = pars['a spiral (v)'] * tau * np.sin(np.pi * 2 * pars['n spirals'] * tau)
            # y = 0.15 * pars['a spiral (v)'] * tau * np.cos(np.pi * 2 * pars['n spirals'] * tau)
            y = 0.5 * np.ones_like(x) * phase / int(pars['slim'])

            a = 0.63 * pars['a spiral (v)']
            t0 = np.max(t) / 2
            x = np.exp(-(t - t0) ** 2 / (0.5 * t0) ** 2)
            x = np.diff(x)
            x = np.append(x, 0)
            x /= np.max(x)
            x *= a
            x = np.roll(x, len(t) // 2)
            x = np.linspace(a, -a, len(x) )

            tmp = x
            angle = np.pi / 6
            angle = -np.pi/2
            # angle = (5/6) *np.pi
            x = np.cos(angle) * x + np.sin(angle) * y
            y = np.sin(angle) * tmp + np.cos(angle) * y

            x += pars['xc (v)']
            y += pars['yc (v)']

            if phase == 0:
                c_block = [add_transition(0, np.ones_like(x), 0, n_transition, kind='zero')]
                x_block = [add_transition(pars['x0 (v)'], x, pars['x0 (v)'], n_transition)]
                y_block = [add_transition(pars['y0 (v)'], y, pars['y0 (v)'], n_transition)]
            else:
                c_block.append(add_transition(0, np.ones_like(x), 0, n_transition, kind='zero'))
                x_block.append(add_transition(pars['x0 (v)'], x, pars['x0 (v)'], n_transition))
                y_block.append(add_transition(pars['y0 (v)'], y, pars['y0 (v)'], n_transition))

    ones = np.ones_like(x)

    channels = ['X (V)', 'Y (V)', 'Z (V)', 'LED (V)', 'Camera', 'UV']
    for step in range(int(pars['nsteps'])):
        for x, y, c in zip(x_block, y_block, c_block):
            new_block = np.zeros([len(channels), n_slice])
            new_block[channels.index('X (V)')][:n_active] = x
            new_block[channels.index('X (V)')][n_active:] = x[0]
            new_block[channels.index('Y (V)')][:n_active] = y
            new_block[channels.index('Y (V)')][n_active:] = y[0]

            new_block[channels.index('Z (V)')][:n_active] = add_transition(
                pars['z0 (v)'],
                ones * (step + 1) * pars['zstep (v)'],
                pars['z0 (v)'], n_transition)
            new_block[channels.index('Z (V)')][n_active:] = pars['z0 (v)']

            new_block[channels.index('Camera')][:n_active] = c
            pwm = np.asarray(range(len(new_block[0]))) % n_pwm < n_pwm * pars['uv (%)'] / 100
            new_block[channels.index('UV')] = pwm
            new_block[channels.index('UV')][:n_active] = 0

            try:
                block = np.append(block, new_block, axis=1)
            except UnboundLocalError:
                block = np.asarray(new_block)

    if pars['led (v)'] > 0:
        new_block = np.zeros([len(channels), n_slice])
        new_block[channels.index('X (V)')] = pars['x0 (v)']
        new_block[channels.index('Y (V)')] = pars['y0 (v)']
        new_block[channels.index('Z (V)')][:n_active] = add_transition(
            pars['z0 (v)'],
            ones * pars['zstep (v)'],
            pars['z0 (v)'], n_transition)
        new_block[channels.index('Z (V)')][n_active:] = pars['z0 (v)']

        new_block[channels.index('Camera')][:n_active] = c
        pwm = np.asarray(range(len(new_block[0]))) % n_pwm < n_pwm * pars['uv (%)'] / 100
        new_block[channels.index('UV')] = pwm
        new_block[channels.index('UV')][:n_active] = 0
        pwm = np.asarray(range(len(ones))) % n_pwm < n_pwm * pars['led (v)'] / 5
        new_block[channels.index('LED (V)')][n_transition: n_transition + len(ones)] = 5 * pwm
        try:
            block = np.append(new_block, block, axis=1)
        except ValueError:
            block = np.asarray(new_block)

    # plt.scatter(block[channels.index('X (V)')], block[channels.index('Y (V)')], marker='.')
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # plt.show()

    if show:
        plt.plot(block.T)
        plt.legend(channels)
        plt.show()
    return block


def LV_create_scan_pattern(filename):
    settings = read_all_parameters(filename)
    channels = create_signals(settings, show=False)
    return channels


if __name__ == '__main__':
    settings = read_all_parameters(r'test.ini')

    # for s in settings:
    #     print(s, settings[s])
    channels = create_signals(settings, show=True)
    check_image(channels, frames=[0,1,2])
    # print(LV_create_scan_pattern(r'C:\Users\noort\PycharmProjects\2Photon-microscope\Scan pattern\test.ini'))
    # x = []
    # x = safe_append(x, np.linspace(5, 10,5))
    # print(x)
    # x = safe_append(x, np.linspace(5, 10, 5))
    # print(x)
