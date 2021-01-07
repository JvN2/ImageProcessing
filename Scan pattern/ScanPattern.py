from configparser import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.ndimage


def read_ini(filename):
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


def check_image(channels, n=250):
    w = 10
    x = np.asarray([list(range(n))] * n)
    im = np.exp(-(x - n / 2) ** 2 / w ** 2) * np.exp(-(x.T - n / 2) ** 2 / w ** 2)
    im *= im

    n_spots = 5
    d_spots = 40

    loc = np.zeros((2, n_spots ** 2))
    for i in range(n_spots):
        loc[0][i * n_spots:(i + 1) * n_spots] = np.arange(0, n_spots * d_spots, d_spots) + 0.5 * d_spots * (i % 2)
        loc[1][i * n_spots:(i + 1) * n_spots] = np.ones(n_spots) * d_spots * i * np.sqrt(3) / 2

    for i, _ in enumerate(loc):
        loc[i] = loc[i] - np.mean(loc[i])

    im2 = im * 0
    for x in loc.T:
        im2 += scipy.ndimage.shift(im, x)

    im = im2
    im2 = im*0
    for x in channels.T:
        im2 += scipy.ndimage.shift(im, x)
    plt.imshow(im2.T, origin='lower')
    plt.show()


def create_signals(pars):
    t = np.arange(0, pars['exposure (s)'], 1.0 / pars['daq rate (hz)'])
    n_slice = int(pars['slice (s)'] * pars['daq rate (hz)'])
    n_transition = int(pars['ramp time (s)'] * pars['daq rate (hz)'])
    n_active = len(t) + 2 * n_transition
    n_slice = np.max([n_slice, n_active])
    n_pwm = 50

    if pars['slim'] == 1:
        tau = np.sqrt(t / pars['exposure (s)']) * np.exp(((t / pars['exposure (s)']) ** 2 - 1) / (2 * pars['sigma']))
        x = pars['a spiral (v)'] * tau * np.sin(np.pi * 2 * pars['n spirals'] * tau) + pars['xc (v)']
        y = pars['a spiral (v)'] * tau * np.cos(np.pi * 2 * pars['n spirals'] * tau) + pars['yc (v)']
    else:
        tau = np.sqrt(t / pars['exposure (s)']) * np.exp(((t / pars['exposure (s)']) ** 2 - 1) / (2 * pars['sigma']))
        x = pars['a spiral (v)'] * tau * np.sin(np.pi * 2 * pars['n spirals'] * tau) + pars['xc (v)']
        y = pars['a spiral (v)'] * tau * np.cos(np.pi * 2 * pars['n spirals'] * tau) + pars['yc (v)']
    ones = np.ones_like(x)
    x = add_transition(pars['x0 (v)'], x, pars['x0 (v)'], n_transition)
    y = add_transition(pars['y0 (v)'], y, pars['y0 (v)'], n_transition)
    y[1] = y[0]
    x[1] = x[0]
    c = add_transition(0, ones, 0, n_transition, kind='zero')

    channels = ['X (V)', 'Y (V)', 'Z (V)', 'LED (V)', 'Camera', 'UV']
    for step in range(int(pars['nsteps'])):
        new_block = np.zeros([len(channels), n_slice])
        new_block[channels.index('X (V)')][:n_active] = x
        new_block[channels.index('X (V)')][n_active:] = x[0]
        new_block[channels.index('Y (V)')][:n_active] = y
        new_block[channels.index('Y (V)')][n_active:] = y[0]
        new_block[channels.index('Z (V)')][:n_active] = add_transition(
            np.max([0, pars['zstep (v)'] * (step - 1)]),
            ones * step * pars['zstep (v)'],
            pars['zstep (v)'] * step, n_transition)
        new_block[channels.index('Z (V)')][n_active:] = pars['zstep (v)'] * step
        new_block[channels.index('Camera')][:n_active] = c
        pwm = np.asarray(range(len(new_block[0]))) % n_pwm < n_pwm * pars['uv (%)'] / 100
        new_block[channels.index('UV')] = pwm
        new_block[channels.index('UV')][:n_active] = 0
        try:
            block = np.append(block, new_block, axis=1)
        except NameError:
            block = new_block

    if pars['led (v)'] > 0:
        new_block = np.zeros([len(channels), n_slice])
        new_block[channels.index('X (V)')] = pars['x0 (v)']
        new_block[channels.index('Y (V)')] = pars['y0 (v)']
        new_block[channels.index('Z (V)')] = pars['z0 (v)']
        new_block[channels.index('Camera')][:n_active] = c
        pwm = np.asarray(range(len(new_block[0]))) % n_pwm < n_pwm * pars['uv (%)'] / 100
        new_block[channels.index('UV')] = pwm
        new_block[channels.index('UV')][:n_active] = 0
        pwm = np.asarray(range(len(ones))) % n_pwm < n_pwm * pars['led (v)'] / 5
        new_block[channels.index('LED (V)')][n_transition: n_transition + len(ones)] = 5 * pwm
        try:
            block = np.append(new_block, block, axis=1)
        except NameError:
            block = new_block

    # plt.scatter(block[channels.index('X (V)')], block[channels.index('Y (V)')], marker='.')
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    plt.plot(block.T)
    plt.legend(channels)
    plt.show()
    return block


if __name__ == '__main__':
    settings = read_ini(r'test.ini')
    # for s in settings:
    #     print(s, settings[s])
    channels = create_signals(settings)
    check_image(35.0*channels[:2])
