from configparser import ConfigParser
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate


def read_ini(filename):
    parser = ConfigParser()
    parser.read(filename)
    # confdict = {section: dict(parser.items(section)) for section in parser.sections()}
    confdict = {}
    for section in parser.sections():
        for item in parser.items(section):
            confdict[item[0]] = float(item[1])
    return confdict


def create_transtion(x, n, x_start=None, x_end=None):
    if x_end:
        t = np.asarray([-2, -1, n + 1, n + 2])
        y = np.asarray([x[-2], x[-1], x_end, x_end])
        f = interpolate.interp1d(t, y, kind='quadratic')
        tnew = np.arange(0, n)
        ynew = f(tnew)
        x = np.append(x, ynew)

    if x_start:
        t = np.asarray([-n - 2, -n - 1, 0, 1])
        y = np.asarray([x_start, x_start, x[0], x[1]])
        f = interpolate.interp1d(t, y, kind='quadratic')
        tnew = np.arange(0, n) - n
        ynew = f(tnew)
        x = np.append(ynew, x)
    return x


def create_signals(pars):
    t = np.arange(0, pars['exposure (s)'], 1.0 / pars['daq rate (hz)'])
    tau = np.sqrt(t / pars['exposure (s)']) * np.exp(((t / pars['exposure (s)']) ** 2 - 1) / (2 * pars['sigma']))
    x = pars['a spiral (v)'] * tau * np.sin(np.pi * 2 * pars['n spirals'] * tau) + pars['xc (v)']
    y = pars['a spiral (v)'] * tau * np.cos(np.pi * 2 * pars['n spirals'] * tau) + pars['yc (v)']

    n = pars['ramp time (s)'] * pars['daq rate (hz)']
    x = create_transtion(x, n, x_end=pars['x0 (v)'], x_start=pars['x0 (v)'])
    y = create_transtion(y, n, x_end=pars['y0 (v)'], x_start=pars['y0 (v)'])

    plt.plot(x)
    plt.plot(y)
    # plt.scatter(x, y)
    plt.show()
    return


if __name__ == '__main__':
    settings = read_ini(r'test.ini')
    # for s in settings:
    #     print(s, settings[s])
    create_signals(settings)
