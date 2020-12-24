from configparser import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
import bezier


def read_ini(filename):
    parser = ConfigParser()
    parser.read(filename)
    # pars = {section: dict(parser.items(section)) for section in parser.sections()}
    pars = {}
    for section in parser.sections():
        for item in parser.items(section):
            pars[item[0]] = float(item[1])
    return pars


def create_signals(pars):
    t = np.arange(0, pars['exposure (s)'], 1.0 / pars['daq rate (hz)'])
    tau = np.sqrt(t / pars['exposure (s)']) * np.exp(((t / pars['exposure (s)']) ** 2 - 1) / (2 * pars['sigma']))
    x = pars['a spiral (v)'] * tau * np.sin(np.pi * 2 * pars['n spirals'] * tau) + pars['xc (v)']
    y = pars['a spiral (v)'] * tau * np.cos(np.pi * 2 * pars['n spirals'] * tau) + pars['yc (v)']

    t_ramp = np.arange(0, pars['ramp time (s)'], 1.0 / pars['daq rate (hz)'])
    print(len(t_ramp))

    plt.plot(t, x)
    plt.plot(t, y)
    # plt.scatter(x, y)
    plt.show()
    return


if __name__ == '__main__':
    settings = read_ini(r'test.ini')
    for s in settings:
        print(s, settings[s])
    create_signals(settings)
