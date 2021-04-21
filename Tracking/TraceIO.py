from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from cycler import cycler
import os, re, time, configparser
from datetime import datetime
import pandas as pd
import h5py
#import warnings, tables
from mpl_toolkits.axes_grid1 import Divider, Size
from nptdms import TdmsFile
import glob
from lmfit import Model
import uncertainties as unc
from pathlib import Path

import ForceSpectroscopy as fs

# Allow for spaces in column names
#warnings.simplefilter('ignore', tables.NaturalNameWarning)

DELIMITER = ' > '


class Traces(object):
    """
    This class is to simplify handling data files and to enhance synergy between different projects.
    The basis is that all data consists of (time-)traces. Such a trace is defined by the variables:
    directory, file, trace, channel, label. For example:
    in FORCE SPECTROSCOPY a trace is a particular bead, it's label would be a number, a channel would be
    the extension, z (um).
    in SINGLE-MOLECULE FLUORESCENCE MICROSCOPY a trace is a particular molecule, it's label would be a number,
    a channel would be it's intensity I (kHz), or its position x (um).
    in FLUORESCENCE CORRELATION SPECTROSCOPY a trace is a particular correlation signal, i.e. R632xG524, its label
    could for example be the corresponding composition(nucleosome or salt concentration). The channel would be
    it's correlation curve, G-1 (-).
    Shared traces (Time (s), F (pN), Tau (s), ...) do not have a label.
    All traces are stored in the same HDF5 file, and are organized in groups. HDF5 files (extension *.h5) are
    hierarchical binary files, whose structure has been amply described (https://www.hdfgroup.org/). Download and
    install HDFview (https://www.hdfgroup.org/downloads/hdfview/) to directly see the file contents.
    Access traces in Python by read_channel() and write_channel(). It's good practice to maintain the following file
    structure: ../users/USERNAME/data/DATE/FILE.h5
    a trace is then uniquely defined by: ../users/USERNAME/data/YYYYMMDDD/FILE.h5 /GROUP/TRACE/CHANNEL > LABEL
    Upon initializing an instance of this class, this default structure is generated. The function contents() lists all
    groups/traces in the file.
    Fitted curves can be stored in a group with a name that describes the model. Keep the same channel name to easily
    connect the data to the fit. Fit parameters are stored in a pandas dataframe in the group _PARAMETERS. Standard
    errors are stored in the group _STDERRS. The functions write_fit() and read_fit() ensure that all fit parameters are
    properly stored, overwriting previous values of the same parameters. Parameters are shared on a trace level so
    different models can be fit sharing fixed parameters. Access these parameters individually with the functions
    read_par() and write_par(). Parameters need not to be fit parameters, but could also contain other variables that
    are constant in each particular trace (think of: x0 (um), [NaCL] (mM), Excitation intensity (mW) ...).
    A dataframe of all parameters of the file can be obtained with the function parameters(). Familiarize yourself with
    pandas (https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html), to achieve the most efficient
    downstream processing. It has similar functionality as Excel, and much more...
    For convenience, several strings are generated, that can directly be used to document plots and access and manipulate
    series of data: channelname(), filename(), plotname()
    """


def h5_write_trace(data, filename, channel, label=None):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if label is None:
        label = ' shared'
    with h5py.File(filename, 'a') as hf:
        entry = rf'/traces/{label}/{channel}'
        try:
            stored_data = hf[entry]
            stored_data[...] = data.astype(float)
        except KeyError:
            hf.create_dataset(entry, data=data.astype(float))
        except AttributeError:
            pass


def h5_read_trace(filename, channel, label=' shared'):
    entry = rf'/traces/{label}/{channel}'
    with h5py.File(filename, 'r') as f:
        data = f.get(entry)
        if data is None:
            entry = rf'/traces/ shared/{channel}'
            data = f.get(entry)
        if data is None:
            channel = list(f[f'traces/{label}'].items())[0][0]
            entry = rf'/traces/{label}/{channel}'
            data = [np.nan] * len(f.get(entry))
        data_out = np.asarray(data)
    return data_out


def h5_contents(filename, to_file=False, label=None):
    with h5py.File(filename, 'a') as f:
        if label is None:
            labels = sorted([i[0] for i in list(f['traces'].items())], key=natural_keys)
        else:
            labels = [label]

        channels = []
        for l in labels:
            channels += [i[0] for i in list(f[f'traces/{l}'].items())]
        channels = sorted(set(channels), key=natural_keys)
        try:
            parameters = sorted(list(set([i[0] for i in list(f[r'parameters/values'].items())])))
        except KeyError:
            parameters = []

        if to_file:
            entry = rf'/parameters/label'
            try:
                stored_data = f[entry]
                stored_data[...] = np.asarray(labels).astype('S')
            except KeyError:
                f.create_dataset(entry, data=np.asarray(labels).astype('S'))
    try:
        labels.remove(' shared')
    except:
        pass
    # labels.insert(0, 'shared')
    return channels, labels, parameters


def h5_write_par(filename, label, name, value, error=0, type='local'):
    label = str(label)
    if not os.path.exists(filename):
        print(f'@h5_write_par: File does not exist: {filename}')
        return
    _, labels, _ = h5_contents(filename)

    try:
        index = labels.index(label)
    except ValueError:
        print(f'@h5_write_par: Label does not exist: {label}')
        return

    if type is not 'fit':
        error = np.nan
    items = ['values', 'errors', 'types']
    vars = [value, error, ['fit', 'local', 'global'].index(type)]

    with h5py.File(filename, 'a') as hf:
        for item, var in zip(items, vars):
            entry = rf'/parameters/{item}/{name}'
            try:
                stored_data = hf[entry]
                if type == 'global':
                    data = var * np.ones_like(np.asarray(stored_data))
                else:
                    data = np.asarray(stored_data)
                    data[index] = var
                stored_data[...] = data
            except KeyError:
                if type == 'global':
                    data = [var] * len(labels)
                else:
                    data = [np.nan] * len(labels)
                    data[index] = var
                hf.create_dataset(entry, data=data)


def h5_read_par(filename, label, name):
    if not os.path.exists(filename):
        print(f'@h5_read_par: File does not exist: {filename}')
        return
    _, labels, _ = h5_contents(filename)
    try:
        index = labels.index(label)
    except ValueError:
        print(f'@h5_read_par: Label does not exist: {label}')
        return

    items = ['values', 'errors', 'types']
    types = ['fit', 'local', 'global']
    result = [np.nan] * 3

    with h5py.File(filename, 'r') as hf:
        for i, item in enumerate(items):
            entry = rf'/parameters/{item}/{name}'
            try:
                stored_data = hf[entry]
                result[i] = stored_data[index]
            except (KeyError, ValueError) as error:
                return np.nan, np.nan, 'unknown'
        try:
            type = types[int(result[2])]
        except ValueError:
            type = 'unknown'
    return result[0], result[1], type


def h5_list_pars(filename, label, selection=None):
    if not os.path.exists(filename):
        print(f'@h5_read_par: File does not exist: {filename}')
        return

    _, labels, parameters = h5_contents(filename)

    if selection is not None:
        parameters = [p for p in selection if p in parameters]

    index = labels.index(label)
    types = ['fit', 'local', 'global']

    pars_list = []
    with h5py.File(filename, 'r') as hf:
        for i, par in enumerate(parameters):
            value = hf[rf'/parameters/values/{par}'][index]
            try:
                type = types[int(hf[rf'/parameters/types/{par}'][index])]
            except ValueError:
                continue
            error = hf[rf'/parameters/errors/{par}'][index]
            if type == 'fit':
                pars_list.append([par, round_significance2(value, error), type])
            else:
                pars_list.append([par, f'{value:.6g}', type])
    return pars_list


def prepare_fit(func):
    """
    Prepare all settings for fitting or evaluating a function.
    Make sure the that the function:
    - has a fully filled out doc string
    - the independent variable is called 'x' and comes first
    - each variable, including the independent variable and the return entry, has a proper, readable name.
    - the readable name includes units in brackets
    - the readable name is terminated by a hash and optionally supplemented by a description.
      (i.e. a valid entry would be:  ':param x: F (pN) # The force that was applied.')
    :param func: a function that is implemented in an imported module
    :return: an LMfit model, the corresponding LMfit parameters, and a dictionary of the names
             that describe the used function variables.
    """
    model = Model(getattr(fs, func))
    params = model.make_params()

    doc = getattr(fs, func).__doc__
    names = {}
    names['x'] = doc.split('x:')[1].split('#')[0].strip()
    names['y'] = doc.split('return:')[1].split('#')[0].strip()
    for key, item in params.items():
        names[key] = doc.split(f'{key}:')[1].split('#')[0].strip()
        params[key].vary = False
        try:
            range = doc.split(f'{key}:')[-1].split('\n')[0].split(']')[0].split('[')[-1].strip().split(', ')
            range = np.asarray(range).astype(float)
            params[key].set(min=range[0], max=range[1])
        except ValueError:
            pass
    return model, params, names


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{time.asctime()} @timeit: {method.__name__} took {te - ts:.3f} s')
        return result

    return timed


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''

    def atof(text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def sort_list(alist):
    return alist.sort(key=natural_keys)


def round_significance2(x, stderr=None, par_name=None, delimiter=' ± '):
    """
    Compute proper significant digits. Standard error has 1 significant digit, unless
    that digit equals 1. In that cas it has 2 significant digits. Accuracy of x value
    matches accuracy of error.
    :param x: value
    :param stderr: standard error or standard deviation
    :param par_name: name of variable, preferably with units in brackets: 'x (a.u.)'
    :return: If par_name is provided a string will result: 'x (a.u.) = value +/- err'.
             Otherwise the rounded values of x and stderr.
    """
    value = unc.ufloat(x, stderr)
    if f'{value.std_dev}' == 'nan':
        str_value = f'{value:g}'
    else:
        str_value = f'{value:P}'
        str_value = str_value.replace('(', '')
        str_value = str_value.replace(')', '')
        str_value = str_value.replace('×', ' ')
        str_value = str_value.replace('±', delimiter)

    if par_name is not None:
        str_value = f'{par_name} = ' + str_value

    return str_value


def round_significance(x, stderr=None, par_name=None):
    """
    Compute proper significant digits. Standard error has 1 significant digit, unless
    that digit equals 1. In that cas it has 2 significant digits. Accuracy of x value
    matches accuracy of error.
    :param x: value
    :param stderr: standard error or standard deviation
    :param par_name: name of variable, preferably with units in brackets: 'x (a.u.)'
    :return: If par_name is provided a string will result: 'x (a.u.) = value +/- err'.
             Otherwise the rounded values of x and stderr.
    """

    if stderr is np.NAN or stderr == 0 or stderr is None or stderr == '':
        stderr = np.NAN
    else:
        sig = (np.floor(np.log10(np.abs(stderr)))).astype(float)
        if np.around(stderr / 10. ** sig, 0) == 1.0:
            sig -= 1
        x = np.round(x / (10 ** sig)) * 10 ** sig
        stderr = np.round(stderr / (10 ** sig)) * 10 ** sig

    if par_name is None:
        return x, stderr
    else:
        if stderr == np.NAN:
            txt = f'{par_name} = {x}'
        else:
            e_value = int(f'{x:e}'.split('e')[-1])
            if e_value == 0:
                txt = f"{par_name} = {(x / (10 ** e_value)):g} +/- {(stderr / (10 ** e_value)):g} "
            else:
                txt = f"{par_name} = {(x / (10 ** e_value)):g} +/- {(stderr / (10 ** e_value)):g} 10^{e_value}"
        return txt


def fix_axes(axew=7, axeh=7):
    # axew = axew / 2.54
    # axeh = axeh / 2.54

    topmargin = 0.5

    # lets use the tight layout function to get a good padding size for our axes labels.
    fig = plt.gcf()
    fig.set_size_inches(axew + 1.5, axeh + topmargin)

    ax = plt.gca()
    fig.tight_layout()
    # obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top + topmargin * 1.5
    b = ax.figure.subplotpars.bottom + topmargin * 1.5

    # work out what the new  ratio values for padding are, and the new fig size.
    neww = axew + oldw * (1 - r + l)
    newh = axeh + oldh * (1 - t + b)
    newr = r * oldw / neww
    newl = l * oldw / neww
    newt = t * oldh / newh
    newb = b * oldh / newh

    # right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
    vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    divider = Divider(fig, (0.0, 0.0, 0.95, 0.95), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    # we need to resize the figure


def key_press_action(event):
    print(event.key)
    fig = plt.gcf()
    if event.key == 'z':
        print(fig.texts[0].get_text())
    return


def format_plot(xtitle='x (a.u.)', ytitle='y (a.u.)', title='', xrange=None, yrange=None,
                ylog=False, xlog=False, scale_page=1.0, aspect=0.5, save=None, boxed=True,
                GUI=False, ref='', legend=None, fig=None, ax=None, txt=None):
    # adjust the format to nice looks
    from matplotlib.ticker import AutoMinorLocator
    import os, subprocess

    page_width = 7  # inches ( = A4 width - 1 inch margin)
    margins = (0.55, 0.45, 0.2, 0.2)  # inches
    fontsize = 14

    if fig == None:
        fig = plt.gcf()
    if ax == None:
        ax = plt.gca()

    # Set up figure
    fig_width = page_width * scale_page
    fig_height = (fig_width - (margins[0] + margins[2])) * aspect + margins[1] + margins[3]

    if txt:
        txt_width = 2
    else:
        txt_width = 0

    fig.set_size_inches(fig_width + txt_width, fig_height)

    # Set up axes
    ax_rect = [margins[0] / fig_width]
    ax_rect.append(margins[1] / fig_height)
    ax_rect.append((fig_width - margins[0] - margins[2]) / (fig_width + txt_width))
    ax_rect.append((fig_height - margins[1] - margins[3]) / fig_height)

    # ax_rect = [margins[0] / fig_width,
    #            margins[1] / fig_height,
    #            (fig_width - margins[2] - margins[0]) / fig_width,
    #            (fig_height - margins[3] - margins[1]) / fig_height
    #            ]

    ax.set_position(ax_rect)

    # Add axis titles and frame label; use absolute locations, rather then leaving it up to Matplotlib
    if ref is not None:
        plt.text(ax_rect[1] * 0.15, ax_rect[-1] + ax_rect[1], ref, horizontalalignment='left',
                 verticalalignment='center',
                 fontsize=fontsize * 1.2, transform=fig.transFigure)
    plt.text(ax_rect[0] + 0.5 * ax_rect[2], 0, xtitle, horizontalalignment='center',
             verticalalignment='bottom', fontsize=fontsize, transform=fig.transFigure)
    plt.text(ax_rect[1] * 0.005, ax_rect[1] + 0.5 * ax_rect[3], ytitle, horizontalalignment='left',
             verticalalignment='center', fontsize=fontsize, transform=fig.transFigure, rotation=90)

    if legend is not None:
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.edgecolor"] = 'none'
        plt.rcParams["legend.labelspacing"] = 0.25
        plt.rcParams["legend.handlelength"] = 1
        plt.rcParams["legend.handletextpad"] = 0.25
        plt.legend(legend, prop={'size': fontsize * 0.8}, )

    # fig.canvas.mpl_connect("key_press_event", key_press_action)

    if not boxed:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', axis='both', bottom=True, top=boxed, left=True, right=boxed, direction='in')
    ax.tick_params(which='major', length=4, labelsize=fontsize * 0.8, width=1)
    ax.tick_params(which='minor', length=2, width=1)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)

    if xlog:
        ax.semilogx()
    else:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    if ylog:
        ax.semilogy()
    else:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    if xrange is not None:
        ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)

    if txt:
        pos = (ax.get_xlim()[1] * 1.025, ax.get_ylim()[1])
        if ylog:
            ypos = np.logspace(np.log10(ax.get_ylim()[1]), np.log10(ax.get_ylim()[0]), 25)
        else:
            ypos = np.linspace(ax.get_ylim()[1], ax.get_ylim()[0], 25)
        for t, y in zip(txt, ypos):
            plt.text(pos[0], y, t, size=fontsize / 2)

    if not GUI and save is None: plt.show()

    if save is not None:
        if not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))
        base, ext = os.path.splitext(save)
        if ext == '.emf':
            save = base + '.pdf'
        fig.savefig(save, dpi=600, transparent=True)
        if ext == '.emf':
            try:
                subprocess.call(["C:\Program Files\Inkscape\inkscape.exe", "--file", save, "--export-emf",
                                 base + '.emf'])
                os.remove(save)
            except:
                print('Install Inkscape for conversion to emf.\nPlot saved as pdf in stead.')

    # plt.close()
    return fig, ax


def change_extension(filename, new_extension, label=None, dir=None):
    if dir is not None:
        filename = f'{os.path.dirname(filename)}\\{dir}\\{os.path.basename(filename)}'
    if label is None:
        return f'{os.path.splitext(filename)[0]}.{new_extension}'
    else:
        return Path(f'{os.path.splitext(filename)[0]}_{label}.{new_extension}')


def convert_to_h5(filename, check_existing=True, over_write=False):
    file_out = change_extension(filename, 'h5')
    # pd.set_option('io.hdf.default_format', 'table')

    if check_existing or over_write:
        if os.path.isfile(file_out):
            if over_write:
                os.remove(file_out)
            else:
                print(f'{file_out} already exists, no conversion necessary')
                return file_out

    extension = os.path.splitext(filename)[1][1:]
    if extension == 'dat':
        traces = pd.read_csv(filename, sep='\t', header=(0))
    elif extension == 'tdms':
        tdms_file = TdmsFile(filename)
        # traces = pd.concat(
        # [tdms_file.object('Processed data').as_dataframe(), tdms_file.object('Tracking data').as_dataframe()], sort=True)
        traces = tdms_file.object('Processed data').as_dataframe()

        tracked = tdms_file.object('Tracking data').as_dataframe()
        cols = tracked.columns
        for c in cols:
            if ('Amp' in c):
                label = re.findall(r'\d+', c)
                name = f'Amplitude{label} (a.u.)'
                if name in traces.columns:
                    traces[name] *= np.asarray(tracked[c])
                else:
                    traces = traces.join(tracked[c])
                    traces.rename(columns={c: name})
    else:
        print(f'Convert_to_h5: Format of file *.{extension} is not implemented!')
        return

    for trace in traces.items():
        label = re.findall(r'\d+', trace[0])
        if label == []:
            label = ' shared'
        else:
            label = label[0]
        channel = trace[0].replace(label, '')
        h5_write_trace(trace[1].values, file_out, channel, label)

    # preliminary data processing
    _, labels, _ = h5_contents(file_out, to_file=True)
    # labels.remove('shared')
    for label in labels:
        h5_write_par(file_out, label, ' selected', 1, type='local')

    for channel in ['X (um)', 'Y (um)', 'Z (um)']:
        for label in labels:
            trace = h5_read_trace(file_out, channel, label)
            par = channel[0] + '0' + channel[1:]
            if par == 'Z0 (um)':
                value = np.quantile(trace, 0.05)
                h5_write_par(file_out, label, 'dZ (um)', np.quantile(trace, 0.95) - value, type='local')
            else:
                value = np.median(trace)
            h5_write_par(file_out, label, par, value, type='local')

    return file_out


def convert_log_h5(filename):
    """
    Convert init file to h5. Avoid characters '#' and '/', as these are not properly converted
    :param filename: File to be converted
    """
    f = configparser.ConfigParser()
    f.read(change_extension(filename, 'log'))
    log = dict(f._sections)
    for k in log:
        log[k] = dict(f._defaults, **log[k])
        log[k].pop('__name__', None)

    hf = h5py.File(change_extension(filename, 'h5'), 'a')
    for section in log:
        for item in log[section]:
            entry = f'log/{section}/{item}'
            data = log[section][item]
            try:
                stored_data = hf[entry]
                stored_data[...] = data
            except KeyError:
                hf.create_dataset(entry, data=data)
    hf.close


if __name__ == "__main__":
    #   Intended as examples. Make your own script and import this module to your script.
    #   You can copy the code below as a start of your own script

    if False:
        filenames = [rf'C:\Users\noort\data\ForceTDMS\data_026.h5']
        # filenames = [rf'C:\Users\noort\LTS\werk\Software\Python\testdata\ForceTDMS\\data_025.h5']
        for filename in filenames:
            data = Traces(filename=filename)
            # with data:            # for v in np.random.random(10):
            #     data.write_par('Z0 (um)', 0, 100 * v, 5 * v)
            # print(data.read_par(1, 'Z0 (um)'))
            # print(data.read_df('processed', filter='pN'))
            # t, l = data.contents()
            # print(t)
            # print(l)
            plt.plot(np.arange(0, 10, 1), np.random.random(10))
            # format_plot('x-as', 'y-ax', 'title', width=5, height=5)

    if False:
        filenames = [rf'C:\Users\noort\data\ForceDAT\data_002.dat']

        for filename in filenames:
            filename = convert_dat_h5(filename)

            with Traces(filename) as data:
                for trace in data.contents():
                    print(trace)
                z = data.read_trace(r'C:\Users\noort\data\ForceDAT\data_002.h5/data/processed/Z (um) > 91')
                t = data.read_trace(r'C:\Users\noort\data\ForceDAT\data_002.h5/data/processed/Time (s)')
                plt.plot(t, z)
                format_plot('t (s)', 'z (um)')

    # file_contentsnew_files
    filename = r'C:/Users/noort/PycharmProjects/TraceEditor/data_001.tdms'
    filename = convert_to_h5(filename, over_write=True)
    # channels, labels, parameters = h5_contents(filename, to_file=True)
    # print(channels)
    # print(labels)
    # print(parameters)
    x = h5_read_trace(filename, 'Time (s)')
    print(x)
