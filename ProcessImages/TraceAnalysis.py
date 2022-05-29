import fnmatch
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm

import ProcessImages.ImageIO as iio

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class Traces():
    def __init__(self, filename=None):
        self.from_file(filename)
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def from_file(self, filename):
        self.filename = Path(filename).with_suffix('.xlsx')
        try:
            self.traces = pd.read_excel(self.filename, 'traces')
        except ValueError:
            self.traces = pd.DataFrame()
        except FileNotFoundError:
            self.traces=pd.DataFrame()
            self.pars=pd.DataFrame()
            self.globs=pd.DataFrame().squeeze()
            return
        try:
            self.pars = pd.read_excel(self.filename, 'parameters', index_col=0)
        except:
            self.pars = pd.DataFrame()
        try:
            self.globs = pd.read_excel(self.filename, 'globals', index_col=0).squeeze()
            for key, value in self.globs.iteritems():
                if isinstance(value, str):
                    if value[0] == '[' and value[-1] == ']':
                        ar = value[1:-1].split(', ')
                        ar = [a[1:-1] if a[0] == "'" else a for a in ar]
                        self.globs.at[key] = ar

        except ValueError:
            self.globs = pd.DataFrame()
        return


    def to_file(self, filename=None):
        if filename is not None:
            self.filename = Path(filename).with_suffix('.xlsx')
        with pd.ExcelWriter(self.filename) as writer:
            if not self.traces.empty:
                self.traces.to_excel(writer, sheet_name='traces', index=False)
            if not self.pars.empty:
                self.pars.to_excel(writer, sheet_name='parameters')
            if not self.globs.empty:
                self.globs.to_excel(writer, sheet_name='globals')
        return


def save_hdf(filename, traces=None, pars=None, globs=None):
    filename = filename[:-4] + '.hdf'
    if traces is not None:
        columns = [name for name in traces.columns if ':' in name]
        columns = sorted(columns, key=lambda x: int(x[:x.index(':')]))
        columns = [name for name in traces.columns if ':' not in name] + columns
        traces = traces.reindex(columns, axis=1)
        traces.to_hdf(filename, 'traces')
    if pars is not None:
        pars.to_hdf(filename, 'parameters')
    if globs is not None:
        pd.Series(globs).to_hdf(filename, 'globals')

    with pd.ExcelWriter(filename.replace('hdf', 'xlsx')) as writer:
        if traces is not None:
            traces.to_excel(writer, sheet_name='traces', index=False)
        if pars is not None:
            pars.to_excel(writer, sheet_name='parameters')
        if globs is not None:
            pd.Series(globs).to_excel(writer, sheet_name='globals')

    return filename


def read_hdf(filename):
    filename = filename[:-4] + '.hdf'
    traces = pd.read_hdf(filename, key='traces')
    pars = pd.read_hdf(filename, key='parameters')
    globs = dict(pd.read_hdf(filename, 'globals'))
    return traces, pars, globs


def get_color(name):
    colors = {'561': 'green', '637': 'red', '488': 'blue'}
    color = None
    for key in colors:
        if key in name:
            color = colors[key]
    return color


def get_label(name, sep=': '):
    return int(name.split(sep)[0]) if sep in name else None


def fitHMM(Q):
    nSamples = len(Q)
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q, [len(Q), 1]))

    # classify each observation as state 0 or 1
    hidden_states = model.predict(np.reshape(Q, [len(Q), 1]))

    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    # find log-likelihood of Gaussian HMM
    logProb = model.score(np.reshape(Q, [len(Q), 1]))

    # generate nSamples from Gaussian HMM
    samples = model.sample(nSamples)

    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states

    return hidden_states, mus, sigmas, P, logProb, samples


def hidden_markov_fit(data, traces, treshold=None):
    time = data.traces['Time (s)'].values
    dt = np.median(np.diff(time))

    for trace in tqdm(traces, postfix='Fit HMM'):
        label = int(trace.split(':')[0])
        color = trace.split(' I ')[-1].split(' (a.u.)')[0]
        try:
            states, mus, sigmas, P, logProb, samples = fitHMM(data.traces[trace].values)
            if treshold is not None:
                if (mus[1] - mus[0]) < treshold:
                    states *= 0
                    mus[0] = np.median(data.traces[trace].values)
                else:
                    data.pars.at[label, f'{color}: I (a.u.)'] = mus[1] - mus[0]
                    if P[1, 0] != 0:
                        data.pars.at[label, f'{color}: Tau on (s)'] = np.min([dt / P[1, 0], time[-1]])
                    if P[0, 1] != 0:
                        data.pars.at[label, f'{color}: Tau off (s)'] = np.min([dt / P[0, 1], time[-1]])
        except ValueError:
            mus[0] = np.median(data.traces[trace].values)
            states = data.traces[trace].values * 0
        data.traces[trace.replace(' I ', ' HM ')] = mus[1] * states + mus[0] * (1 - states)
    return


if __name__ == '__main__':
    filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.csv'
    # filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV13_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.29.35.ims'

    data = Traces(filename)

    if len(fnmatch.filter(data.traces.columns, '*: HM * (a.u.)')) == 0:
        # Hidden Markov fit
        selected_traces = fnmatch.filter(data.traces.columns, '*: I * (a.u.)')
        hidden_markov_fit(data, selected_traces, treshold=100)
        data.to_file()

    if False:  # histogram
        save_hdf(filename, traces=traces, pars=pars, globs=globs)
        for color in globs['Colors']:
            selected_traces = fnmatch.filter(traces.columns, f'*: HM {color} (a.u.)')
            for trace in selected_traces:
                pars.at[get_label(trace), f'I {color} (a.u.)'] = traces[trace].max() - traces[trace].min()
            plt.hist(pars[f'I {color} (a.u.)'], color=get_color(color), range=(-100, 500), bins=100)
        plt.show()

    if True:  # movie of plots
        movie = iio.Movie()
        with movie(str(data.filename).replace('.xlsx','_traces.mp4')):
            for trace_nr in tqdm(data.pars.index, postfix='Save plots'):
                for color in data.globs['Colors']:
                    trace_name = f'{trace_nr}: HM {color} (a.u.)'
                    offset = data.traces[trace_name].min()
                    plt.scatter(data.traces['Time (s)'], data.traces[trace_name.replace(' HM ', ' I ')] - offset,
                                edgecolors=get_color(color), s=40, facecolors='none')
                    plt.plot(data.traces['Time (s)'], data.traces[trace_name] - offset, color=get_color(color),
                             label=color)
                plt.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Intensity (a.u.)')
                plt.ylim((-100, 900))
                plt.title(f'trace {trace_nr}')

                movie.add_plot()