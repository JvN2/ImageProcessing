import fnmatch
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm

import ProcessImages.ImageIO as iio

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


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


def analyze_traces(df, treshold=None):
    time = df['Time (s)'].values
    dt = np.median(np.diff(time))

    traces = fnmatch.filter(df.columns, '*: I * (a.u.)')
    for trace in tqdm(traces, postfix='Fit HMM'):
        intensity = df[trace].values
        try:
            states, mus, sigmas, P, logProb, samples = fitHMM(intensity)
            if treshold is not None:
                if (mus[1] - mus[0]) < treshold:
                    states *= 0
                    mus[0] = np.median(intensity)
                    # comment = ', No binding'
                else:
                    tau_on = np.min([dt / P[1, 0], time[-1]])
                    tau_off = np.min([dt / P[0, 1], time[-1]])
                    # comment = f', tau_on = {tau_on:.1f} s, tau_off = {tau_off:.1f} s'
        except ValueError:
            mus[0] = np.median(intensity)
            states = intensity * 0
        df[trace.replace(' I ', ' HM ')] = mus[1] * states + mus[0] * (1 - states)
    return df


if __name__ == '__main__':
    filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.csv'
    # filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV13_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.29.35.ims'

    traces = analyze_traces(read_hdf(filename)[0], 100)
    save_hdf(filename, traces=traces)

    traces, pars, globs = read_hdf(filename)
    # save_hdf(filename, traces=traces, pars=pars, globs=globs)
    # for color in globs['Colors']:
    #     selected_traces = fnmatch.filter(traces.columns, f'*: HM {color} (a.u.)')
    #     for trace in selected_traces:
    #         pars.at[get_label(trace), f'I {color} (a.u.)'] = traces[trace].max() - traces[trace].min()
    #     plt.hist(pars[f'I {color} (a.u.)'], color=get_color(color), range=(-100, 500), bins=100)
    # plt.show()


    movie = iio.Movie()
    with movie(filename[:-4] + '_traces.mp4', 2):
        for label in tqdm(pars.index, postfix='Save plots'):
            for color in globs['Colors']:
                offset = traces[f'{label}: HM {color} (a.u.)'].min()
                plt.scatter(traces['Time (s)'], traces[f'{label}: I {color} (a.u.)'] - offset,
                            edgecolors=get_color(color), s=40, facecolors='none')
                plt.plot(traces['Time (s)'], traces[f'{label}: HM {color} (a.u.)'] - offset, color=get_color(color),
                         label=color)
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Intensity (a.u.)')
            plt.ylim((-100, 900))
            plt.title(f'trace {label}')
            try:
                movie.add_plot()
            except Exception as inst:
                print(inst)
