import fnmatch, warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def save_traces(df, filename):
    columns = [name for name in df.columns if ':' in name]
    columns = sorted(columns, key=lambda x: int(x[:x.index(':')]))
    columns = [name for name in df.columns if ':' not in name] + columns
    df = df.reindex(columns, axis=1)
    df.to_csv(filename, index=False)


def get_color(name):
    colors = {'561': 'green', '637': 'red'}
    color = None
    for key in colors:
        if key in name:
            color = colors[key]
    return color


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


def analyze_traces(df):
    time = df['Time (s)'].values
    dt = np.median(np.diff(time))

    traces = fnmatch.filter(df.columns, '*: I * (a.u.)')

    for trace in tqdm(traces, postfix='Fit HMM'):
        # fig = plt.figure()
        # plt.gcf().canvas.get_renderer()
        intensity = df[trace].values
        # plt.scatter(time, intensity, color="none", edgecolor="green")
        # comment = ''
        try:
            states, mus, sigmas, P, logProb, samples = fitHMM(intensity)
            df[trace.replace(' I ', ' HM ')] = mus[1] * states + mus[0] * (1 - states)
            if mus[1] < 25:
                states *= 0
                mus[0] = np.median(intensity)
                # comment = ', No binding'
            else:
                tau_on = np.min([dt / P[1, 0], time[-1]])
                tau_off = np.min([dt / P[0, 1], time[-1]])
                # comment = f', tau_on = {tau_on:.1f} s, tau_off = {tau_off:.1f} s'
            # plt.plot(time, mus[1] * states + mus[0] * (1 - states), color="green")
        except ValueError:
            pass

    #     plt.title(f'{trace}{comment}')
    #     plt.ylim(-50, 500)
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Intensity (a.u.)')
    #     plt.show()
    #     if filename:
    #         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #         img = np.reshape(img, fig.canvas.get_width_height()[::-1] + (3,))
    #         ims.append(img)
    #
    # if filename:
    #     save_movie(filename.replace('.csv', '_binding.mp4'), ims, 1)

    save_traces(df, filename)

    if False:
        traces = fnmatch.filter(df.columns, '15: * (a.u.)')
        for trace in traces:
            if ' I ' in trace:
                plt.scatter(df['Time (s)'], df[trace], facecolors='none', edgecolors=get_color(trace))
            else:
                plt.plot(df['Time (s)'], df[trace], color=get_color(trace))
        plt.ylim((-50,100))
        plt.show()


if __name__ == '__main__':
    filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.csv'
    df = pd.read_csv(filename)
    # columns = [name for name in df.columns if 'I 637' in name]
    # for c in columns:
    #     hist, edges = np.histogram(df[c],bins = 100, range=(-50, 100))
    #     try:
    #         histogram += hist
    #     except NameError:
    #         histogram = hist
    #
    # plt.plot(edges[1:], histogram)
    # # plt.semilogy()
    # plt.show()

    analyze_traces(df)
