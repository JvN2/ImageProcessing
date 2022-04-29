from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ProcessImages.CoSMoS import save_cv2_movie

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

def analyze_traces(filename, save = False):
    df = pd.read_csv(filename)
    time = df['Time (s)'].values
    dt = np.median(np.diff(time))
    ims = []

    traces = [trace for trace in df.columns if '561' in trace]
    for trace in traces:
        fig = plt.figure()
        plt.gcf().canvas.get_renderer()
        intensity = df[trace].values
        plt.scatter(time, intensity, color="none", edgecolor="green")
        comment = ''
        try:
            states, mus, sigmas, P, logProb, samples = fitHMM(intensity)
            if mus[1] < 200:
                states *= 0
                mus[0] = np.median(intensity)
                comment = ', No binding'
            else:
                tau_on = np.min([dt / P[1, 0], time[-1]])
                tau_off = np.min([dt / P[0, 1], time[-1]])
                comment = f', tau_on = {tau_on:.1f} s, tau_off = {tau_off:.1f} s'
            plt.plot(time, mus[1] * states + mus[0] * (1 - states), color="green")
        except ValueError:
            pass

        plt.title(f'{trace}{comment}')
        plt.ylim(-200, 2500)
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity (a.u.)')
        plt.show()
        if save:
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = np.reshape(img, fig.canvas.get_width_height()[::-1] + (3,))
            ims.append(img)

    if save:
        save_cv2_movie(filename.replace('.csv', '_binding.mp4'), ims, 1)


filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV14_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.38.58.csv'
# filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV13_512_Exp50g60r50o_Rep100_Int130_2022-04-10_Protocol 4_16.29.35.csv'
filename = r'C:\Users\noort\Downloads\Slide1_Chan1_FOV3_512_Exp50r50o_pr%70r40o_Rep100_Int120_2022-04-22_Protocol 5_14.33.32.csv'
# filename = r'C:\Users\noort\Downloads\Slide1_Chan2_FOV1_512_Exp50o50r_pr%40o70r_Rep100_Int120_T+P+P_2022-04-22_Protocol 5_15.08.02.csv'

analyze_traces(filename, save=True)

