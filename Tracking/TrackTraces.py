import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy import ndimage
import glob
import os
from natsort import natsorted, ns
from scipy.optimize import curve_fit
import Extra_functions as Ef
import cv2
from lmfit import Model, Parameters, minimize
import math
import warnings

## FUNCTIONS FOR ANALYSE IMAGES
def scale_image_u8(image_array, z_range=None):
    if z_range is None:
        z_range = [-2 ** 15, 2 ** 15]
    image_array = np.uint8(np.clip((255 * (image_array - z_range[0]) / (z_range[1] - z_range[0])), 0, 255))
    return image_array


def create_circular_mask(width, size=None, center=None, steepness=3):
    if size is None:
        size = [width, width]
    if center is None:
        center = -0.5 + np.asarray(size) / 2
    x = np.outer(np.linspace(0, size[0] - 1, size[0]), np.ones(size[1]))
    y = np.outer(np.ones(size[0]), np.linspace(0, size[1] - 1, size[1]))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = 1 - 1 / (1 + np.exp(-(r - width / 2) * steepness))
    return mask


def get_roi(image_array, center, width):
    start = (np.asarray(center) - width // 2).astype(int)
    roi = image_array[start[0]: start[0] + width, start[1]: start[1] + width]  # invert axis for numpy array of image
    return roi


def set_roi(image_array, center, roi):
    width = len(roi[0])
    start = (np.asarray(center) - width // 2).astype(int)
    image_array[start[0]: start[0] + width, start[1]: start[1] + width] = roi  # invert axis for numpy array of image
    return image_array


def check_roi(loc, image, width):
    new_loc = np.clip(loc, width // 2, len(image) - width // 2)
    return new_loc


def fit_peak(Z, show=False, center=[0, 0]):
    # Our function to fit is a two-dimensional Gaussian elipse
    def gaussian_elipse(x, y, x0, y0, sigma,aspect_ratio, theta, intensity):
        theta = math.radians(theta)
        sigma_X = sigma
        sigma_Y = sigma * aspect_ratio
        a = (np.cos(theta) ** 2) / (2 * (sigma_X ** 2)) + (np.sin(theta) ** 2) / (2 * (sigma_Y ** 2))
        c = (np.sin(theta) ** 2) / (2 * (sigma_X ** 2)) + (np.cos(theta) ** 2) / (2 * (sigma_Y ** 2))
        b = (-np.sin(2 * theta)) / (4 * (sigma_X ** 2)) + (np.sin(2 * theta)) / (4 * (sigma_Y ** 2))
        Z = intensity * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))
        Z /= np.pi * 2 * sigma_Y * sigma_X
        return Z

    def residual(pars, M, data=None):
        x, y = M
        vals = pars.valuesdict()
        model = gaussian_elipse(x, y, vals['x0'], vals['y0'], vals['sigma'],vals['aspect_ratio'], vals['theta'], vals['intensity'])
        if data is None:
            return model
        return model - data

    Z = np.asarray(Z.T)
    N = len(Z)
    X, Y = np.meshgrid(np.linspace(1, N - 1, N) - N / 2 + center[0],
                       np.linspace(1, N - 1, N) - N / 2 + center[1])
    xdata = np.vstack((X.ravel(), Y.ravel()))
    params = Parameters()
    params.add('x0', value=center[0], min = -N/2, max = N/2)
    params.add('y0', value=center[1], min = -N/2, max = N/2)
    params.add('sigma', value = 3, min = 1, max = N)
    params.add('aspect_ratio', value=1.1, min = 1, max = N/2)
    params.add('theta', value = 0, min = 0, max = 180)
    params.add('intensity', value = np.max(Z)/(8*np.pi), min = 0, max = 1e5)
    out = minimize(residual, params, args=(xdata,), kws={'data': Z.ravel()}, method='differential_evolution')
    out = minimize(residual, out.params , args=(xdata,), kws={'data': Z.ravel()})
    R2 = 1 - out.residual.var() / np.var(Z.ravel())
    p_fitted = [out.params[p].value for p in out.params]
    err_p = [out.params[p].stderr for p in out.params]
    fit = gaussian_elipse(X, Y, *p_fitted)
    p_fitted = np.append(p_fitted, R2)
    p_fitted = np.append(p_fitted, err_p)
    if show and R2>0.5:
        # Plot the 3D figure of the fitted function and the residuals.
        out.params.pretty_print()
       # print(f'R2 = {R2:0.4f}')
        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='afmhot', )
        plt.colorbar(surf, shrink=0.5, aspect=7)
        ax.set_zlim(-20, 90)
        plt.title("raw peak", fontsize=16)
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, fit, cmap='afmhot')
        ax.set_zlim(-20, 90)
        plt.title("fit peak", fontsize=16)
        plt.colorbar(surf, shrink=0.5, aspect=7)
        fig = plt.figure(2)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z - fit, cmap='afmhot')
        ax.set_zlim(-20, 90)
        plt.colorbar(surf, shrink=0.5, aspect=7)
        plt.title("residu peak", fontsize=20)
        #plt.close()
        plt.show()
    return fit, p_fitted, R2


def find_peaks(image_array, width=20, treshold_sd=5, n_traces=200):
    max = np.max(image_array)
    treshold = np.median(image_array) + treshold_sd * np.std(image_array)
    #print(f'Treshold = {treshold_sd} sd = {treshold:.1f}')
    trace_i = 0
    pos = []
    while max > treshold and trace_i < n_traces:
        max_index = np.asarray(np.unravel_index(np.argmax(image_array, axis=None), image_array.shape))
        max_index = check_roi(max_index, image_array, width)
        roi = get_roi(image_array, max_index, width)
        fit, pars,R2 = fit_peak(roi, show=True)
        image_array = set_roi(image_array, max_index, roi - fit)
        # pars[0], pars[1] =  pars[1], pars[0]
        pars[:2] += max_index
        max = np.max(image_array)
        trace_i += 1
        pos.append(pars)

    return np.asarray(pos), image_array


## FUNCTIONS FOR ANALYSE DATASETS
def link_peaks(df, image, n_image, max_dist=5, show=False):
    pp_df = df.copy()
    for j in range(int(n_image) - 1):
        result1 = pp_df.loc[pp_df['Filenr'] == j]
        result2 = pp_df.loc[pp_df['Filenr'] == j + 1]
        result2_coord = result2.loc[:, ['x (pix)', 'y (pix)']]
        for peak_num, peak_values in result1.iterrows():
            result1_coord = peak_values.loc[['x (pix)', 'y (pix)']]
            distance_2 = np.sum((result2_coord - result1_coord) ** 2, axis=1).values
            if np.min(distance_2) < max_dist ** 2:
                index = np.max(np.where(pp_df['Filenr'] == j)[0]) + 1 + np.argmin(distance_2)
                pp_df.loc[index, 'tracenr'] = peak_values.loc['tracenr']
        new_trace_nr = int(pp_df['tracenr'].max()) + 1
        no_trace = np.where(pp_df[:np.max(np.where(pp_df['Filenr'] == j + 1)[0]) + 1] == -1)[0]
        for no_peak_num, _ in enumerate(no_trace):
            pp_df.loc[int(no_trace[no_peak_num]), 'tracenr'] = new_trace_nr + no_peak_num
    if show:
        Ef.plot_link_traces(image, pp_df)
    pp_df.to_csv('dataset_linkpeaks.csv', index=False)
    return pp_df


def link_traces(df, image, show=False, max_dist=5):
    pp_df = df.copy()
    start_df = []
    end_df = []
    sorted_tracelength = df['tracenr'].value_counts().index.values
    for traces in sorted_tracelength:
        start_df.append(pp_df.loc[pp_df['tracenr'] == traces].iloc[:1, :])
    start_df = pd.concat(start_df, ignore_index=True)
    for traces in sorted_tracelength:
        end_df.append(pp_df.loc[pp_df['tracenr'] == traces].iloc[-1:, :])
    end_df = pd.concat(end_df, ignore_index=True)
    for trace, trace_values in end_df.iterrows():
        nstart_df = start_df.drop(np.asarray(start_df.index[start_df['Filenr'] <= trace_values.loc['Filenr']]))
        nstart_df = nstart_df.drop(np.asarray(nstart_df.index[nstart_df['Filenr'] > trace_values.loc['Filenr'] + 3]))
        if len(nstart_df):
            distance_2 = np.sum(
                (nstart_df.loc[:, ['x (pix)', 'y (pix)']] - trace_values.loc[['x (pix)', 'y (pix)']]) ** 2, axis=1)
            if np.min(distance_2) < max_dist ** 2:
                old_tracenr = nstart_df.iloc[np.argmin(distance_2)]['tracenr']
                pp_df['tracenr'] = pp_df['tracenr'].replace([old_tracenr], trace_values.loc['tracenr'])
    if show:
        Ef.plot_link_traces(image, pp_df)
    return pp_df

def msd_trajectory(df, tracenr, pixsize_um,show=False):
    positions = df[df['tracenr'] == tracenr].loc[:, ['x (pix)', 'y (pix)']].values * pixsize_um
    filenrs = df[df['tracenr'] == tracenr].loc[:, ['Filenr']].values
    tau_max = int(np.max(filenrs) - np.min(filenrs))
    squared_displacements = np.zeros((tau_max, tau_max))
    for i, (peak1, filenr1) in enumerate(zip(positions, filenrs)):
        for peak2, filenr2 in zip(positions[i+1:], filenrs[i+1:]):
            current_tau = filenr2 - filenr1 - 1
            last = np.where(squared_displacements[current_tau] == 0)[1][0]
            squared_displacements[current_tau,last] = np.sum((peak2 - peak1) ** 2)
    tau = np.arange(1, tau_max+1)
    tau_max_array = np.full( shape=len(tau),fill_value=tau_max,dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        msd = [np.nanmean(sd[sd != 0]) for sd in squared_displacements]
        msd_error = [np.std(sd[sd !=0]) for sd in squared_displacements ]
    N = tau_max_array / tau
    msd_error=np.divide(msd_error,np.sqrt(N))
    if show:
        plt.errorbar(tau*0.4, msd,fmt = 'o', yerr=msd_error)
        plt.xlim((0, 11))
        plt.ylim((0, 10))
        plt.title(f"msd of tracenr {tracenr}", fontsize=20)
        plt.xlabel(r'$\tau$ (s)', fontsize=16)
        plt.ylabel(r'msd um^2', fontsize=16)
        #plt.savefig(fr"trajectory_{tracenr}\MSD_{tracenr}.png")
        #plt.show()
        plt.close()
    msd_df=pd.DataFrame(msd, columns=[fr'msd_{tracenr}'])
    msd_df[fr'error_{tracenr}']= msd_error
    return msd_df

##Analyse, image, images, dataset and trajectories
def analyse_image(image, file_nr, filefolder, filename, highpass=4, lowpass=1, show=False):
    highpass_image = image - ndimage.gaussian_filter(image, highpass)
    bandpass_image = ndimage.gaussian_filter(highpass_image, lowpass)
    filtered_image = np.copy(bandpass_image)
    peak_positions, cleared_image = find_peaks(bandpass_image, n_traces=1000, width=10, treshold_sd=treshold)
    peak_positions = [np.append([filefolder + fr"\{filename}", file_nr, -1], p) for p in peak_positions]
    if file_nr == 0:
        for i, _ in enumerate(peak_positions):
            peak_positions[i][2] = i
    if show:
        Ef.plot_find_peaks(peak_positions, image, filtered_image, cleared_image, plotting=False, show=True)
    # pp_dataframe = pd.DataFrame(peak_positions, columns=('Filename', 'Filenr', 'tracenr', 'x (pix)', 'y (pix)', 'sigma (pix)', 'intensity (a.u.)', 'error x (pix)',
    # 'error y (pix)', 'error sigma (pix)', 'error intensity (a.u.)'))
    pp_dataframe = pd.DataFrame(peak_positions, columns=('Filename', 'Filenr', 'tracenr', 'x (pix)', 'y (pix)', 'sigma (pix)', 'aspect_ratio','theta', 'intensity (a.u.)','R2', 'error x (pix)',
    'error y (pix)', 'error sigma (pix)', 'error aspect_ratio','error theta','error intensity (a.u.)'))
    return pp_dataframe, filtered_image, cleared_image


def analyse_images(files, first_im, last_im, filefolder):
    empty_pp_df = []
    for num, file in enumerate(files[first_im:last_im]):
        image = np.asarray(tiff.imread(file).astype(float))[200:800,200:800]
        pp_df, filtered_image, cleared_image = analyse_image(image, num, filefolder, file, show=False)
        empty_pp_df.append(pp_df)
        print(num)
    all_pp_df = pd.concat(empty_pp_df, ignore_index=False)
    all_pp_df.to_csv('dataset_pp_v3.csv', index=False)
    return all_pp_df


def filter_image(file, highpass=4, lowpass=1):
    image = np.asarray(tiff.imread(file).astype(float))[200:800,200:800]
    highpass_image = image - ndimage.gaussian_filter(image, highpass)
    bandpass_image = ndimage.gaussian_filter(highpass_image, lowpass)
    filtered_image = np.copy(bandpass_image)
    return filtered_image


def analyse_dataset(df, files):
    df = pd.read_csv(df)
    image = filter_image(files[0])
    sorted_tracelength = df['Filenr'].value_counts().index.values
    link_df_old = link_peaks(df, image, len(sorted_tracelength), show=True)
    link_df = link_df_old.copy()
    for i in range(10):
        link_df = link_traces(link_df, image, show=False)
        if i == 0:
            link_df.to_csv(fr'dataset_linktraces_loop{i}.csv', index=False)
    trace_df = link_df
    trace_df.to_csv('dataset_final_loop.csv', index=False)
    return link_df_old, trace_df

def analyse_trajectories(df, files):
    sorted_tracelength = df['tracenr'].value_counts().index.values
    for i in sorted_tracelength:
        if len(df.loc[df['tracenr'] == i]) < 2:
            sorted_tracelength = np.delete(np.asarray(sorted_tracelength), np.where(sorted_tracelength == i)[0])
    msd_df = []
    taus= np.arange(1,df['Filenr'].max())
    for i in sorted_tracelength:
        single_df= msd_trajectory(df, i, 0.26, show=False)
        msd_df.append(single_df)
    msd_df=pd.concat(msd_df, ignore_index=False, axis=1)
    tau_df=pd.DataFrame(taus*0.4, columns=['tau'])
    msd_df=pd.concat([tau_df, msd_df], ignore_index=False, axis=1)
    #msd_df.to_csv('dataset_msd_all.csv', index=False)
    return

treshold = 5
vmax =50
foldername =fr"C:\Users\Santen\Documents\Images\data_030XX"
os.chdir(foldername)
files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)

dataset_pp = foldername+ "\dataset_pp_v2.csv"
df_pp=pd.read_csv(dataset_pp)

dataset_traces = foldername+ "\dataset_final_loop.csv"
df_traces = pd.read_csv(dataset_traces)

dataset_msd = foldername+ "\dataset_msd_all.csv"
df_msd=pd.read_csv(dataset_msd)

dataset_diffusie = foldername+ "\dataset_diffusie_all.csv"
df_diffusie=pd.read_csv(dataset_diffusie)
## analysis images,dataset and trajectories
analyse_images(files, 0, 2, foldername)
#analyse_dataset(foldername+"\dataset_pp.csv", files)
#analyse_trajectories(df_traces, files)

###validations functions
##histograms
# Ef.show_histogram_values(df_traces,'amplitude (a.u.)', bins=np.linspace(20,150, 200))
# Ef.show_histogram_values(df_traces,"sigma (pix)" ,bins= "auto")
# Ef.show_intensity_histogram_filename(files[0])
#Ef.show_histogram_values(df_pp,'aspect_ratio' ,bins= "auto")
#Ef.show_histogram_values(df_pp,'intensity (a.u.)' ,bins=200)
#Ef.show_histogram_values2(df_diffusie, 'diffusie',0, bins='auto')
#Ef.show_histogram_values2(df_diffusie, 'tracking',1, bins=np.linspace(0,2,40))

##plot
def plot_msd(df):
    x=df['tau']
    msd=df.columns[df.columns.str.startswith('msd')]
    for col_name in df[msd].columns.values:
        y=df[col_name].dropna(0)
        plt.plot(x[:len(y)],y, label=fr'{col_name}')
    plt.title(f"msd", fontsize=20)
    plt.xlabel(r'$\tau$ (s)', fontsize=16)
    plt.ylabel(r'msd um^2', fontsize=16)
    plt.legend()
    plt.show()
    return
#plot_msd(df_msd.iloc[:,:20])

def fit_msd(df):
    x=df['tau']
    msd = df.columns[df.columns.str.startswith('msd')]
    empty_df=[]
    for col_name in df[msd].columns.values:
        y = df[col_name].dropna(0)
        z=np.polyfit(x[:len(y)],y,1)
        print(z)
        p=np.poly1d(z)
        xp=np.linspace(0,15,100)
        plt.plot(x[:len(y)],y,'.', xp,p(xp),'-', label=fr'{col_name}')
        plt.legend()
        plt.xlim((0, 11))
        plt.ylim((0, 10))
        plt.show()
        plt.close()
        df_diffusie_trace=pd.DataFrame(np.asarray(z), columns=[col_name], index=['diffusie','tracking'])
        empty_df.append(df_diffusie_trace)
    df_diffusie = pd.concat(empty_df, ignore_index=False, axis=1)
    #df_diffusie.to_csv('dataset_diffusie_all.csv', index=False)
    return
#fit_msd(df_msd.iloc[:4,:20])


##follow trajectory
def video_traces(traces):
    for trace in traces[:20]:
        Ef.plot_trajectory(df_traces, trace, vmax=vmax, width=50, show_other_peaks=True)
    for trace1 in traces[:20]:
        os.chdir(foldername + fr"\trajectory_{trace1}")
        files = natsorted(glob.glob("*.jpg"), alg=ns.IGNORECASE)
        img_array = []
        for filename in files:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(fr'video_trajactory.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
#video_traces(df_traces['tracenr'].value_counts().index.values)

