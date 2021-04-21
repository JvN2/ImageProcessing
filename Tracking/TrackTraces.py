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
    # Our function to fit is a two-dimensional Gaussian
    def gaussian(x, y, x0, y0, sigma, A):
        return A * np.exp(-((x - x0) / sigma) ** 2 - ((y - y0) / sigma) ** 2)
    # This is the callable that is passed to curve_fit. M is a (2,N) array
    # where N is the total number of data points in Z, which will be ravelled
    # to one dimension.
    def _gaussian(M, *args):
        x, y = M
        arr = gaussian(x, y, *args)
        return arr

    Z = np.asarray(Z.T)
    N = len(Z)
    X, Y = np.meshgrid(np.linspace(0, N - 1, N) - N / 2 + center[0],
                       np.linspace(0, N - 1, N) - N / 2 + center[1])
    p = (center[0], center[1], 2, np.max(Z))

    xdata = np.vstack((X.ravel(), Y.ravel()))
    popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p)
    fit = gaussian(X, Y, *popt)
    err = np.sqrt(np.diag(pcov))
    if show:
        # Plot the 3D figure of the fitted function and the residuals.
        print(f'fit result: {err}')
        Ef.plot_fit_peaks(popt, err, X, Y, Z, fit)
    popt = np.append(popt, err)
    return fit, popt, err

def find_peaks(image_array, width=20, treshold_sd=7, n_traces=200):
    max = np.max(image_array)
    treshold = np.median(image_array) + treshold_sd * np.std(image_array)
    trace_i = 0
    pos = []
    while max > treshold and trace_i < n_traces:
        max_index = np.asarray(np.unravel_index(np.argmax(image_array, axis=None), image_array.shape))
        max_index = check_roi(max_index, image_array, width)
        roi = get_roi(image_array, max_index, width)
        fit, pars, _ = fit_peak(roi, show=False)
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
        result1 =pp_df.loc[pp_df['Filenr']==j]
        result2 = pp_df.loc[pp_df['Filenr']==j+1]
        result2_coord =result2.loc[:,['x (pix)', 'y (pix)']]
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
    pp_df.to_csv('dataset_linkpeaks_v2.csv', index=False)
    return pp_df

def link_traces(df, image, show=False, max_dist=5):
    pp_df = df.copy()
    range_traces =[]
    start_df=[]
    end_df =[]
    [range_traces.append(i) for i in pp_df.loc[:,'tracenr'].to_numpy() if i not in range_traces]
    for traces in range_traces:
        start_df.append(pp_df.loc[pp_df['tracenr'] == traces].iloc[:1,:])
    start_df =pd.concat(start_df, ignore_index=True)
    for traces in range_traces:
        end_df.append(pp_df.loc[pp_df['tracenr'] == traces].iloc[-1:, :])
    end_df = pd.concat(end_df, ignore_index=True)
    for trace, trace_values in end_df.iterrows():
        nstart_df=start_df.drop(np.asarray(start_df.index[start_df['Filenr']<=trace_values.loc['Filenr']]))
        if len(nstart_df):
            distance_2 = np.sum((nstart_df.loc[:, ['x (pix)', 'y (pix)']] - trace_values.loc[['x (pix)', 'y (pix)']]) ** 2, axis=1)
            if np.min(distance_2) < max_dist ** 2:
                old_tracenr = nstart_df.iloc[np.argmin(distance_2)]['tracenr']
                pp_df['tracenr'] = pp_df['tracenr'].replace([old_tracenr], trace_values.loc['tracenr' ] )
    if show:
        Ef.plot_link_traces(image, pp_df)
    return pp_df

## Functions for analysis trajectories
def MSD_traj(traj_df, diffusie_df,tracenr,cf_pos, cf_time, show =False):
    trace_df_traj = traj_df.loc[traj_df['tracenr'] == tracenr]
    n_steps =len(trace_df_traj)
    steps = range(1,n_steps)
    MSD_array=np.zeros((4, n_steps))
    xcoords = trace_df_traj.loc[:, 'x (pix)']
    ycoords = trace_df_traj.loc[:, 'y (pix)']
    rad = np.sqrt(xcoords ** 2 + ycoords ** 2)
    coordsxy = np.append([rad],[xcoords], axis=0)
    coordsxy = np.append(coordsxy, [ycoords], axis=0)
    for step in steps:
        a =np.mean(trace_df_traj.loc[trace_df_traj.index[step]:,'Filenr'].values-trace_df_traj.loc[:trace_df_traj.index[-1-step],'Filenr'].values)
        MSD_array[0, step] =a
        i =1
        for axis in [coordsxy[0],coordsxy[1], coordsxy[2]]:
            b =np.mean((axis[step:]-axis[:-step])**2)
            MSD_array[i, step] = b
            i+=1
    if show:
        Ef.plot_MSD(MSD_array)

    return MSD_array



def diffusie(traj_df,diffusie_df,tracenr):
    MSD_array = MSD_traj(traj_df, diffusie_df, tracenr, 0.26, 0.4, show=False)
    index = np.where([diffusie_df['tracenr'] == tracenr][0])[0]
    tau = np.mean(MSD_array[0])
    MSDxy = np.mean(MSD_array[1])
    MSDx= np.mean(MSD_array[2])
    MSDy=np.mean(MSD_array[3])
    diffusie_df.loc[index, 'Tau (s)'] = tau
    diffusie_df.loc[index, 'MSDxy'] = MSDxy
    diffusie_df.loc[index, 'MSDx'] = MSDx
    diffusie_df.loc[index, 'MSDy'] = MSDy
    return diffusie_df

##Analyse, image, images, dataset and trajectories
def analyse_image(image, file_nr, filefolder,filename, highpass=4, lowpass=1, show=False):
    highpass_image = image - ndimage.gaussian_filter(image, highpass)
    bandpass_image = ndimage.gaussian_filter(highpass_image, lowpass)
    filtered_image = np.copy(bandpass_image)
    peak_positions, cleared_image = find_peaks(bandpass_image, n_traces=1000, width=10, treshold_sd=treshold)
    peak_positions = [np.append([filefolder+fr"\{filename}", file_nr, -1], p) for p in peak_positions]
    if file_nr == 0:
        for i, _ in enumerate(peak_positions):
            peak_positions[i][2] = i
    if show:
        Ef.plot_find_peaks(peak_positions, image, filtered_image, cleared_image, plotting = False, show=False)
    pp_dataframe = pd.DataFrame(peak_positions, columns=('Filename','Filenr', 'tracenr', 'x (pix)', 'y (pix)', 'sigma (pix)', 'amplitude (a.u.)', 'error x (pix)', 'error y (pix)', 'error sigma (pix)', 'error amplitude (a.u.)' ))
    return pp_dataframe, filtered_image, cleared_image

def analyse_images(files, first_im, last_im, filefolder):
    empty_pp_df =[]
    for num, file in enumerate(files[first_im:last_im]):
        image = np.asarray(tiff.imread(file).astype(float)[200:800, 200:800])
        pp_df, filtered_image, cleared_image = analyse_image(image,num, filefolder,file, show=False)
        empty_pp_df.append(pp_df)
    all_pp_df =pd.concat(empty_pp_df, ignore_index=False)
    all_pp_df.to_csv('dataset_pp_v2.csv',index=False)
    return all_pp_df

def filter_image_plot(file, highpass =4, lowpass=1):
    image = np.asarray(tiff.imread(file).astype(float)[200:800, 200:800])
    highpass_image = image - ndimage.gaussian_filter(image, highpass)
    bandpass_image = ndimage.gaussian_filter(highpass_image, lowpass)
    filtered_image = np.copy(bandpass_image)
    return filtered_image

def analyse_dataset(df, files):
    df= pd.read_csv(df)
    image = filter_image_plot(files[0])
    range_peaks = []
    [range_peaks.append(i) for i in df.loc[:, 'Filenr'].to_numpy() if i not in range_peaks]
    link_df_old = link_peaks(df, image,len(range_peaks), show=False)
    link_df = link_df_old.copy()
    for i in range(4):
        link_df =link_traces(link_df, image, show=False)
        link_df.to_csv(fr'dataset_linktraces_loop{i}_v2.csv', index=False)
    trace_df = link_df
    trace_df.to_csv('dataset_final_loop_v2.csv', index=False)
    return link_df_old, trace_df

def analyse_trajectories(df):
    traj_df=pd.read_csv(df)
    range_traces = []
    [range_traces.append(i) for i in traj_df.loc[:, 'tracenr'].to_numpy() if i not in range_traces]
    for i in np.asarray(range_traces):
        tracenr_df=traj_df.loc[traj_df.loc[:,'tracenr'] == i]
        if len(tracenr_df)<2:
            range_traces=np.delete(range_traces,np.where(range_traces==i))
    diffusie_array = [np.append(p,[-2,-3,-4 ]) for p in [np.append([-1],p) for p in range_traces]]
    diffusie_df = pd.DataFrame(diffusie_array, columns=( 'Tau (s)', 'tracenr', 'MSDxy', 'MSDx','MSDy'))
    for i in range_traces:
        diffusie_df = diffusie(traj_df,diffusie_df, i)
    #diffusie_df.to_csv('dataset_diffusie.csv', index=False)
    return


treshold =7

os.chdir(fr"C:\Users\Santen\Documents\Images\data_030XX")
files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)
dataset_for_analysis= fr"C:\Users\Santen\Documents\Images\data_030XX\dataset_final_loop_v2.csv"
df=pd.read_csv(dataset_for_analysis)

#analysis images,dataset and trajectories
#analyse_images(files,0,29,fr"C:\Users\Santen\Documents\Images\data_030XX")
#link_df, trace_df =analyse_dataset(fr"C:\Users\Santen\Documents\Images\data_030XX\dataset_pp_v2.csv", files)
analyse_trajectories(dataset_for_analysis)

###validations functions
##histograms
#Ef.show_histogram_values(df,'amplitude (a.u.)', bins=np.linspace(20,150, 200))
#Ef.show_histogram_values(df,"sigma (pix)" ,bins= "auto")
#Ef.show_intensity_histogram_filename(files[2])
##plots
select_image_df = int(df.loc[df['tracenr'] == 2]['Filenr'].max())
#Ep.plot_full_tracjectory(2, files[select_image_df], df, df, select2=None, show=True)

#Ef.plot_trajectory(df, i, 30)


