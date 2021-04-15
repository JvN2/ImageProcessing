import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy import ndimage
import glob
import os
from natsort import natsorted, ns
from scipy.optimize import curve_fit
import copy
from TraceIO import format_plot

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

    Z = np.asarray(Z)
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
        print(f'fit result:')
        plot_fit_peaks(popt, err,X, Y, Z, fit)
    popt = np.append(popt, err)
    return fit, popt, err

def find_peaks(image_array, width=20, treshold_sd=5, n_traces=20):
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
        plot_link_peaks(image, pp_df)
    #pp_df.to_csv('dataset_linkpeaks.csv', index=False)
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
        distance_2 = np.sum((nstart_df.loc[:, ['x (pix)', 'y (pix)']] - trace_values.loc[['x (pix)', 'y (pix)']]) ** 2, axis=1)
        if np.min(distance_2) < max_dist ** 2:
            old_tracenr = nstart_df.iloc[np.argmin(distance_2)]['tracenr']
            replace_tracenr = pp_df.index[pp_df['tracenr'] == old_tracenr]
            if len(replace_tracenr) > 0:
                pp_df.loc[replace_tracenr, 'tracenr'] = trace
    if show:
        plot_link_peaks(image,pp_df)
    return pp_df

def select_tracenr(select, image_df, df, show=True):
    select_image_df = int(df.loc[df['tracenr']==select]['Filenr'].max())
    if show:
        plot_trace_peaks(select, image_df[select_image_df], df, show =True)
def MSD(xdata, ydata):
    rad= np.sqrt(xdata**2+ydata**2)
    diffusie = np.diff(rad)
    diffusie_sqrt = diffusie**2
    MSD = np.mean(diffusie_sqrt)
    return MSD

##FUNCTIONS ANALYSES FOR IMAGES AND DATASET
def analyse_image(image, file_nr, highpass=4, lowpass=1, show=False):
    highpass_image = image - ndimage.gaussian_filter(image, highpass)
    bandpass_image = ndimage.gaussian_filter(highpass_image, lowpass)
    filtered_image = np.copy(bandpass_image)
    peak_positions, cleared_image = find_peaks(bandpass_image, n_traces=1000, width=10, treshold_sd=treshold)
    peak_positions = [np.append([file_nr, -1], p) for p in peak_positions]
    if file_nr == 0:
        for i, _ in enumerate(peak_positions):
            peak_positions[i][1] = i
    if show:
        plot_find_peaks(peak_positions,image, filtered_image, cleared_image, plotting = False, show=False)
    pp_dataframe = pd.DataFrame(peak_positions, columns=('Filenr', 'tracenr', 'x (pix)', 'y (pix)', 'sigma (pix)', 'amplitude (a.u.)', 'error x (pix)', 'error y (pix)', 'error sigma (pix)', 'error amplitude (a.u.)' ))
    return pp_dataframe, filtered_image, cleared_image

def analyse_images(files, first_im, last_im):
    empty_pp_df =[]
    for num, file in enumerate(files[first_im:last_im]):
        image = np.asarray(tiff.imread(file).astype(float)[200:800, 200:800])
        pp_df, filtered_image, cleared_image = analyse_image(image, file_nr=num, show=False)
        empty_pp_df.append(pp_df)
    all_pp_df =pd.concat(empty_pp_df, ignore_index=False)
    show_intensity_histogram_image(image, 'original image', show=True)
    show_intensity_histogram_image(filtered_image, 'filtered image', show=True)
    show_intensity_histogram_image(cleared_image, 'cleared image', show=True)
    #all_pp_df.to_csv('dataset_pp.csv',index=False)
    return all_pp_df

def analyse_dataset(df, files):
    df= pd.read_csv(df)
    image = filter_image_plot(files[0])
    range_peaks = []
    [range_peaks.append(i) for i in df.loc[:, 'Filenr'].to_numpy() if i not in range_peaks]
    link_df = link_peaks(df, image,len(range_peaks), show=False)
    trace_df = link_traces(link_df, image, show=False)
    #show_histogram_values(trace_df,'amplitude (a.u.)', bins=np.linspace(20,150, 200),  show=True)
    #show_histogram_values(trace_df,"sigma (pix)" ,bins= "auto", show=False)
    #show_intensity_histogram_filename(files[0], show =True)
    #select_tracenr(6, files, trace_df, show=True)
    #plot_link_peaks_withtrace(image, link_df, trace_df, show=False)
    #trace_df.to_csv('dataset_final.csv', index=False)
    return

##VALIDATION FUNCTIONS
#Plot functions
def plot_fit_peaks(popt, err,X, Y, Z, fit):
    for p, e in enumerate(zip(popt, err)):
        print(f'par[{p}] = {e[0]} +/- {e[1]}')
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='afmhot')
    plt.title("raw peak")
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, fit, cmap='afmhot')
    plt.title("fit peak")
    fig = plt.figure(2)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z - fit, cmap='afmhot')
    plt.title("residu peak")
    ax.set_zlim(-100, np.max(fit))
    plt.show()


def plot_find_peaks(peak_positions,image, filtered_image, cleared_image, plotting =False,show = False):
    if plotting:
        titles = ["Original", 'Filtered', 'Cleared']
        image -= np.median(image)
        for i, plot_image in enumerate([image, filtered_image, cleared_image]):
            plt.figure(i)
            plt.imshow(plot_image, vmin=-20, vmax=50)
            plt.colorbar()
            plt.gray()
            plt.title(titles[i])
        plt.show()
    if show:
        plt.imshow(image, vmin=-20, vmax=50, origin='lower')
        plt.colorbar()
        plt.gray()
        for trace_nr, pos in enumerate(peak_positions):
            plt.plot(pos[p_ycoord], pos[p_xcoord], marker='o', markerfacecolor='none', color='red')
            plt.text(pos[p_ycoord] + 6, pos[p_xcoord], str(trace_nr), color='blue')
        plt.show()

def plot_link_peaks(image, pp_df):
    plt.imshow(image, origin="lower")
    plt.gray()
    for i in range(int(pp_df['tracenr'].max())):
        peak_values = pp_df.loc[pp_df['tracenr'] == i]
        trace = np.append([peak_values.loc[:,'x (pix)']],[peak_values.loc[:,'y (pix)']], axis =0 )
        for axis in [trace[0], trace[1]]:
            if len(trace[0]) >7:
                plt.plot(trace[1], trace[0], color='orange')
                plt.plot(trace[1][0], trace[0][0], marker="o", markerfacecolor="none", color="orange")
                plt.text(trace[1][0], trace[0][0], str(i), color="orange")
    plt.title('Filtered image with traces')
    plt.colorbar()
    plt.show()
    return

def plot_link_peaks_withtrace(image, df1,df2, show=False):
    if show:

        plt.imshow(image, origin="lower")
        plt.gray()
        for i in range(int(df2['tracenr'].max())):
            peak_values = df2.loc[df2['tracenr'] == i]
            trace = np.append([peak_values.loc[:,'x (pix)']],[peak_values.loc[:,'y (pix)']], axis =0 )
            for axis in [trace[0], trace[1]]:
                if len(trace[0]) > 7:
                    plt.plot(trace[1], trace[0], color='orange')
                    plt.plot(trace[1][0], trace[0][0], marker="o", markerfacecolor="none", color="orange")
                    plt.text(trace[1][0], trace[0][0], str(i), color="orange")
        for i in range(int(df1['tracenr'].max())):
            peak_values = df1.loc[df1['tracenr'] == i]
            trace = np.append([peak_values.loc[:,'x (pix)']],[peak_values.loc[:,'y (pix)']], axis =0 )
            for axis in [trace[0], trace[1]]:
                if len(trace[0]) > 7:
                    plt.plot(trace[1], trace[0], color='green')
                    plt.plot(trace[1][0], trace[0][0], marker="o", markerfacecolor="none", color="green")
                    plt.text(trace[1][0], trace[0][0], str(i), color="green")
        plt.title("Filtered image with traces")
        plt.colorbar()
        plt.show()

## DATASET analyse
def filter_image_plot(file, highpass =4, lowpass=1):
    image = np.asarray(tiff.imread(file).astype(float)[200:800, 200:800])
    highpass_image = image - ndimage.gaussian_filter(image, highpass)
    bandpass_image = ndimage.gaussian_filter(highpass_image, lowpass)
    filtered_image = np.copy(bandpass_image)
    return filtered_image

def plot_trace_peaks(select,image, dataset, show =False, width=30):
    filtered_image = filter_image_plot(image)
    xcoords = dataset.loc[dataset["tracenr"]==select,'x (pix)' ]
    ycoords = dataset.loc[dataset["tracenr"]==select,'y (pix)' ]
    roi = get_roi(filtered_image, [xcoords.iloc[0], ycoords.iloc[0]], width)
    xcoords =xcoords-xcoords.iloc[0]+ (width/2)
    ycoords =ycoords - ycoords.iloc[0]+ (width/2)
    if show:
        plt.imshow(roi, origin='lower')
        plt.colorbar()
        plt.gray()
        plt.plot(ycoords.iloc[0], xcoords.iloc[0], marker="o", markerfacecolor="none",  color="orange")
        plt.plot( ycoords,xcoords, color="orange")
        plt.show()
    return

# histograms
def show_intensity_histogram_filename(filename, show =False):
    image = np.asarray(tiff.imread(filename).astype(float)[200:800, 200:800])
    filtered_image = filter_image_plot(filename)
    #his_oi, bins_oi = np.histogram(image, bins=np.linspace(0, 100, 200))
    his_fi, bins_fi = np.histogram(filtered_image, bins=np.linspace(-50,100, 200))
    print(f'sd = {np.std(filtered_image)}')
    if show:
        #plt.bar(bins_oi[1:], his_oi, label = 'original')
        plt.bar(bins_fi[1:], his_fi, label = 'filtered')
        plt.semilogy()
        #format_plot(xtitle='amplitude (a.u.)', ytitle='frequency', title=fr"Frequency {category} fitted peaks", scale_page=1, aspect=0.5, boxed=False)
        plt.title("Intensity")
        plt.show()
def show_intensity_histogram_image(image, image_name, show =False):
    his, bins= np.histogram(image, bins='auto')
    print(f'sd = {np.std(image)}')
    if show:
        plt.bar(bins[1:], his)
        plt.semilogy()
        #format_plot(xtitle='amplitude (a.u.)', ytitle='frequency', title=fr"Frequency {category} fitted peaks", scale_page=1, aspect=0.5, boxed=False)
        plt.title(fr"Intensity {image_name} ")
        plt.show()

def show_histogram_values(dataset, category,bins,show=False):
    dataset_value = dataset.loc[:,category]
    # format_plot(xtitle='x (a.u.)', ytitle='y (a.u.)', title='', xrange=None, yrange=None,
    #             ylog=False, xlog=False, scale_page=1.0, aspect=0.5, save=None, boxed=True,
    #             GUI=False, ref='', legend=None, fig=None, ax=None, txt=None):
    if show:
        print(f'sd = {np.std(dataset_value)}')
        his, bins = np.histogram(dataset_value, bins=bins)
        plt.bar(bins[1:], his)
        format_plot(xtitle='amplitude (a.u.)', ytitle='frequency', title= fr"Frequency {category} fitted peaks", scale_page=1, aspect=0.5, boxed=False)
        plt.semilogy()


p_xcoord = 2
p_ycoord = 3
p_sd = 4
p_A = 5
e_xcoord =6
e_ycoord=7
e_sd=8
e_A=9
treshold =7

os.chdir("G:\Santen\Data images\data_030XX")
files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)

#call analysis function
#analyse_images(files,0,10, "G:\Santen\Data images\data_030XX")
#analyse_dataset("G:\Santen\Data images\data_030XX\dataset_pp.csv", files)
analyse_images(files,0,10)
#analyse_dataset(fr"C:\Users\santen\PycharmProjects\test\try_pp.csv", files)
