import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
from scipy import ndimage
import os
import seaborn as sns
import TraceIO as tio
from lmfit import Model

p_xcoord = 3
p_ycoord = 4
vmin = -20
vmax = 50


def get_roi(image_array, center, width):
    start = (np.asarray(center) - width // 2).astype(int)
    roi = image_array[start[0]: start[0] + width, start[1]: start[1] + width]  # invert axis for numpy array of image
    return roi


def filter_image(file, highpass=4, lowpass=1):
    image = np.asarray(tiff.imread(file).astype(float))[200:800, 200:800]
    highpass_image = image - ndimage.gaussian_filter(image, highpass)
    bandpass_image = ndimage.gaussian_filter(highpass_image, lowpass)
    filtered_image = np.copy(bandpass_image)
    return filtered_image


def plot_fit_peaks(popt, err, X, Y, Z, fit):
    for p, e in enumerate(zip(popt, err)):
        print(f'par[{p}] = {e[0]} +/- {e[1]}')
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='afmhot', )
    ax.set_zlim(-20, 90)
    plt.title("raw peak", fontsize=16)
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, fit, cmap='afmhot')
    ax.set_zlim(-20, 90)
    plt.title("fit peak", fontsize=16)
    fig = plt.figure(2)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z - fit, cmap='afmhot')
    ax.set_zlim(-20, 90)
    plt.title("residu peak", fontsize=20)
    plt.show()


def plot_find_peaks(peak_positions, filtered_image, filenr, vmin, vmax, show):
    if show:
        plt.imshow(filtered_image, vmin=vmin, vmax=vmax, origin='lower')
        plt.gray()
        for trace_nr, pos in enumerate(peak_positions):
            plt.plot(float(pos[p_ycoord]), float(pos[p_xcoord]), marker='o', markerfacecolor='none', color='orange')
            plt.text(float(pos[p_ycoord]) + 6, float(pos[p_xcoord]), str(trace_nr), color='orange')
        tio.format_plot(r'x (pix)', r'y (pix)', title='find peaks', aspect=1.0,
                        save=fr'find peaks\filtered image with pp in {filenr}.png',
                        scale_page=0.5)
        plt.cla()


def plot_link_traces(image, pp_df):
    try:
        directory = fr"links"
        os.mkdir(directory)
    except FileExistsError:
        pass
    plt.imshow(image, origin="lower")
    plt.gray()
    for i in range(int(pp_df['tracenr'].max())):
        peak_values = pp_df.loc[pp_df['tracenr'] == i]
        trace = np.append([peak_values.loc[:, 'x (pix)']], [peak_values.loc[:, 'y (pix)']], axis=0)
        for axis in [trace[0], trace[1]]:
            if len(trace[0]) > 5:
                plt.plot(trace[1], trace[0], color='orange')
                plt.plot(trace[1][0], trace[0][0], marker="o", markerfacecolor="none", color="orange")
                plt.text(trace[1][0], trace[0][0], str(i), color="orange")
    tio.format_plot(xtitle = 'x (pix)', ytitle = r'y (pix)', title = 'Filtered image with traces', xrange = None, yrange = None, ylog = False, xlog = False, scale_page = 0.5, aspect = 1.0, save = None, boxed = True, GUI = False, ref = '', legend = None, fig = None, ax = None, txt = None) #r'x (pix)', r'y (pix)', title='Filtered image with traces', aspect=1.0, scale_page=0.5,save=fr'filtered image with traces.png'

    plt.cla()
    return


def plot_trajectory(files, df, tracenr, width=20, vmin=-10, vmax=vmax, show_other_peaks=False):
    trace = []
    selected_filenrs = df[df['tracenr'] == tracenr].loc[:, 'Filenr'].values
    selected_filenrs = np.arange(np.min(selected_filenrs), np.max(selected_filenrs) + 1)
    filenrs = np.arange(0, len(files))
    missingfilenr = set(filenrs) - set(selected_filenrs)
    for i, filenr in enumerate(selected_filenrs):
        try:
            row = df[(df['tracenr'] == tracenr) & (df['Filenr'] == filenr)].iloc[0]
        except IndexError:
            row = []
        filename = df[df.loc[:, 'Filenr'] == filenr]['Filename'].values[0]
        filtered_image = filter_image(filename)
        try:
            roi = get_roi(filtered_image, [x_center, y_center], width)
        except UnboundLocalError:
            x_center = float(row.loc['x (pix)'])
            y_center = float(row.loc['y (pix)'])
            roi = get_roi(filtered_image, [x_center, y_center], width)
        plt.ioff()
        plt.imshow(roi.T, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
        if show_other_peaks:
            # non selected peaks
            # new_df_non_selection = df_non_selection[(df_non_selection.loc[:, 'Filename'] == filename) &
            #             (df_non_selection['x (pix)'].between(x_center - width / 2,x_center + width / 2)) &
            #             (df_non_selection['y (pix)'].between(y_center - width / 2,  y_center + width / 2))]
            # plt.plot(new_df_non_selection.loc[:, 'x (pix)'] - x_center + width / 2,
            #          new_df_non_selection.loc[:, 'y (pix)'] - y_center + width / 2,
            #          linestyle='None', marker="o", markerfacecolor="none", color='red')
            # selected peaks
            new_df_selection = df[(df.loc[:, 'Filename'] == filename) &
                                  (df['x (pix)'].between(x_center - width / 2, x_center + width / 2)) &
                                  (df['y (pix)'].between(y_center - width / 2, y_center + width / 2))]
            plt.plot(new_df_selection.loc[:, 'x (pix)'] - x_center + width / 2,
                     new_df_selection.loc[:, 'y (pix)'] - y_center + width / 2, linestyle='None', marker="o",
                     markerfacecolor="none", color='blue')
            for i, new_df_trace in new_df_selection.iterrows():
                df_trace = df[df['tracenr'] == new_df_trace.loc['tracenr']]
                if len(df_trace) < 4:
                    plt.plot(new_df_trace.loc['x (pix)'] - x_center + width / 2,
                             new_df_trace.loc['y (pix)'] - y_center + width / 2, linestyle='None', marker="o",
                             markerfacecolor="none", color='red')
        if len(row):
            trace.append([row.loc['x (pix)'] - x_center + width / 2, row.loc['y (pix)'] - y_center + width / 2])
            plt.plot(row['x (pix)'] - x_center + width / 2, row['y (pix)'] - y_center + width / 2, marker="o",
                     markerfacecolor="none", color="lime")
        else:
            trace.append([np.NAN, np.NAN])
        plt.plot(np.asarray(trace).T[0], np.asarray(trace).T[1], color="lime")
        filenr_save = filenr + 1
        tio.format_plot(r'x (pix)', r'y (pix)', aspect=1.0, xrange=[0, width - 1], yrange=[0, width - 1],
                        save=fr"trajectory_{tracenr:g}\In image {filenr_save}_trace.jpg", scale_page=0.5)
        plt.cla()
        if i > 0:
            for n_filenr in missingfilenr:
                if len(df[df.loc[:, 'Filenr'] == n_filenr]['Filename'].values) > 0:
                    n_filename = df[df.loc[:, 'Filenr'] == n_filenr]['Filename'].values[0]
                    n_filtered_image = filter_image(n_filename)
                    try:
                        n_roi = get_roi(n_filtered_image, [x_center, y_center], width)
                    except UnboundLocalError:
                        x_center = float(row.loc['x (pix)'])
                        y_center = float(row.loc['y (pix)'])
                        n_roi = get_roi(n_filtered_image, [x_center, y_center], width)
                    plt.imshow(n_roi.T, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
                    # # non selected peaks
                    # new_df_non_selection = df_non_selection[(df_non_selection.loc[:, 'Filename'] == n_filename) &
                    #             (df_non_selection['x (pix)'].between(x_center - width / 2, x_center + width / 2)) &
                    #             (df_non_selection['y (pix)'].between(y_center - width / 2, y_center + width / 2))]
                    # plt.plot(new_df_non_selection.loc[:, 'x (pix)'] - x_center + width / 2, new_df_non_selection.loc[:, 'y (pix)'] - y_center + width / 2,
                    #          linestyle='None', marker="o", markerfacecolor="none", color='red')
                    # selected peaks
                    new_df_selection = df[(df.loc[:, 'Filename'] == n_filename) &
                                          (df['x (pix)'].between(x_center - width / 2, x_center + width / 2)) &
                                          (df['y (pix)'].between(y_center - width / 2, y_center + width / 2))]
                    plt.plot(new_df_selection.loc[:, 'x (pix)'] - x_center + width / 2,
                             new_df_selection.loc[:, 'y (pix)'] - y_center + width / 2,
                             linestyle='None', marker="o", markerfacecolor="none", color='blue')
                    for i, new_df_trace in new_df_selection.iterrows():
                        df_trace = df[df['tracenr'] == new_df_trace.loc['tracenr']]
                        if len(df_trace) < 4:
                            plt.plot(new_df_trace.loc['x (pix)'] - x_center + width / 2,
                                     new_df_trace.loc['y (pix)'] - y_center + width / 2, linestyle='None', marker="o",
                                     markerfacecolor="none", color='red')
                    n_filenr_save = n_filenr + 1
                    tio.format_plot(r'x (pix)', r'y (pix)', aspect=1.0, xrange=[0, width - 1], yrange=[0, width - 1],
                                    save=fr"trajectory_{tracenr:g}\In image {n_filenr_save}_trace.jpg", scale_page=0.5)
                    plt.cla()
    return


def show_intensity_histogram_filename(image1):
    fig = sns.histplot(data=image1, discrete=False)
    fig.set(xlabel=None)
    fig.set(ylabel=None)
    plt.semilogy()
    tio.format_plot(fr'Intensity', r'', aspect=0.6, xrange=[0, 300], yrange=[0, 10 ** 4], scale_page=0.5)
    print(f'sd = {np.std(image1)}')
    plt.show()
    plt.cla()
    return


def show_histogram_values(dataset, category, xrange, yrange, binwidth,select=None, pdf=False, histogram=False):
    if pdf:
        sum = dataset.loc[:, category]
        positions = sum.to_numpy()
        min_bound = 0
        max_bound = dataset.loc[:, category].max()
        dx = 0.01
        bins = np.arange(min_bound, max_bound, dx)
        x = bins[:-1] + (dx / 2)
        H, _ = np.histogram(positions,bins=bins)
        rho = H / (len(dataset.loc[:, category])*dx)
        plt.plot(x,rho, 'x')
        tio.format_plot(fr'{category}', fr'$\rho$', title=fr'{category}',xrange=xrange, aspect=1,
                        ylog=False, save=fr'histograms\pdf of {category}.png', scale_page=0.5)
    plt.cla()
    if histogram:
        dataset_value = dataset.loc[:, category]
        print(np.percentile(dataset_value, 80))
        print(f'sd = {np.std(dataset_value)}')
        print('amount:', len(dataset))
        fig = sns.histplot(dataset_value, discrete=False, binwidth=binwidth)
        fig.set(xlabel=None)
        fig.set(ylabel=None)
        tio.format_plot(fr'{category}', 'Frequency', title=fr'{category}', xrange=xrange, yrange=yrange, aspect=1, ylog=False, save=fr'histograms\histogram of {category}.png', scale_page=0.5)
    plt.cla()
    if select:
        dataset=dataset.loc[dataset['R2']>select]
        dataset_value = dataset.loc[:, category]
        print(np.percentile(dataset_value, 80))
        print(f'sd R2= {np.std(dataset_value)}')
        print('amount:', len(dataset))
        fig = sns.histplot(dataset_value, discrete=False, binwidth=binwidth)
        fig.set(xlabel=None)
        fig.set(ylabel=None)
        tio.format_plot(fr'{category}', 'Frequency', title=fr'{category}', xrange=xrange, yrange=yrange, aspect=1,    ylog=False, save=fr'histograms\histogram of {category}_R2.png', scale_page=0.5)
    plt.cla()


def histogram_length_traces(dataset, category, xrange, yrange):
    sorted_tracelength = dataset['tracenr'].value_counts().index.values
    all_length = []
    print(fr'Amount = {len(sorted_tracelength)}')
    for trace in sorted_tracelength:
        dataset_trace = dataset.loc[dataset['tracenr'] == trace]
        if len(dataset_trace) > 0:
            all_length.append(len(dataset_trace))
    print(f'sd = {np.std(all_length)}')
    fig = sns.histplot(all_length, discrete=False, binwidth=1)
    fig.set(xlabel=None)
    fig.set(ylabel=None)
    tio.format_plot(r'length', r'Frequency', ylog=False, xrange=xrange, yrange=yrange, aspect=1.0, scale_page=0.5,
                    save=f'tracelength\ link{category}_alt.png')
    plt.cla()


def msd_curve(x, sigma=0.05, diffusion=1.0, velocity=0.5):
    msd = 4 * (sigma ** 2) + 4 * diffusion * x + (velocity ** 2) * (x ** 2)
    return msd


def fit_msd(tau, msd, weight):
    model = Model(msd_curve)
    params = model.make_params()
    params.add('sigma', value=0.005, max=2, min=0)
    params.add('diffusion', value=1, max=100, min=0)
    params.add('velocity', value=0.5, max=100, min=0)
    result = model.fit(msd, params, x=tau, weights=weight)
    # result.params.pretty_print()
    p_fitted = [result.params[p].value for p in result.params]
    err_p = [result.params[p].stderr for p in result.params]
    p_fitted = np.append(p_fitted, err_p)
    R2 = 1 - result.residual.var() / np.var(msd)
    return model.eval(result.params, x=tau), p_fitted, R2


if __name__ == "__main__":
    tau = np.linspace(0.01, 10, 50)
    msd = msd_curve(tau, 0.01, 0, 0.6)
    msd += np.random.random(len(tau)) * 5
    plt.plot(tau, msd, 'o')
    fit = fit_msd(tau, msd, msd_error=np.ones_like(tau))
    plt.plot(tau, fit)
    tio.format_plot('tau (s)', 'msd (um^2)')


def show_scatterplot(dataset, category1, category2, cat1_error=None, cat2_error=None, error=False, non_error=False):
    if non_error:
        x = dataset.loc[:, category1]
        y = dataset.loc[:, category2]
        # plt.scatter(x,y )
        plt.hist2d(x, y, bins=50, cmap='Greys', range=[[0, 1], [0, 1500]])
        rect = patches.Rectangle((0.75, 0), 0.25, 500, linewidth=1, edgecolor='r', facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)
        tio.format_plot(fr'{category1}', fr'{category2}', aspect=1)
    if error:
        x = dataset.loc[:, category1]
        y = dataset.loc[:, category2]
        xerr = dataset.loc[:, cat1_error]
        yerr = dataset.loc[:, cat2_error]
        selection = (xerr > 0) & (xerr < 0.2) & (yerr > 0) & (yerr < 0.2)
        plt.scatter(x, y, alpha=0.1, color='red')
        plt.errorbar(x[selection], y[selection], xerr=xerr[selection], yerr=yerr[selection], fmt='o',
                     markerfacecolor="none")
        tio.format_plot(fr'{category1}', fr'{category2}', aspect=1)


def scale_array(array, min, max):
    array = np.asarray(array)
    array -= min
    array /= (max - min)
    array[array > 1] = 1
    array[array < 0] = 0
    return array


def plot_peaks_colors(df, df_diff, files, category, min, max, normal=True):
    import matplotlib.cm as cm
    df_diff = df_diff[df_diff['R2'] > 0.5]
    try:
        directory = fr"peak_colors"
        os.mkdir(directory)
    except FileExistsError:
        pass
    for num, file in enumerate(files[1:31]):
        df_file = df.loc[df['Filenr'] == num + 1]
        image_original = np.asarray(tiff.imread(files[num]).astype(float))[200:800, 200:800]
        image_original -= np.median(image_original)
        plt.imshow(image_original, vmin=-20, vmax=75, origin="lower", cmap='gray')
        tracenrs = df_file.loc[:, 'tracenr'].copy()
        tracenrs2 = df_diff.loc[:, 'tracenr'].copy()
        val = [df_diff[df_diff['tracenr'] == trace][category].values if len(
            df_diff[df_diff['tracenr'] == trace][category]) > 0 else [0] for trace in tracenrs]
        xcoords = [df_file[df_file['tracenr'] == trace]['x (pix)'].values for trace in
                   list(set(tracenrs) - set(tracenrs2)) if len(df_file[df_file['tracenr'] == trace]) > 0]
        ycoords = [df_file[df_file['tracenr'] == trace]['y (pix)'].values for trace in
                   list(set(tracenrs) - set(tracenrs2)) if len(df_file[df_file['tracenr'] == trace]) > 0]
        val = scale_array(val, min, max)
        if normal:
            val = val
        else:
            val = 1 - val
        colors = [cm.seismic(color) for color in val]
        plt.scatter(df_file.loc[:, 'y (pix)'], df_file.loc[:, 'x (pix)'], s=40, facecolors='none', edgecolors=colors)
        plt.scatter(ycoords, xcoords, color='black', marker='o', facecolor='none')
        tio.format_plot(r'x (pix)', r'y (pix)', aspect=1.0, save=fr"peak_colors\{category}\Image {num}.jpg",
                        scale_page=0.5)
        plt.cla()
    return
