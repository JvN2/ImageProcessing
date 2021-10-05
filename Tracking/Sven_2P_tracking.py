import TrackTraces as tt
import pandas as pd
import glob
from natsort import natsorted, ns
import os.path





if __name__ == "__main__":

    # DETECT PEAKS
    threshold = 5
    vmin = 10
    vmax = 20
    first_im = 10
    last_im = 20
    image_size_min = 75
    image_size_max = 500
    highpass = 4
    lowpass = 1
    threshold_sd = 5
    spot_width = 5
    n_traces = 20

    # PEAK SELECTION
    selection_ar = 10
    selection_R2 = 0
    selection_int = 500

    # PEAK LINKING
    link_dist = 10
    gap_images = 1
    pix_size = 0.112
    timelag = 0.2

    foldername = fr"F:\2FOTON\210325 - 25-03-21  - Transgenic\data_052"
    df_filename = rf'{foldername}\dataset_pp.csv'

    # DETECT PEAKS
    if os.path.isfile(df_filename):
        df = pd.read_csv(df_filename)
        print(rf'Reading dataframe: {df_filename}')
        os.chdir(foldername)
        files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)
    else:
        os.chdir(foldername)
        files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)
        tt.average_images(files, first_im, last_im)
        df = tt.find_peaks(files, first_im, last_im, foldername, highpass, lowpass, vmin, vmax, threshold_sd, spot_width, n_traces, show=False)
        df.to_csv(df_filename)
        print(rf'Dataframe stored in {df_filename}')


    # PEAK SELECTION AND PLOTTING
    df = tt.select_peaks(df, selection_ar, selection_R2, selection_int, df_filename)
    tt.plot_peaks(df, files, first_im, last_im, image_size_min, image_size_max, vmin, vmax)


    #PEAK LINKING
    df = tt.find_traces(df, files, image_size_min, image_size_max, highpass, lowpass, link_dist, gap_images, df_filename, show=False)
    tt.analyse_traces_stats(df)


    #MEAN SQUARED DISPLACEMENT CALCULATION
    df_filename_msd = rf'{foldername}\dataset_msd.csv'
    df_diffusie = rf'{foldername}\dataset_diffusie.csv'

    if os.path.isfile(df_filename_msd):
        os.chdir(foldername)
        files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)
        df = pd.read_csv(df_filename_msd)
        df_diffusie = pd.read_csv(df_diffusie)
        print(rf'Reading dataframe: {df_filename_msd}')
        print(rf'Reading dataframe: {df_diffusie}')
    else:
        os.chdir(foldername)
        files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)
        df_msd = tt.analyse_trajectories(df, pix_size, timelag)
        df_msd = tt.analyse_msd(df_msd)
        print(rf'Dataframe stored in {df_filename_msd}')
