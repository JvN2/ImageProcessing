pimport Tracking.TrackTraces as tt


if __name__ == "__main__":
    # 3) VARIABLES FOR
    pix_size = 0.112
    treshold = 5
    vmin = -10
    vmax = 100
    first_im = 10
    last_im = 12
    image_size_min = 75
    image_size_max = 500
    highpass=4
    lowpass=1

#Peak fitting
    n_traces = 200

    foldername = fr"/Volumes/Drive Sven/2FOTON/210325 - 25-03-21  - Transgenic/data_052"
    os.chdir(foldername)
    files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)

    # 5) CALLING ANALYSIS FUNCTIONS
    #averag_images(files,first_im,last_im,foldername)
    analyse_images(files, first_im, last_im, foldername)
    dataset_pp = foldername + "/dataset_pp.csv"
    df_pp = pd.read_csv(dataset_pp)

    # peak_selection(df_pp,files,10,0,500, show1=True)
    dataset_pp_selection = foldername + "\dataset_pp_selection.csv"
    # df_pp=pd.read_csv(dataset_pp_selection)

    # analyse_dataset(df_pp, files)

    dataset_traces = foldername + "\dataset_final_loop.csv"
    df_traces = pd.read_csv(dataset_traces)
    # analyse_trajectories(df_traces)

    dataset_link = foldername + "\dataset_linkpeaks.csv"
    df_link = pd.read_csv(dataset_link)


    def analyse_pp(df_pp, df_link, length, sort):
        filenrs = np.sort(df_pp['Filenr'].value_counts().index.values)
        total_linked, total_not_linked, total_traces = 0, 0, 0
        for filenr in filenrs:
            selected_traces = df_link[df_link['Filenr'] == filenr].loc[:, 'tracenr'].values
            total_traces += len(selected_traces)
            linked_trace, not_linked_trace = 0, 0
            for peak in selected_traces:
                selected_tracenr = df_link[df_link['tracenr'] == peak]
                if len(selected_tracenr) > length:
                    linked_trace += 1
                else:
                    not_linked_trace += 1
            total_linked += linked_trace
            total_not_linked += not_linked_trace
        print(sort)
        print(fr"Average of traces in a filenr: {total_traces / len(filenrs)}", '\n',
              fr"Average of linked traces in a filenr: {total_linked / len(filenrs)} (fraction ={(total_linked / len(filenrs)) / (total_traces / len(filenrs))}%) ",
              '\n',
              fr"Average of non linked traces in a filenr: {total_not_linked / len(filenrs)} (fraction ={(total_not_linked / len(filenrs)) / (total_traces / len(filenrs))}%) ")
        return


    # analyse_pp(df_pp,df_link,1,'After linking peaks')
    # analyse_pp(df_pp,df_traces,3,'After linking traces')

    dataset_msd = foldername + "\dataset_msd.csv"
    df_msd = pd.read_csv(dataset_msd)
    # analyse_msd(df_msd)

    dataset_diffusie = foldername + "\dataset_diffusie.csv"
    df_diffusie = pd.read_csv(dataset_diffusie)

    # Ef.plot_peaks_colors(df_traces, df_diffusie, files, r'D (um^2 s^-1)', min=0, max=0.1, normal=False)
    # Ef.plot_peaks_colors(df_traces, df_diffusie, files, 'sigma (um)', min=0, max=0.25, normal=True)
    # Ef.plot_peaks_colors(df_traces, df_diffusie, files, r'v (um s^-1)', min=0, max=0.5, normal=False)


    # 6) FUNCTIONS FOR VALIDATIONS
    # 6.1) Histograms
    # 6.1.1) Dataset peak positions and traces;circulair fitten
    # Ef.show_histogram_values(df_pp,'amplitude (a.u.)', bins=np.linspace(-20,80, 200))
    # Ef.show_histogram_values(df_traces,"sigma (pix)" ,bins= "auto")

    # 6.1.2) Dataset peak positons;
    # Ef.show_histogram_values(df_pp,'aspect_ratio' )
    # Ef.show_histogram_values(df_pp2,'R2')
    # Ef.show_histogram_values(df_pp2,'error x (pix)' ,bins= "auto")
    # Ef.show_histogram_values(df_pp2,'error y (pix)' ,bins= "auto")
    # Ef.show_histogram_values(df_pp2,'intensity (a.u.)')

    # 6.1.3.) Dataset diffusie;

    # Ef.histogram_length_traces(df_link,  'peaks', xrange=[0,100], yrange=[0,400])
    # Ef.histogram_length_traces(df_traces, 'traces',xrange=[0,100], yrange=[0,400] )

    # Ef.show_histogram_values(df_diffusie,'sigma (um)', xrange=[0,1],yrange=[0,150], binwidth=0.1,select=0.5)
    # Ef.show_histogram_values(df_diffusie,  r'D (um^2 s^-1)', xrange=[0,1],yrange=[0,150], binwidth=0.1, select=0.5)
    # Ef.show_histogram_values(df_diffusie, r'v (um s^-1)', xrange=[0,1], yrange=[0,150],binwidth=0.1,select=0.5)
    # Ef.show_histogram_values(df_diffusie, r'R2', xrange=[0,1.1], yrange=[0,100], binwidth=0.1,select=0.5)

    # 6.2) Plots


    # 6.3) Scatter plots
    # 6.3.1) Aspect ratio and intensity
    # Ef.show_scatterplot(df_pp,'R2', 'intensity (a.u.)')
    # Ef.show_scatterplot(df_diffusie,r'diffusion (um^2 s^-2)','sigma (um)','diffusion_error (um^2 s^-2)','sigma_error (um)', error=True)
    # Ef.show_scatterplot(df_diffusie,r'diffusion (um^2 s^-2)','velocity (um s^-1)','diffusion_error (um^2 s^-2)','velocity_error (um s^-1)', error=True)
    # Ef.show_scatterplot(df_diffusie,r'sigma (um)', 'velocity (um s^-1)','sigma_error (um)','velocity_error (um s^-1)', error=True)

    # 6.3) Follow trajectories
    def video_select(filename):
        os.chdir(foldername + filename)
        files = natsorted(glob.glob("*.jpg"), alg=ns.IGNORECASE)
        img_array = []
        for filename in files:
            img = opencv.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        out = opencv.VideoWriter(fr'video_selection.avi', opencv.VideoWriter_fourcc(*'DIVX'), 1, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


    # video_select(fr"\selection_peaks")
    def video_traces(files, traces, df):
        for trace in tqdm(traces, desc='plot trajectory'):
            print(trace)
            Ef.plot_trajectory(files, df, trace, vmax=vmax, width=50, show_other_peaks=True)
        for trace in tqdm(traces, desc='video trajectory'):
            os.chdir(foldername + fr"\trajectory_{trace:g}")
            files = natsorted(glob.glob("*.jpg"), alg=ns.IGNORECASE)
            img_array = []
            for filename in files:
                img = opencv.imread(filename)
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)
            out = opencv.VideoWriter(fr'video_trajactory.avi', opencv.VideoWriter_fourcc(*'DIVX'), 2, size)
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

    # video_traces(files, df_diffusie['tracenr'].value_counts().index.values, df_traces)
