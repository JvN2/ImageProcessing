import Tracking.TrackTraces as tt

# 3) VARIABLES
pix_size = 0.112
treshold = 5
vmin = -20
vmax = 50
foldername = fr"C:\Users\Santen\Documents\Images\data_030XX"
os.chdir(foldername)
files = natsorted(glob.glob("*.tiff"), alg=ns.IGNORECASE)

tt.msd_trajectory(df, 3 , pix_size)
df = tt.analyse_msd(df)