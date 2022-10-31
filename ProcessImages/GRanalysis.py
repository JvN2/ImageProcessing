from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import ProcessImages.ImageIO as im3


def make_movie(df):
    movie = im3.Movie(filename.replace('.dat', '.mp4'), 6)
    movie.set_range(grey=[-10, 1000])
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        im = im3.read_tiff(row['Filename'])
        im -= np.percentile(im, 30)
        label = f'Wavelength = {df.at[i, "Wavelength (nm)"]:4.0f} (nm)'
        movie.add_frame(grey=im, label=label, label_size=40)


def create_mask(df, wavelength=[890, 930], percentile=99, normailize=False, show=False):
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if wavelength[0] < df.at[i, "Wavelength (nm)"] < wavelength[1]:
            im = im3.read_tiff(row['Filename'])
            try:
                sum += im
            except NameError:
                sum = im
    sum = sum - np.percentile(np.reshape(sum, np.prod(np.shape(sum))), 50)
    treshold = np.percentile(np.reshape(sum, np.prod(np.shape(sum))), percentile) / 2
    mask = np.asarray(sum > treshold).astype(float)
    if show:
        plt.imshow(mask, cmap='Greys_r')
        plt.colorbar()
        plt.show()
    if normailize:
        return np.asarray(mask / np.sum(mask)).astype(int)
    else:
        return mask


def calculate_spectrum(df, mask, name='Fluorescence (a.u)', show=False):

    mask = np.asarray(mask).astype(int)
    intensity = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        im = im3.read_tiff(row['Filename'])
        im -= np.percentile(im, 30)
        intensity.append(np.mean(im[mask>0.5]))

    df[name] = intensity

    excitation = df['Photodiode (mW)']
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    excitation = np.convolve(excitation, kernel, mode='same')
    excitation /= np.mean(excitation)
    df['Excitation (a.u.)'] = excitation

    df['Corrected ' + name] = intensity/np.square(np.asarray(excitation).astype(float))
    df['Corrected ' + name] = df['Corrected ' + name] / np.sum( df['Corrected ' + name][ df['Corrected ' + name] <800])

    if show:
        plot_label = 'Corrected ' + name
        # plot_label = name
        fig, ax = plt.subplots()
        ax.plot(df['Wavelength (nm)'], df[plot_label], color='blue')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(plot_label, color='blue')

        ax2 = ax.twinx()
        ax2.plot(df["Wavelength (nm)"], excitation**2, label='Excitation', color='red')
        ax2.set_ylabel('Excitation^2 (a.u.)', color = 'red')
        plt.show()
    return df


if __name__ == "__main__":
    filename = r'\\data02\pi-vannoort\Noort\Data\Alex GR\data_004\data_004.dat'
    filename = r'D:\Internship 2022\TPMM data\221017 (NFP5 part2)\data_021\data_021a.dat'

    # read and process imaging parameters
    df = pd.read_csv(filename, sep='\t')

    tiff_names = [str(Path(filename).with_name(f'image{image_nr:g}.tiff')) for image_nr in df['Filenr'].values]
    df.insert(loc=0, column='Filename', value=tiff_names)
    # df['Time (s)'] = np.asarray(df['Time (s)']).astype(float) - df['Time (s)'].min()
    print(f'Opened file: {filename}')

    df = df[df['Photodiode (mW)'] > 0.05]
    df.sort_values(by='Wavelength (nm)', inplace=True)

    # make_movie(df)
    nucleus_mask = create_mask(df, show=False)
    df = calculate_spectrum(df, nucleus_mask, show=True, name='nucleus')

    cell_mask = create_mask(df, [730, 830], percentile=80)
    cell_mask -= nucleus_mask

    df = calculate_spectrum(df, cell_mask, show=True, name='cytoplasm')
    df.to_csv(filename.replace('.dat', '.csv'))
