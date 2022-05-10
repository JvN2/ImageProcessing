from datetime import timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import ProcessImages.CoSMoS as cms
import ProcessImages.ImageIO2 as imio


class Movie():
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.out.release()

    def __clear__(self):
        self.out.release()
        print('Done!')

    def __call__(self, filename, fps=1):
        self.filename = filename
        ext = filename.split('.')[-1].lower()
        codec = {'avi': 'DIVX', 'mp4': 'mp4v'}
        self.codec = codec[ext]
        self.out = None
        self.rgb_image = None
        self.fps = fps
        self.range = [None] * 4
        self.frame_nr = -1
        self.circle_radius = None
        self.circle_coords = None
        return self

    def add_frame(self, red=None, green=None, blue=None, grey=None, label = None):
        self.frame_nr += 1
        if self.rgb_image is not None:
            self.rgb_image *= 0
        for i, image in enumerate([red, green, blue, grey]):
            if image is not None:
                if self.out is None:
                    size = np.asarray(image).shape
                    self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*self.codec), self.fps, size)
                    self.rgb_image = np.zeros((size[0], size[1], 3), 'int32')
                if self.range[i] is None:
                    self.range[i] = [np.percentile(image, 10), np.percentile(image, 90)]

                if i == 3:
                    for j in [0, 1, 2]:
                        self.rgb_image[..., j] += imio.scale_u8(image, self.range[i]).astype(np.int32)
                else:
                    self.rgb_image[..., i] += imio.scale_u8(image, self.range[i]).astype(np.int32)
        im = Image.fromarray(np.clip(self.rgb_image, 0, 255).astype(np.uint8), mode='RGB')

        if label is not None:
            font_family = "arial.ttf"
            font = ImageFont.truetype(font_family, 25)
            draw = ImageDraw.Draw(im)
            draw.text([10, 10], label, color='white', font=font)
            del draw

        # im.save(self.filename.replace('.mp4', f'_{self.frame_nr}.jpg'))
        self.out.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        return

    def set_range(self, red=None, green=None, blue=None, grey=None):
        for i, intensity_range in enumerate([red, green, blue, grey]):
            if intensity_range is not None:
                self.range[i] = intensity_range
        return

    def set_circles(self, coords, radius=10):
        self.circle_radius = radius
        self.circle_coords = coords


def read_tiff(filename):
    return np.asarray(plt.imread(Path(filename)).astype(float))


def read_tiffs_from_df(filename, df):
    tiff_names = [str(Path(filename).with_name(f'image{image_nr:g}.tiff')) for image_nr in df['Filenr'].values]

    time = list(set(df['Frame']))
    transmission_image = df.iloc[0]['PiezoZ (um)'] > df.iloc[1]['PiezoZ (um)']
    color = [0, 1] if transmission_image else [0]
    z_position = list(set(df['PiezoZ (um)']))[1:] if transmission_image else set(df['PiezoZ (um)'])
    image = read_tiff(tiff_names[0])

    ims = np.zeros([len(time), 3, len(z_position), len(image[0, :]), len(image[:, 0])], dtype='u8')
    for _, row in tqdm(df.iterrows(), 'Opening files'):
        im = read_tiff(Path(filename).with_name(f'image{row["Filenr"]:g}.tiff'))
        ims[int(time.index(row['Frame'])), int(row['Slice'] != 0),
        int(row['Slice']) - 1 if int(row['Slice'] != 0) else 0, :, :] = cms.scale_image_u8(im, z_range=[0, 255])

    return ims


if __name__ == "__main__":
    filename = r'\\data02\pi-vannoort\Noort\Data\Alex GR\data_004\data_004.dat'

    df = pd.read_csv(filename, sep='\t')
    tiff_names = [str(Path(filename).with_name(f'image{image_nr:g}.tiff')) for image_nr in df['Filenr'].values]
    df.insert(loc=0, column='Filename', value=tiff_names)
    df['Time (s)'] -= df['Time (s)'].min()

    print(f'Opened file: {filename}')
    print(f'Duration of experiment = {timedelta(seconds=int(df["Time (s)"].max()))}.')

    movie = Movie()
    with movie(filename.replace('.dat', '.mp4'), 4):
        movie.set_range(grey=[-10, 70])
        frames = df['Frame'].unique()
        for i, frame in enumerate(tqdm(frames, 'Processing frames')):
            stack = [read_tiff(tiff_file) for tiff_file in df[(df['Frame'] == frame) & (df['Slice'] != 0)]['Filename']]
            projection = np.max(np.asarray(stack), axis=0)
            projection = cms.filter_image(projection, 0, 10)
            label = f'T = {timedelta(seconds = int(df[(df["Frame"] == frame) & (df["Slice"] != 0)]["Time (s)"].min()))}'
            movie.add_frame(grey=projection, label=label)
