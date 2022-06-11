from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob, tifffile, re, os

I16 = 2 ** 16 - 1


def natural_sort_key(s, _re=re.compile(r'(\d+)')):
    return [int(t) if i & 1 else t.lower() for i, t in enumerate(_re.split(s))]


def PIL2array(img, range=[0, I16]):
    """ Convert a PIL/Pillow image to a numpy array """
    frame = np.array(img.getdata(), np.uint16)
    frame = 255 * (np.clip(frame, range[0], range[1]) - range[0]) / (range[1] - range[0])
    return frame.reshape(img.im_shape[1], img.im_shape[0]).astype(np.uint8)


def create_multi_tiff(foldername, filename_out="multiImage.tiff", range=[0, I16], selection=None, stack_size=2):
    """ Convert a PIL/Pillow image to a numpy array """
    if selection is None:
        selection = np.arange(0, stack_size)

    filenames = sorted(glob.glob(rf"{foldername}\image*.tiff"), key=natural_sort_key)
    filename_out = rf"{foldername}\{filename_out}"
    with tifffile.TiffWriter(filename_out) as tiff:
        for filename in filenames:
            filenumber = int(''.join(filter(str.isdigit, os.path.basename(filename)))) - 1
            print(filename)
            if filenumber % stack_size in selection:
                img = Image.open(filename)
                frame = np.array(img.getdata(), np.uint16)
                frame = 255 * (np.clip(frame, range[0], range[1]) - range[0]) / (range[1] - range[0])
                tiff.save(PIL2array(img, range=[90, 150]), compression=None)


def select_frames(foldername, range=None, selection=None, stack_size=2):

    filenames = sorted(glob.glob(rf"{foldername}\image*.tiff"), key=natural_sort_key)
    if selection is None:
        selection = np.arange(0, len(filenames)-1)

    if range is None:
        range = [0, len(filenames)-1]

    selected_filenames = []
    for filename in filenames[range[0]:range[1]]:
        filenumber = int(''.join(filter(str.isdigit, os.path.basename(filename)))) - 1
        print(filenumber, filenumber % stack_size)
        if filenumber % stack_size in selection:
            selected_filenames.append(filename)
    return selected_filenames


def open_tiff(foldername):
    filenames = glob.glob(rf"{foldername}\*.tiff")
    for filename in filenames:
        print(filename)
        # im = Image.open(filename)
        # im.show()
    return

    # im = Image.open(filename)
    # im.show()
    # imarray = np.array(im)
    # print(imarray.shape)

    im = plt.imread(filename)
    plt.imshow(im)
    plt.show()
    return

def aggegrate_images(filenames):
    for filename in filenames:
        im = Image.open(filename)

if __name__ == '__main__':
    folder = r'images'
    folder = r'C:\MeasurementData\201123\data_011'
    filenames = select_frames(folder, [0, 10], stack_size=2, selection=[1])
    for f in filenames:
        print(f)
    # create_multi_tiff(folder, stack_size=2, range=[90, 150], selection=None)
