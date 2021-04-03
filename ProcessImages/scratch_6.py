from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

filename = Path(r'C:\Users\noort\Downloads\20nmStack_2021-03-30-191748-0000.tif')

res = read_tiff(filename)
print(np.shape(res))
res = np.reshape(res,(768,1024,3))
plt.imshow(res[:,:,2]-res[:,:,0])
plt.show()
