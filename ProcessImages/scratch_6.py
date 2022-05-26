import numpy as np
import pandas as pd

import ProcessImages.ImageIO as im3
import matplotlib.pyplot as plt

im = np.random.normal(1,1,(100,100))

f_im = im3.filter_image(im, highpass=5)
plt.imshow(f_im)
plt.show()

