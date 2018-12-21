import numpy as np
import scipy
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from local_norm import LocalNorm

from PIL import Image
import cv2

#imag_path ='/Users/yc1/Documents/Lu/dance_detection/data/dance/dance_xiuse/1130/7063583/'
#image = mpimg.imread(imag_path + '222352.jpg')
#im = scipy.ndimage.imread(imag_path + '222352.jpg', True)

imag_path ='/Users/yc1/Documents/Lu/dance_detection/data/dance/dance_xiuse/1130/7047338/'
image = mpimg.imread(imag_path + '134424.jpg')
im = scipy.ndimage.imread(imag_path + '134424.jpg', True)

im = im.astype('int32')
dx = sobel(im, 1, mode='wrap')
dy = sobel(im, 0, mode='wrap')

local_norm = LocalNorm(num=3)
dy = local_norm.run(dy)
dx = local_norm.run(dx)

mag = np.hypot(dx,dy)
mag *= 255.0/np.max(mag)

plt.figure()
plt.imshow(image, cmap='gray')
plt.figure()
plt.imshow(mag, cmap = 'gray')
plt.figure()
plt.imshow(dx, cmap = 'gray')
plt.figure()
plt.imshow(dy, cmap = 'gray')
plt.show()

