# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 20:51:13 2020

@author: jessm
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
import skimage.filters
#from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import color
from skimage import io

#image=plt.imread('rebinnedimagePT2.png')
#print(image.shape)


img=np.load('np_rebinned5e7.npy')

global_thresh = skimage.filters.threshold_otsu(img)
binary_global = img > global_thresh

#block_size = 45
#binary_adaptive = skimage.filters.threshold_local(img, block_size, offset=10)

block_size = 35
local_thresh = skimage.filters.threshold_local(img, block_size, offset=10)
binary_local = img >= local_thresh

#binary_adaptive = skimage.filters.threshold_local(img)

plt.imshow(img)
plt.title('Image')
plt.show()
plt.imshow(binary_global)
plt.title('Global thresholding')
plt.show()
plt.imshow(local_thresh )
plt.title('Adaptive thresholding')
plt.show()
plt.imshow(binary_local)
plt.title('Original >= Local')
plt.show()

fig, ax = skimage.filters.try_all_threshold(img, figsize=(6, 12), verbose=True)

