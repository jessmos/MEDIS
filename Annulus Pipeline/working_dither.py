# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:28:39 2021

@author: jessm

The file accepts the dithered cube and slices a single image out.
The image is then log normalized between 0 and 1
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
#from mkidpipeline.hdf.photontable import Photontable
from numpy import inf

## View Dither Image Slice
# dither_cube = np.load(f'{scratch_path}/MEC_{target_name}_ditheredCube_forJessica.npz')
din = np.load(f'MEC_HIP109427_ditheredCube_forJessica.npz', mmap_mode='r')
list(din.files)
dither_cube = din['intensity_images']

slice = 12
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.suptitle(f'Total Pixel Count Image\n'
             f'Step={slice}')

dslice=dither_cube[slice].T
ax.imshow(dslice, interpolation='none')  # [70:140,10:90,:]


plt.show()

#slicecube=dslice[3:143, :]

#crop out the really bad 
slicecube=dslice[20:100,60:140]

print(dither_cube.shape, dslice.shape, slicecube.shape)

fig = plt.figure(frameon=False, figsize=(5.12, 5.12),dpi=100)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
#cax = ax.imshow(slicecube, interpolation='none')
cax = ax.imshow(slicecube,  origin='lower', cmap = "YlGnBu_r")

fig.savefig('dither12.png', dpi=100)
plt.show()


x=slicecube
print("max", np.amax(x), "min", np.amin(x))
x[x == -inf] = 0
normalized = (x-np.amin(x))/(np.amax(x)-np.amin(x))
arrayuint= np.array(normalized*65535, dtype=np.uint16)
print("uint max", np.amax(arrayuint), "min", np.amin(arrayuint))
np.save('np_dither12uint', arrayuint)

img=x
img= np.array(normalized, dtype=np.float64)
print("float max", np.amax(img), "min", np.amin(img), img.dtype)
np.save('np_dither12', img)