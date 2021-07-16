# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:01:35 2021

source: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html

This code is meant to use and compare three different blob detection algorithms
The Determinant of Heissian seemed the most promising at first,
 but it is failing completely for the recent images,
 I think it has something to do with the artificially increased flux
"""

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

"""This is the data that this code works for/was written for"""
#image = data.hubble_deep_field()[0:500, 0:500]
#thresh = color.rgb2gray(image)

"""this is our data"""
image= img_as_ubyte(io.imread('rebinnedgraph.png'))
thresh=color.rgb2gray(image)


blobs_log = blob_log(thresh, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
blobs_dog = blob_dog(thresh, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(thresh, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    circlenum=0
    ax[idx].set_title(title)
    ax[idx].imshow(image)
    for blob in blobs:
        y, x, r = blob
        #print(blob.shape)
        c = plt.Circle((x, y), r, color=color, linewidth=1.5, fill=False)
        ax[idx].add_patch(c)
        circlenum+=1 #this is the line I added to count the number of circles
    ax[idx].set_axis_off()
    print("Number of circles", circlenum)

plt.tight_layout()
fig.savefig('circles12.png', dpi=100)
plt.show()

fig2 = plt.figure()
ax2 = plt.Axes(fig2, [0., 0., 1., 1.])     
fig2.add_axes(ax2)

print(blobs_doh.shape)
#print(blobs_doh)
sequence2 = zip(blobs_doh, 'red', 'Determinant of Hessian')
circlenum2=0

for s in blobs_doh:
    y,x, r = s
    c = plt.Circle((x, y), r, color='red', linewidth=1.5, fill=False)
    ax2.add_patch(c)
    circlenum2+=1
print("Number of circles 2", circlenum2)

ax2.set_title('Determinant of Hessian')
ax2.set_axis_off()
ax2.imshow(thresh, cmap=plt.cm.gray)