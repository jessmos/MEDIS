# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 00:04:44 2021

@author: jessm

This is just plotting the results in a table
it probably definitely could be done more efficiently and look better
"""

import matplotlib.pyplot as plt
import numpy as np
import tables as pt

from astropy.table import QTable, Table, Column
from astropy import units as u

reimage=np.load('np_rebinned5e7.npy')
rethresh=np.load('thresh5e7.npy')
retable=np.load('table5e7.npy')

dimage=np.load('np_align30.npy')
dthresh=np.load('thresh_a30.npy')
dtable=np.load('table_a30.npy')


"""This is the plot"""
fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=False, sharey=False)
ax = axes.ravel()
#plt.tight_layout()

ary=ax[0].imshow(reimage, origin='lower',  cmap = "YlGnBu_r")
ax[0].set_title('MEDIS Rebinned Cube')
plt.colorbar(ary, ax=ax[0], fraction=0.046, pad=0.04)

imgplt=ax[1].imshow(reimage*rethresh, origin='lower',  cmap = "YlGnBu_r")
ax[1].set_title('Masked MEDIS Rebinned Cube \nOnly Speckles')
plt.colorbar(imgplt, ax=ax[1], fraction=0.046, pad=0.04)

ary=ax[2].imshow(dimage, origin='lower',  cmap = "YlGnBu_r")
ax[2].set_title('Temporal Cube Slice of MEC Image')
plt.colorbar(ary, ax=ax[2], fraction=0.046, pad=0.04)

imgplt=ax[3].imshow(dimage*dthresh, origin='lower',  cmap = "YlGnBu_r")
ax[3].set_title('Masked Temporal MEC Image \nOnly Speckles')
plt.colorbar(imgplt, ax=ax[3], fraction=0.046, pad=0.04)

plt.show()


"""This is the table"""
middle=np.array([0,0,0,0])

retable=np.vstack((np.round(retable, decimals=2)))
dtable=np.vstack((np.round(dtable, decimals=2)))

def reshape_rows(array1, array2):
    if array1.shape[0] > array2.shape[0]:
        resizea2=array2.copy()
        resizea2.resize(array1.shape[0], array2.shape[1])   
        reshaped=np.hstack([array1, resizea2])
        return(reshaped)
    if array1.shape[0] < array2.shape[0]:
        resizea1=array1.copy()
        resizea1.resize(array2.shape[0], array1.shape[1])   
        reshaped=np.hstack([resizea1,array2])
        return(reshaped)

sidebyside=reshape_rows(retable, dtable)


show=Table(sidebyside, names=('Sim Pixels', 'Sim Speckles', 'Sim Percent', 'Sim Intensity', 'Pixels', 'Speckles', 'Percent', 'Avg Intensity'))
show.pprint_all()
