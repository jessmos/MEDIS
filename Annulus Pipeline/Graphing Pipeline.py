
"""
Created on Tue Dec 29 19:42:32 2020

@author: jessm

This file formats and plots the .h5 and fields files from the MEDIS simulation. 
There is still unresloved ringing in the fields plot. 

"""


import os
import matplotlib.pyplot as plt
import numpy as np
import tables as pt
from matplotlib.colors import LogNorm, SymLogNorm
from skimage.util import img_as_ubyte
from skimage.util import invert
from skimage import color
from skimage import io
from numpy import inf
#import optics as opx


#c=os.chdir('C:/Users/jessm/PycharmProjects')
#print(c)
import sys 
sys.path.append("C:/Users/jessm/OneDrive/Documents/Coding/proper_v3.2.3_python_3.x")


def open_obs_sequence_hdf5(obs_seq_file='hyper.h5'):
    """opens existing obs sequence .h5 file and returns it"""
    # hdf5_path = "my_data.hdf5"
    read_hdf5_file = pt.open_file(obs_seq_file, mode='r')
    # Here we slice [:] all the data back into memory, then operate on it
    obs_sequence = read_hdf5_file.root.data[:]
    # hdf5_clusters = read_hdf5_file.root.clusters[:]
    read_hdf5_file.close()
    return obs_sequence

def cpx_to_intensity(data_in):
    """
    converts complex data to units of intensity

    WARNING: if you sum the data sequence over object or wavelength with simple case of np.sum(), must be done AFTER
    converting to intensity, else results are invalid
    """
    return np.abs(data_in)**2


def crop_center(img):
    y,x = img.shape
    if img.shape[0]<img.shape[1]:
        cropx=img.shape[0]
        startx = x//2-(cropx//2)
        return img[:,startx:startx+cropx]
    elif img.shape[1]<img.shape[0]:
        cropy=img.shape[1]
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,:]
    else :
        print("it is already a cube")
        return img
    
#rebinned = open_obs_sequence_hdf5('C:/Users/jessm/OneDrive/Documents/Coding/rebinned_cube_PT2.h5')
#rebinned = open_obs_sequence_hdf5('C:/Users/jessm/.spyder-py3/MEDIS_spy/rebinned_cube_418_5e7.h5')
rebinned = open_obs_sequence_hdf5('C:/Users/jessm/.spyder-py3/MEDIS_spy/rebinned_cube5e8.h5')
#C:\Users\jessm\.spyder-py3\MEDIS_spy\
savename='np_rebinned5e8'

"""looking at fields"""
fields0 = open_obs_sequence_hdf5('C:/Users/jessm/.spyder-py3/MEDIS_spy/fields5e8.h5')
fields=fields0.astype(float)
#h5 file=complex image
#'timesteps', 'save planes', 'wavelengths', 'astronomical bodies', 'x', 'y'
print("Fields shape", fields.shape)
focal_sun=rebinned[0,-1,:,:]
focal_planet=fields[0,-1,:,:,:,:]
print("focal planet shape", focal_planet.shape)
print("rebinned cube shape", rebinned.shape)
#FOR REBINNED CUBE
#-no object or plane axis
#-rectangle

"""plotting fields"""
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#cax = ax.imshow(np.sum(focal_planet, axis=(0,1)), vmin=1e-9, vmax=1e-4, origin='lower', norm=SymLogNorm(1e-10))
cax = ax.imshow(np.sum(cpx_to_intensity(focal_planet), axis=(0,1)), origin='lower', norm=LogNorm(vmin=1e-7, vmax=1e-3), cmap = "YlGnBu_r")
plt.title("Star and Planet Broadband - Unresolved Ringing")
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
cb = plt.colorbar(cax)
plt.show()

"""cropping rebinned cube into cube"""
#print(crop_rebinned.shape)
rebinsum= np.sum(rebinned, axis=(0,1))
print("this is before cropping \n rebinned sum =", rebinsum.shape)
rebinsum=crop_center(rebinsum)
print("this is after cropping \n rebinned sum =", rebinsum.shape)

"""plotting lognorm rebinned cube"""
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#cax=ax.imshow(np.sum(rebinned))
cax = ax.imshow(rebinsum, origin='lower', norm=SymLogNorm(1e-10,vmin=1e-1, base=np.e), cmap = "YlGnBu_r")
#SymLogNorm values were hand selected
#Symlognorm only uses positive values, but thats ok because we only have positive values
plt.title("Rebinned Cube")
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
cb = plt.colorbar(cax)
plt.show()

"""log normalizing from 0 to 1"""
x=np.log(rebinsum)
print("max", np.amax(x), "min", np.amin(x))
x[x == -inf] = 0
normalized = (x-np.amin(x))/(np.amax(x)-np.amin(x))
imguint= np.array(normalized*65535, dtype=np.uint16)
#img=img/10
img= np.array(normalized, dtype=np.float32)
#print(img)
print("log normalized max=", np.amax(img), "log normalized min=", np.amin(img))


"""The log normalized rebinned cube image is saved in uint16 form (inferior) for use in the 
Refined Otsu Pipeline, and also saved in float32 form (superior) for use in 
the Intensity file."""

np.save(savename+'uint', imguint)
np.save(savename, img)


"""plotting"""
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)
ax = axes.ravel()

ary=ax[0].imshow(imguint, origin='lower', cmap=plt.cm.gray)
ax[0].set_title('Log Normalized in uint16')
plt.colorbar(ary, ax=ax[0], fraction=0.046, pad=0.04)

imgplt=ax[1].imshow(img, origin='lower', cmap=plt.cm.gray)
ax[1].set_title('Log Normalized in float32')
plt.colorbar(imgplt, ax=ax[1], fraction=0.046, pad=0.04)

plt.show()


"""plotting again"""
fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)
ax = axes.ravel()

ary=ax[0].imshow(rebinsum, origin='lower', cmap=plt.cm.gray)
ax[0].set_title('Rebinned Cube')
plt.colorbar(ary, ax=ax[0], fraction=0.046, pad=0.04)


imgplt=ax[1].imshow(img, origin='lower', cmap=plt.cm.gray)
ax[1].set_title('Log Normalized Rebinned Cube')
plt.colorbar(imgplt, ax=ax[1], fraction=0.046, pad=0.04)

plt.show()

fig.savefig('dither12.png', dpi=100)
plt.show()


"""creating an image to save"""
fig = plt.figure(frameon=False, figsize=(5.12, 5.12),dpi=100)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
cax = ax.imshow(img,  origin='lower', cmap = "YlGnBu_r")

fig.savefig('rebinnedgraph.png', dpi=100)
plt.show()
