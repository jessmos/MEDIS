# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:15:58 2021

@author: jessm

The file accepts the temporal cube and slices a single image out.
The image is then log normalized between 0 and 1
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from numpy import inf
#fits_image_filename = fits('C:/Users/jessm/.spyder-py3/MEDIS_spy/Hip109427_temporal_align_PA.fits')

hdul = fits.open('C:/Users/jessm/.spyder-py3/MEDIS_spy/Hip109427_temporal_align_PA.fits')
inform=hdul.info()
savename='np_align26'

#print(inform)
#print(hdul.info())
#print("\n info \n",  align.info())
data = hdul[2].data
print(data.shape, data.dtype)
#plt.imshow(data)
plt.imshow(data[30,0,:,:].T)
plt.show()
align_slice=(data[26, 0,:,:].T) #20x #26 ##30 #38 ##40 #44
plt.imshow(align_slice)
plt.title("Before")
plt.show()
print("slice shape" , align_slice.shape)
#np.save('slice40', align_slice)

"""cropping the slice to only show the star image"""
slicecube=align_slice[96:176,18:98] #these values do not change, they center the image
plt.imshow(slicecube)
plt.show()

#np.save('align40', slicecube)
img=slicecube
print(img.shape, img.dtype)

"""this is a visual confirmation that the coronagraph is centered"""
circle = plt.Circle((40, 40), 16, color='r', fill=False)
circle2 = plt.Circle((40, 40), 38, color='b', fill=False)
plt.imshow(img, origin='lower')
plt.plot(40,40,'r*')
plt.gca().add_patch(circle)
plt.gca().add_patch(circle2)
plt.title("Demonstrated Centering")
plt.show()

"""saving in uint16 form for use in the Refined Otsu Pipeline""" 
x=slicecube
print("max", np.amax(x), "min", np.amin(x))
x[x == -inf] = 0
normalized = (x-np.amin(x))/(np.amax(x)-np.amin(x))
arrayuint= np.array(normalized*65535, dtype=np.uint16)
print("uint max", np.amax(arrayuint), "min", np.amin(arrayuint))
np.save(savename+'uint', arrayuint)


"""saving in float32 form for use in Intensity"""
img= np.array(normalized, dtype=np.float64)
#plt.imshow(img)
#plt.title("Float")
#plt.show()
print("float max", np.amax(img), "min", np.amin(img))
np.save(savename, img)
print(img.shape, img.dtype)