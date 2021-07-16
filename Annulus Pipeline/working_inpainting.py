import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.restoration import inpaint
from numpy import inf
"""
Created on Tue May 25 2021

@author: jessm

code sourced from https://scikit-image.org/docs/dev/auto_examples/filters/plot_inpaint.html

This file inpaints a temporal cube slice
The goal is to fill in the dead pixels 
Perhaps a superior approach would be to mask out the dead pixels 
(either with a baseline threshold or include the low energy pixels with the Refined Otsu Thresholding)
and apply that mask to the MEDIS simulated image. It is less pretty but we 
are not simulating any data)
"""

savename='inpaint_a38'
image=np.load('np_align38.npy')
thresh=np.load('thresh_a38.npy')
thresh= np.invert(thresh)

image_orig = np.expand_dims(image, axis=2)

print(image_orig.shape, image_orig.dtype)
#print(thresh.shape, thresh.dtype)
print("shes thinking")

"""simple threshold mask"""
boolimage = image > 0.01 #keeps everything with pixel value over set number
mask= np.invert(boolimage)

print(mask.shape, mask.dtype)

 
"""Apply defect mask to the image"""
image_defect = image_orig * ~mask[..., np.newaxis]

"""inpaint the masked part"""
image_result = inpaint.inpaint_biharmonic(image_defect, mask,
                                          multichannel=True)

"""plotting"""
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15,5))
ax = axes.ravel()

ax[0].set_title('Original image')
ax[0].imshow(image_orig, origin='lower')

ax[1].set_title('Mask')
ax[1].imshow(image_defect, origin='lower', cmap=plt.cm.gray)

ax[2].set_title('Inpainted image')
ax[2].imshow(image_result, origin='lower')

print("float max", np.amax(image_result), "min", np.amin(image_result))

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()

"""reshaping to 2d"""
#print("image result", image_result.shape, image_result.dtype)
result = image_result[:, :, 0]
#print("result", result.shape, result.dtype)
np.save(savename, result)


"""log normalizing between 0 and 1"""
x=result
print("max", np.amax(x), "min", np.amin(x), x.dtype)
x[x == -inf] = 0
normalized = (x-np.amin(x))/(np.amax(x)-np.amin(x))

"""saving uint16"""
arrayuint= np.array(normalized*65535, dtype=np.uint16)
print("uint max", np.amax(arrayuint), "min", np.amin(arrayuint), arrayuint.dtype)
np.save(savename+'uint', arrayuint)

"""saving float64"""
img= np.array(normalized, dtype=np.float64)
print("float max", np.amax(img), "min", np.amin(img))
np.save(savename, img)
print(img.shape, img.dtype)


"""show centering again just for another visual"""
circle = plt.Circle((40, 40), 16, color='r', fill=False)
circle2 = plt.Circle((40, 40), 38, color='b', fill=False)
plt.imshow(img, origin='lower')
plt.plot(40,40,'r*')
plt.gca().add_patch(circle)
plt.gca().add_patch(circle2)
plt.title("Centering")
plt.show()