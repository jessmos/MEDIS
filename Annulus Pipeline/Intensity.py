"""
Created on Tue Jan 26 18:48:32 2021

@author: jessm

This is the heart of the pipeline
it saves the number of pixels, number of speckle pixels, percent speckle pixels, and average intensity per annulus
it also saves the locations and values of speckle pixels seperately for later use
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk
import math
import sys 

#sys.path.append("C:/Users/jessm/OneDrive/Documents/Coding/proper_v3.2.3_python_3.x")
#import proper
#import optics as opx
np.set_printoptions(edgeitems=20, linewidth=120)

"""this is computing the annulus size, these values are from MEDIS 
but they are NOT right for the MEC images
this makes sense because the scaling is different- these values are too  
honestly it might not even be right for the MEDIS images 
since I am doing this on the rebinned cube which is smaller than the original simulated image
something to look in to"""

wavelength=1*10**-6 #m
distance=7.8 #m
theta=1.22*wavelength/distance 
#pix=2.2 #millarcsec/pixels
radpix= 2.2*0.001*(math.pi/648000)
pixels=theta/radpix
pixels_int=round(pixels)
print("pixels per annulus:", pixels_int) #lamda over d 

image=np.load('np_rebinned5e7.npy')
thresh=np.load('thresh5e7.npy')
savename='table_test'


masked=image*thresh

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()
plt.tight_layout()

ax[0].imshow(image, origin='lower', cmap=plt.cm.gray)
ax[0].set_title('Inpainted MEC Image')
#ax[0].axis('off')

ax[1].imshow(thresh, origin='lower', cmap=plt.cm.gray)
ax[1].set_title('Otsu Thresholded Inpainted Image')
#ax[1].axis('off')

ax[2].imshow(masked, origin='lower', cmap=plt.cm.gray)
ax[2].set_title("Image with the Background Masked Out \nLeaving Only Speckles")
plt.show()

plt.imshow(masked, origin='lower', cmap=plt.cm.gray)
plt.title("Image with the Background Masked Out \nLeaving Only Speckles")
plt.show()


def draw_ring(x, y, r, shape, inside): #these are (x,y) coords, radius of circle, the shape of the file image, and the width of the annulus
    circle_image = np.zeros(shape, np.uint8)
    rr, cc = disk((x, y), r)
    circle_image[rr, cc] = 1 #this creates a solid circle of value 1
    rr, cc = disk((x, y), r-inside) #this sets the center of the circle to value 0, making a ring
    circle_image[rr, cc] = 0
    #remove this if you don't want the rings printing
    plt.imshow(circle_image, origin='lower', cmap=plt.cm.gray)
    plt.grid()
    plt.show()
    return(circle_image)

"""This giant function accepts the radius on the annulus, the threshold mask, and the image file 
it then computes and records analysis per annulus"""
def annulus(width, threshfile, img): 
    d=0
    firstwidth=width
    shape=threshfile.shape
    x0=(int((shape[0])/2))
    #x1=(int((shape[1])/2))
    
    tablerow=int(x0/firstwidth)
    #columns will be 
    #number of pixels in ring
    #number of speckles in ring
    #percentage speckle
    #average intensity
    rows, cols = (tablerow, 4)
    table = [[0 for i in range(cols)] for j in range(rows)]
    
    intensity_loc=[]
    intensity_val=[]
    
    masked=threshfile*img
    
    while width <= x0: 
        #here width is the outside radius of the circle
        ring=draw_ring(x0, x0, width, shape, firstwidth)
        specklering=ring*masked
        width+=firstwidth
        numwhitecircle=np.count_nonzero(ring)
        whitespeckle=(specklering[specklering>0])
        numwhitespeckle=len(whitespeckle)
        percent= (numwhitespeckle/numwhitecircle)*100
        avg_intensity=sum(whitespeckle) / len(whitespeckle)
        print("\nring", d+1)
        #print("outside of the ring is at", thickness)
        print("the number of pixels in this ring is", numwhitecircle, "and the number of speckle pixels in this ring is", numwhitespeckle)
        print("The percent of the ring that is speckle is", percent, "%")
        print("The average speckle intensity in this ring is", avg_intensity)
        
        #saving to table
        table[d][0]=numwhitecircle
        table[d][1]=numwhitespeckle
        table[d][2]=percent
        table[d][3]=avg_intensity
        d+=1

        #this is for getting the indexs and values of speckles
        #I dont actually use these yet
        index_array= np.argwhere(specklering != 0)
        index=index_array.tolist()
        intensity_loc.extend(index)
        intense_array=img[index_array]
        intense=intense_array.tolist()
        intensity_val.extend(intense)
        
    print("\namount of pixels with an intensity", len(intensity_loc), "\nversus the total number of pixels", shape[0]*shape[1])
    table=np.array(table)
    #print(table)
    np.save(savename, table)

annulus(pixels_int, thresh, image)

