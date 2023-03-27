import skimage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
PATH = os.path.join('_CARTOONS','NTV22_09_S03_3 (Fehlersuche)')
imA = Image.open(os.path.join(PATH,'NTV22_09_S03_3_A.tif'))
imB = Image.open(os.path.join(PATH,'NTV22_09_S03_3_B.tif'))
imA_gray = np.asarray(imA.convert('L'))
imB_gray = np.asarray(imB.convert('L'))
diff = np.abs(imA_gray - imB_gray)
thresh = threshold_otsu(diff)
binary = diff > thresh
num = np.Inf 
counter = 1
while num > 10:
    binary = closing(diff > thresh, square(counter))
    labels, num = skimage.measure.label(binary, background=0, connectivity=2, return_num = True)
    counter += 1
props = regionprops(labels)
main_components = np.zeros((10,4))
mask = np.zeros(diff.shape)
for i,region in enumerate(props):
    orientation = region.orientation
    x0,y0 = region.centroid
    x1 = x0 - np.sin(orientation) * 0.5 * region.axis_major_length
    y1 = y0 - np.cos(orientation) * 0.5 * region.axis_major_length
    main_components[i,:] = [x0,y0,x1,y1]
    rr,cc = draw.ellipse(x0,y0,region.axis_major_length/2, region.axis_minor_length/2, rotation = orientation)
    mask[rr,cc] += i+1
_,ax = plt.subplots(2,2)
ax[0,0].imshow(imA)
ax[0,0].imshow(mask!=0, alpha = .5)
ax[0,1].imshow(imB)
ax[1,0].imshow(binary)
ax[1,1].imshow(mask)
plt.show()



