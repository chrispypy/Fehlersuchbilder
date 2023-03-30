import skimage
import random
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import square, binary_closing, remove_small_objects
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import os
def load_random_image():
    
    PATH = os.path.join(np.random.choice([b'_FOTOS',b'_CARTOONS']))
    files = random.choice([[files,path] for path, _, files in os.walk(PATH) if path != PATH ])

    for file in files[0]:
        file = file.decode('utf-8')
        if '_a.tif' in file.lower():
            imA = Image.open(os.path.join(files[1].decode('utf-8'), file)).convert('RGB')
        if '_b.tif' in file.lower():
            imB = Image.open(os.path.join(files[1].decode('utf-8'), file)).convert('RGB')
        if '_lsg.tif' in file.lower():
            imLSG = Image.open(os.path.join(files[1].decode('utf-8'), file)).convert('RGB')
    return imA, imB, imLSG

    



# pyright: reportUnboundVariable=false 

imA, imB, imLSG = load_random_image()
imA_gray = np.array(imA.convert('L'))
imB_gray = np.array(imB.convert('L'))
diff =np.abs(imA_gray - imB_gray)
_,diff = skimage.metrics.structural_similarity(imA_gray, imB_gray, full = True) 
binary = diff < .95
binary_old = binary
# binary = diff > thresh
# binary = clear_border(binary)
num = np.Inf 

counter = 1

while num > 10:
    binary = binary_closing(binary, square(counter))
    binary = remove_small_objects(binary,counter)
    labels, num = skimage.measure.label(binary, background=0, connectivity=2, return_num = True)
    print(num)
    counter += 1
    # if counter > 30:
        # break

props = regionprops(labels) 
# main_components = np.zeros((10,4))
mask_ellipses = np.zeros(diff.shape)
mask_perimeter = np.zeros(diff.shape)

for i,region in enumerate(props):
    orientation = region.orientation
    x0,y0 = (int(c) for c in region.centroid)
    x1 = x0 - np.sin(orientation) * 0.5 * region.axis_major_length
    y1 = y0 - np.cos(orientation) * 0.5 * region.axis_major_length
    r_major = max(int(region.axis_major_length/1.8),20)
    r_minor = int(max(region.axis_minor_length/1.8,r_major/4))
    # main_components[i,:] = [x0,y0,x1,y1]

    rr, cc = draw.ellipse(x0, y0, r_major, r_minor, rotation = orientation, shape = diff.shape)
    rr_1, cc_1 = draw.ellipse_perimeter(x0, y0, r_major, r_minor, np.pi-orientation, shape = diff.shape)
    print(f'{r_major = }, {r_minor = }')

    mask_perimeter[rr_1,cc_1] = 1
    mask_ellipses[rr,cc] = 1

mask_perimeter_color = np.stack([mask_perimeter*c for c in (255,0,0,255)], axis = -1).astype(np.uint8)
mask_ellipses_color = np.stack([mask_ellipses*c for c in (255,255,0,80)], axis = -1).astype(np.uint8)
_,ax = plt.subplots(2,2)

# enhancer_contrast = ImageEnhance.Color(imA)
# imA_dark = enhancer_contrast.enhance(.5)
# enhancer_brightness = ImageEnhance.Brightness(imA_dark)
# imA_dark = enhancer_brightness.enhance(1.8)
# imA_dark.putalpha(255)
imA_dark = np.array(imA)
imA_dark[mask_ellipses == 0] = (imA_dark[mask_ellipses == 0]/2).astype(np.uint8)
# imA_dark[...,-1] = 255
ax[0,0].imshow(imA)
# ax[0,0].imshow(mask_perimeter_color)
ax[0,1].imshow(imB)
# ax[1,0].imshow(binary)
ax[1,0].imshow(imA_dark)

ax[1,0].imshow(mask_ellipses_color)

ax[1,1].imshow(binary_old)
plt.show()



