import re
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


root_path = os.path.abspath(os.path.join(os.getcwd() ,"../../"))
datapath= os.path.join(root_path,"ArcGis\Projects\HELM\data\Azavea\Azavea_processed\Labels")

mask_main = os.path.join(datapath,"kilmacahill_mask135_15.tif")
mask_1 = os.path.join(datapath,"kilmacahill_mask225_15.tif")

output_dir = datapath

mainmask_array = np.asarray(Image.open(mask_main))
mainmask_array = mainmask_array.copy()

mask_array1 = np.asarray(Image.open(mask_1))
mask_array1 = mask_array1.copy()

print(np.shape(mainmask_array))

fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(mainmask_array)
ax[1].imshow(mask_array1)
fig.suptitle("Original",fontsize=26)
plt.show()

H,W = np.shape(mainmask_array)
n_x,n_y = 3,3
x,y = int(W/6*(n_x-1)),int(H/6*(n_y-1))
h,w = int(H/6),int(W/6)

fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(mainmask_array[y:y+h,x:x+w])
ax[1].imshow(mask_array1[y:y+h,x:x+w])
fig.suptitle("Grid to be swapped",fontsize=26)
plt.show()

mainmask_array_copy = mainmask_array.copy()
mainmask_array[y:y+h,x:x+w] = mask_array1[y:y+h,x:x+w]
fig,ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(mainmask_array_copy )
ax[1].imshow(mainmask_array)
ax[2].imshow(mask_array1)
fig.suptitle("After swapping",fontsize=26)
plt.show()

newMask = Image.fromarray(mainmask_array)
newMaskfilename = os.path.join(output_dir ,"kilmacahill_label.tif")
newMask.save(newMaskfilename)


