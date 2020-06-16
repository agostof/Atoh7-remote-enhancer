from __future__ import division
from __future__ import print_function
from past.utils import old_div
import sys, cv2, numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from collections import Counter

fnamecore, windowRad = sys.argv[1], int(sys.argv[2])

cellLocArry = np.load(fnamecore+'_cellLoc.npy')
mask = old_div(cv2.imread(fnamecore+'_retina_mask.tif',0),255)
savename1 = fnamecore+'_hm.tif'
savename2 = fnamecore+'_hm.txt'

x=np.arange(0,windowRad*2)
y=np.arange(0,windowRad*2)
r=windowRad
cellLocImg = np.zeros(mask.shape)

count=0
for cy,cx in cellLocArry:
    count+=1
    if count%100 == 0:
        print(('Cells processed: {0}').format(count))
    y0,y1 = cy-windowRad, cy+windowRad
    x0,x1 = cx-windowRad, cx+windowRad
    circle = (x[np.newaxis,:]-windowRad+1)**2 + (y[:,np.newaxis]-windowRad+1)**2 < r**2
    cellLocSquare = cellLocImg[y0:y1,x0:x1]
    cellLocSquare[circle]+=1
    cellLocImg[y0:y1,x0:x1] = cellLocSquare

cellLocImg = cellLocImg * mask

plt.imshow(cellLocImg, cmap = 'nipy_spectral') #cmap is the color spectrum used for the density mapping and colorbar
plt.axis('off')
plt.clim(0, 65) #scale for colorbar, allows for standardizing the scale
cb = plt.colorbar()
plt.savefig(savename1, bbox_inches = 'tight')
plt.show()

arr = cellLocImg.flatten()

with open(savename2, 'w') as fp:
    fp.write('\n'.join('{}\t{}'.format(x[0],x[1]) for x in list(Counter(arr).items())))
