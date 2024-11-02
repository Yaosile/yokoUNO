import myOwnLibrary as myJazz

import numpy as np
from numpy import asanyarray as ana

from PIL import Image
import PIL

image = ana(Image.open('Images/rotatedCard.png'))

HSV = myJazz.rgb2hsv(image, 'SV')
image = HSV[...,1]*HSV[...,2]*255
image = myJazz.threshHold(image, 50)

t,b,l,right = myJazz.boundingBox(image)
cx,cy = myJazz.midPoint(t,b,l,right)
if False:
    image[t-2:t+2,:] = 255
    image[b-2:b+2,:] = 255
    image[:,l-2:l+2] = 255
    image[:,right-2:right+2] = 255

    image[:,cx-2:cx+2] = 255
    image[cy-2:cy+2,:] = 255

r = myJazz.getRadius(image,cx,cy) + 1
theta = myJazz.getRotation(image, cx, cy, r)
print(np.rad2deg(theta))
temp = myJazz.rotate(ana(Image.open('Images/rotatedCard.png'))[cy-r:cy+r, cx-r:cx+r], theta)
HSV = myJazz.rgb2hsv(temp, 'SV')
image = HSV[...,1]*HSV[...,2]*255
image = myJazz.threshHold(image, 50)
t,b,l,r = myJazz.boundingBox(image)
# image = myJazz.rotate(ana(Image.open('Images/rotatedCard.png'))[cy-r:cy+r, cx-r:cx+r], theta)
image = temp[t-15:b+15,l-15:r+15]



# cx,cy = myJazz.midPoint(t,b,l,right)
# r = myJazz.getRadius(image, cx,cy)+22
# theta = myJazz.getRotation(image, cx,cy,r)
# image = myJazz.rotate(ana(Image.open('Images/rotatedCard.png'))[-r+cy:r+cy, -r+cx:r+cx], theta)



image = Image.fromarray(image.astype(np.uint8))
image.show()





# table = ana(Image.open('Images/table.jpg')).copy()
# cards = ana(Image.open('Images/cards.jpg')).copy()

# table = table[:, 35:, :]
# cards = cards[:, :cards.shape[1]//4, :]
# yc,xc = cards.shape[:2]
# yt,xt = table.shape[:2]
# yt = yt//2
# xt = xt//2
# yc = yc//2
# xc = xc//2
# # table[yt-yc:yt+yc,xt-xc:xt+xc] = cards
# alpha = np.ones(cards.shape[:2])[:,:,None]*255
# alpha[np.sum(cards,axis=2)>720] = 0
# cards = np.dstack((cards, alpha))
# table = myJazz.overlayImage(table, cards,rot = 35, centre=-1)
# # # # HSV = myJazz.rgb2hsv(table, 'SV')
# # # # table = HSV[...,1]*HSV[...,2]*255
# # # # table = myJazz.threshHold(table, 50)


# cards = Image.fromarray(cards.astype(np.uint8)).rotate(35)
# table = Image.fromarray(table.astype(np.uint8))
# table.show()