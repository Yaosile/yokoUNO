import myOwnLibrary as myJazz

import numpy as np
from numpy import asanyarray as ana

from PIL import Image
import PIL

table = ana(Image.open('Images/table.jpg')).copy()
cards = ana(Image.open('Images/cards.jpg')).copy()

table = table[:, 35:, :]
# cards = cards[::2, ::2, :]
yc,xc = cards.shape[:2]
yt,xt = table.shape[:2]
yt = yt//2
xt = xt//2
yc = yc//2
xc = xc//2
table[yt-yc:yt+yc,xt-xc:xt+xc] = cards

HSV = myJazz.rgb2hsv(table, 'SV')
table = HSV[...,1]*HSV[...,2]*255
table = myJazz.threshHold(table, 50)


cards = Image.fromarray(cards.astype(np.uint8))
table = Image.fromarray(table.astype(np.uint8))
table.show()