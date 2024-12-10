import numpy as np
from numpy import asanyarray as ana
import time

from scipy.io import wavfile
import scipy
import scipy.signal

from PIL import Image
import pillow_heif

import MelMyBoy
import myOwnLibrary as myJazz
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import feedForwardybackwards as cnn
import os
import cv2

import MelMyBoy as audio

dira = 'ogTemplates/'
saveDira = 'templates/'

cards = [i for i in os.listdir(dira) if 'png' in i]
cards.sort()
stack = []

for choice in cards:
    img = ana(Image.open(dira + choice))
    img = myJazz.isolateCard(img)
    img = myJazz.scaleImage(img,1256,2048)
    img = myJazz.isolateValue(img)
    # img = myJazz.scaleImage(img,1256,2048)
    img = myJazz.threshHold(img, 127)
    stack.append(img)
    img = Image.fromarray(img.astype(np.uint8))
    print(choice)
    
temp = np.hstack(stack)
img = Image.fromarray(temp.astype(np.uint8))
img.save('templates/template.png')


# choice = np.random.choice(cards)
# choice = '+.png'

# img = ana(Image.open(dira + choice))

# img = myJazz.isolateCard(img)
# img = myJazz.isolateValue(img)

# thresh = np.where(img < 255)
# img = np.zeros_like(img)
# img[thresh] = 255


# img = Image.fromarray(img.astype(np.uint8))
# img.save('temp.png')


# # dira = 'CardSnaps/'

# # cards = os.listdir(dira)
# # choice = np.random.choice(cards)
# # choice = 'b+.png'

# # img = ana(Image.o
# pen(dira + choice))[...,::-1]
# # img = myJazz.rgb2hsv(img, 'SV')
# # img = (1-img[...,1])*img[...,2]*255

# # img = myJazz.adaptiveThreshold(img)
# # img = Image.fromarray(img.astype(np.uint8))
# # print(choice)
# # img.save('temp.png')

# pillow_heif.register_heif_opener()

# dira = 'ogTemplates/'
# cards = [i for i in os.listdir(dira) if 'png' in i]
# choice = np.random.choice(cards)
# choice = '1.png'

# img = ana(Image.open(dira + choice))
# frame = myJazz.rgb2hsv(img, 'SV')
# frame = frame[...,1]*frame[...,2]*255
# frame = myJazz.adaptiveThreshold(frame)
# thresh = np.where(frame < 100)
# frame = np.zeros_like(frame)
# frame[thresh] = 255
# t,b,l,r = myJazz.boundingBox(frame)
# # img = myJazz.isolateCard(img)
# img = frame.copy()
# img = np.pad(img, ((0,0),((4032-3024)//2, (4032-3024)//2)))
# img = myJazz.rotate(img, np.deg2rad(-2.1239))
# center = tuple([img.shape[0]//2]*2)

# diffs = []
# for i in range(1800):
#     i /= 20
#     print(i)
#     mat = cv2.getRotationMatrix2D(center, i, 1)
#     temp = cv2.warpAffine(img, mat, img.shape)


#     # temp = myJazz.rotate(img, np.deg2rad(i))
#     t,b,l,r = myJazz.boundingBox(temp)
#     diffs.append((r-l)/(b-t))

# np.save('angleLUT.npy', diffs)

# plt.plot(diffs)
# plt.show()
# # img = Image.fromarray(img.astype(np.uint8))
# # img.save('temp.png')