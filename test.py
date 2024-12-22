import numpy as np
from numpy import asanyarray as ana
import time

from scipy.io import wavfile
import scipy
import scipy.signal

from PIL import Image
# import pillow_heif

import MelMyBoy
import myOwnLibrary as myJazz
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import feedForwardybackwards as cnn
import os
import cv2

import MelMyBoy as audio


card = ana(Image.open('test.png'))[...,::-1]

lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
# Split LAB channels
l, a, b = cv2.split(lab)
# Apply CLAHE to the L channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
# Merge channels and convert back to BGR color space
limg = cv2.merge((cl, a, b))
card = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
card = card - card.min()
card = card/card.max()
card = card * 255

card = myJazz.isolateCard(card)
card = myJazz.isolateValue(card)
guess = myJazz.compareTemplate(card)

print(myJazz.guessIndex[guess])

img = Image.fromarray(card.astype(np.uint8))
img.show()

# mapX = np.load('xMap.npy')
# mapY = np.load('yMap.npy')

# xDiff = np.max(mapX) - np.min(mapX)
# yDiff = np.max(mapY) - np.min(mapY)
# print(mapX.shape, mapY.shape)
# print(xDiff, yDiff)

# dira = 'CardSnaps/'

# cardCount = 13

# guessIndex = ['+', *[f'{i}' for i in range(10)], 'r', 's']

# template = ana(Image.open('templates/template.png'))

# dimensions = (template.shape[1]//cardCount, template.shape[0])
# print(dimensions)
# cards = os.listdir(dira)

# guesses = 0
# correctness = 0
# incorrect = []
# for choice in cards:
#     correct = choice[1]
#     ref = ana(Image.open(dira + choice))
#     ref = myJazz.scaleImage(ref,*dimensions)
#     ref = myJazz.isolateValue(ref)
#     guess = guessIndex[myJazz.compareTemplate(ref, template)]
#     print(correct, guess)
#     guesses += 1
#     if correct == guess:
#         correctness += 1
#     else:
#         incorrect.append(choice)

# print(correctness, guesses, correctness/guesses)
# print(incorrect)

# # incorrect = ['b8.png', 'r3.png', 'r8.png', 'b4.png', 'b3.png', 'y8.png']
# # choice = incorrect[2]
# # choice = 'b0.png'
# # ref = ana(Image.open(dira + choice))
# # ref = myJazz.scaleImage(ref,*dimensions)

# ref = myJazz.isolateValue(ref)

# # print(myJazz.compareTemplate(ref, template))

# img = Image.fromarray(ref.astype(np.uint8))
# img.show()