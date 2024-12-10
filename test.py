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

dira = 'CardSnaps/'

cardCount = 13

guessIndex = ['+', *[f'{i}' for i in range(10)], 'r', 's']

template = ana(Image.open('templates/template.png'))

dimensions = (template.shape[1]//cardCount, template.shape[0])
print(dimensions)
cards = os.listdir(dira)

guesses = 0
correctness = 0
incorrect = []
for choice in cards:
    correct = choice[1]
    ref = ana(Image.open(dira + choice))
    ref = myJazz.scaleImage(ref,*dimensions)
    ref = myJazz.isolateValue(ref)
    guess = guessIndex[myJazz.compareTemplate(ref, template)]
    print(correct, guess)
    guesses += 1
    if correct == guess:
        correctness += 1
    else:
        incorrect.append(choice)

print(correctness, guesses, correctness/guesses)
print(incorrect)

# incorrect = ['b8.png', 'r3.png', 'r8.png', 'b4.png', 'b3.png', 'y8.png']
# choice = incorrect[2]
# choice = 'b0.png'
# ref = ana(Image.open(dira + choice))
# ref = myJazz.scaleImage(ref,*dimensions)

ref = myJazz.isolateValue(ref)

# print(myJazz.compareTemplate(ref, template))

img = Image.fromarray(ref.astype(np.uint8))
img.show()