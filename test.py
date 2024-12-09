import numpy as np
from numpy import asanyarray as ana
import time

from scipy.io import wavfile
import scipy
import scipy.signal

from PIL import Image

import MelMyBoy
import myOwnLibrary as myJazz
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import feedForwardybackwards as cnn
import os

import MelMyBoy as audio

dira = 'CardSnaps/'

cards = os.listdir(dira)
choice = np.random.choice(cards)
choice = 'b+.png'

img = ana(Image.open(dira + choice))[...,::-1]
img = myJazz.rgb2hsv(img, 'SV')
img = (1-img[...,1])*img[...,2]*255

img = myJazz.adaptiveThreshold(img)
img = Image.fromarray(img.astype(np.uint8))
print(choice)
img.save('temp.png')