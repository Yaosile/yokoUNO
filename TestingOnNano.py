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
import unoLogic
import robotCommands

import MelMyBoy as audio

cap = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0, framerate=1), cv2.CAP_GSTREAMER)
yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')

if cap.isOpened():
    try:
        ret,frame = cap.read()
        time.sleep(3)
        window_handle = cv2.namedWindow('card', cv2.WINDOW_AUTOSIZE)
        prev = []
        while True:
            ret,frame = cap.read()
            frame = frame[yuw,xuw]

            frame = myJazz.isolateCard(frame)

            if cv2.getWindowProperty('card', cv2.WND_PROP_AUTOSIZE) >= 0:
                print('displaying')
                cv2.imshow('card',frame.astype(np.uint8))
            else:
                break
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            prev = frame.copy()
    finally:
        print('found the card played')