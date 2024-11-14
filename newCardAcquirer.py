import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2
from PIL import Image

def captureCard():
    boardFrame = 'Board'
    cardFrame = 'Card'
    blur = np.ones((5,5))
    blur /= blur.sum()
    thresh = int(input('Enter a threshold value: '))
    yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            cv2.namedWindow(boardFrame, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(cardFrame, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                if cv2.getWindowProperty(boardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(boardFrame,frame[::2,::2,:].astype(np.uint8))
                else:
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()