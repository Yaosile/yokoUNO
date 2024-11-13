import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2
from PIL import Image
import feedForwardybackwards as cnn

import serial


def cardPositionFinder():
    output = np.zeros((100,100,1))
    cx,cy = 0,0
    blur = np.ones((5,5))
    blur /= blur.sum()
    boardFrame = 'Board'
    cardFrame = 'Card'
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')
    if video_capture.isOpened():
        try:
            cv2.namedWindow(boardFrame, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(cardFrame, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                frame[cy,:] = 255
                frame[:,cx] = 255
                frame[frame.shape[0]//2:,:] = 0
                if cv2.getWindowProperty(boardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(boardFrame,frame[::2,::2,:].astype(np.uint8))
                else:
                    break

                if cv2.getWindowProperty(cardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(cardFrame,output[::2,::2].astype(np.uint8))
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break

                elif key == ord(' '):
                    output = frame.astype(float)
                    output = myJazz.rgb2hsv(output,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    output = myJazz.convolveMultiplication(output, blur)
                    output = myJazz.threshHold(output, 254)
                    t,b,l,right = myJazz.boundingBox(output)
                    cx,cy = myJazz.midPoint(t,b,l,right)
                    output[cy,:] = 255
                    output[:,cx] = 255
                    x,y = myJazz.pixelToCartesian(cx,cy,frame.shape[1],frame.shape[0])
                    y-=30
                    l,r = myJazz.cartesianToScara(x,y)
                    test = f'{int((l*180/np.pi + 45)*1000)} {int((r*180/np.pi + 45)*1000)} 0'
                    print(test)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    thresh = int(input('Enter a threshold value: '))
    cardPositionFinder()