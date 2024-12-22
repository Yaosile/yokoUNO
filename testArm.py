import numpy as np
import serial
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2

from scipy import signal

def calibrationPoints():
    cx,cy = 100,100
    yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')
    frameName = 'Final Board Frame'
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                frame[cy-1:cy+1,:] = 255
                frame[:,cx-1:cx+1] = 255
                frame[frame.shape[0]//2:, :] = 0
                if cv2.getWindowProperty(frameName, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(frameName,frame[::2, ::2])
                else:
                    break


                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    card = frame.copy()
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
                    card = myJazz.rgb2hsv(card, 'SV')
                    card = card[...,1]*card[...,2]*255
                    card = card-card.min()
                    card = (card/card.max())*255
                    card[card>100] = 255
                    card[card<=100] = 0
                    card = myJazz.removeNoise(frame,5)
                    t,b,l,r = myJazz.boundingBox(frame)
                    cx,cy = myJazz.midPoint(t,b,l,r)
                    x,y = myJazz.pixelToCartesian(cx,cy,frame.shape[1],frame.shape[0])
                    l,r = myJazz.cartesianToScara(x,y)
                    l,r = int(np.rad2deg(l)*1000), int(np.rad2deg(r)*1000)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print('Failed to open camera')

if __name__ == '__main__':
    calibrationPoints()