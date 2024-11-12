import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2

def mainFootage():
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

                if cv2.getWindowProperty(boardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(boardFrame,frame[::2,::2,:])
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
if __name__ == '__main__':
    mainFootage()