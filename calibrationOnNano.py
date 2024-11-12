import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2

def rawFootage():
    frameName = 'Framing Camera'
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()

                if cv2.getWindowProperty(frameName, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(frameName,frame[::4,::4,:])
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('1'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print('Failed to open camera')

def unDistortedFootage():
    frameName = 'Undistorted Frame'
    yu, xu = myJazz.distortionMap()
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yu,xu]

                if cv2.getWindowProperty(frameName, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(frameName,frame[::4,::4,:])
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('2'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print('Failed to open camera')

if __name__ == '__main__':
    print(myJazz.gstreamer_pipeline(flip_method=0))
    rawFootage()
    unDistortedFootage()