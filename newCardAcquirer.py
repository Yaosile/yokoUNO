import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2
from PIL import Image

def captureCard():
    boardFrame = 'Board'
    cardFrame = 'Card'
    output = np.ones((100,100,3))

    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.25 often disables auto-exposure
    video_capture.set(cv2.CAP_PROP_EXPOSURE, -5)         # Set a high exposure value (try different values)
    video_capture.set(cv2.CAP_PROP_GAIN, 100) 
    yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')
    if video_capture.isOpened():
        try:
            cv2.namedWindow(boardFrame, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(cardFrame, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                if cv2.getWindowProperty(boardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(boardFrame,frame[::2,::2,:].astype(np.uint8))
                else:
                    break

                if cv2.getWindowProperty(cardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(cardFrame,output.astype(np.uint8))
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    captureCard()