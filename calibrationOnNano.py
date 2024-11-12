import numpy as np
import myOwnLibrary as myJazz
import cv2

print(myJazz.gstreamer_pipeline(flip_method=0))
video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

frameName = 'Calibration View'

if video_capture.isOpened():
    try:
        window_handle = cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
        while True:
            ret_val, frame = video_capture.read()
            if cv2.getWindowProperty(frameName, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.imshow(frameName,frame[::4,::4].astype(np.uint8))
            else:
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
else:
    print('Failed to open camera')