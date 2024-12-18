import cv2
import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary
""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""
# dist = [k1, k2, p1, p2, k3]
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3280,
    capture_height=2464,
    display_width=3280,
    display_height=2464,
    framerate=10,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def show_camera():
    dist = ana([-0.0639733628476694, -0.059022840140777, 0, 0, 0.0238818089164303])
    mtx = ana([
        [1.734239392051136E3,0,1.667798059392088E3],
        [0,1.729637617052701E3,1.195682065165660E3],
        [0,0,1],
    ])
    optimalMtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(3280,2464),1,(3280,2464))
    cv2.initUndistortRectifyMap(mtx, dist, None, optimalMtx, (3280,2464), cv2.CV_32FC1)
    t = 0
    window_title = "CSI Camera"
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if t:
                    frame = cv2.undistort(frame, mtx, dist, None, optimalMtx)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame = ana(frame)

                # display = ((255-frame + previous)*0.5).astype(np.uint8)
# Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame[::4,::4,:])
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
                elif keyCode == 97:
                    print('Kachow')
                    t = not t
                    # cv2.imwrite(f'{np.random.random()}.png',frame)
                previous = frame.copy()
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()
