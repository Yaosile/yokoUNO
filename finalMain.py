# test

import numpy as np
import myOwnLibrary as myJazz
from numpy import asanyarray as ana
from PIL import Image
import cv2
scaling = 1
cameraWidth = 3264//scaling
cameraHeight = 2464//scaling

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=cameraWidth,
    capture_height=cameraHeight,
    display_width=cameraWidth,
    display_height=cameraHeight,
    framerate=1,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
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

"""
1219, 616
1072, 1851
2276, 1851
2115, 609
"""
dist = ana([-0.0639733628476694, -0.059022840140777, 0, 0, 0.0238818089164303])
mtx = ana([
    [1.734239392051136E3,0,1.667798059392088E3],
    [0,1.729637617052701E3,1.195682065165660E3],
    [0,0,1],
])/scaling
src = [
    [1328, 589],
    [1193, 1716],
    [2270, 1764],
    [2182, 594],
]

blur = np.ones((5,5))
blur /= blur.sum()
boardSize = (517, 605)
def cameraCalibration():
    output = np.zeros((605, 517))
    thresh = 50
    key = input('enter a number: ')
    if key == '1':
        print('calculating distortion map')
        yu, xu = myJazz.distortionMap(dist, mtx, cameraWidth, cameraHeight)
        print('calculating perspective map')
        yw, xw = myJazz.unwarpMap(src, *boardSize, cameraHeight, cameraHeight)
        print('calculating final transform')
        yuw, xuw = myJazz.getFinalTransform(yw,xw,yu,xu)
        np.save('mapY.npy', yuw)
        np.save('mapX.npy', xuw)
    elif key == '2':
        thresh = int(input('enter a threshold: '))
    yuw = np.load('mapY.npy')
    xuw = np.load('mapX.npy')

    # output = np.zeros((*boardSize,))
    window_title = "CSI Camera"

    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title,output.astype(np.uint8))
                else:
                    break



                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('w'):
                    Image.fromarray(frame.astype(np.uint8)).save('Images/Screenshot.png')
                    print('Kachow')
                    break
                elif key == ord(' '):
                    frame = frame[yuw,xuw]
                    frame = frame.astype(float)
                    output = myJazz.rgb2hsv(frame,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    output = myJazz.convolveMultiplication(output, blur)
                    output = myJazz.threshHold(output, 254)
                    t,b,l,r = myJazz.boundingBox(output)
                    x,y = myJazz.midPoint(t,b,l,r)
                    output[y,:] = 255
                    output[:,x] = 255
                    x,y = myJazz.pixelToCartesian(x,y,517,605)
                    l,r = myJazz.cartesianToScara(x,y-60)
                    print('snap')
                    print(l*180/np.pi + 45, r*180/np.pi + 45)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == '__main__':
    cameraCalibration()
    