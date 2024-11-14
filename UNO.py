import numpy as np
import serial
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2

from scipy import signal

predefinedLocations = {
    'draw': 's220000 60000',
    'hand1': 's200000 100000', #TBD
    'hand2': 's220000 120000', #TBD
    'discard': 's220000 60000' #TBD
}

blur = np.ones((5,5))
blur = blur/blur.sum()

def determineLocations():
    while True:
        location = input('Enter a location(draw, hand1, hand2, discard): ')
        if location in list(predefinedLocations.keys()):
            ser = serial.Serial('/dev/ttyUSB0', 115200)
            ser.write(predefinedLocations[location].encode())
            ser.close()
        else:
            print('That location is not found')

def draw7Cards():
    src = np.load('src.npy')
    dst = np.load('dst.npy')
    yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')
    frameName = 'Drawing 7 Cards'
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                if cv2.getWindowProperty(frameName, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(frameName,frame[::2, ::2])
                else:
                    break

                frame[frame.shape[0]//2:, :] = 0

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    output = frame.astype(float)
                    output = myJazz.rgb2hsv(output,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    for i in range(3): 
                        output = signal.fftconvolve(output, blur, mode='same')
                    output = myJazz.threshHold(output, 254)
                    t,b,l,right = myJazz.boundingBox(output)
                    cx,cy = myJazz.midPoint(t,b,l,right)
                    print(cx,cy)

                elif key == ord('m'):
                    gx,gy = myJazz.armCalibrationHomo(src,dst,cx,cy)
                    x,y = myJazz.pixelToCartesian(gx,gy,frame.shape[1],frame.shape[0])
                    l,r = myJazz.cartesianToScara(x,y)
                    test = f's{int((l*180/np.pi + 45)*1000)} {int((r*180/np.pi + 45)*1000)}'

                    ser = serial.Serial('/dev/ttyUSB0', 115200)
                    ser.write(test.encode())
                    ser.close()
                elif key == ord('o'):
                    test = 's220000 60000'
                    ser = serial.Serial('/dev/ttyUSB0', 115200)
                    ser.write(test.encode())
                    ser.close()
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print('Failed to open camera')

if __name__ == '__main__':
    determineLocations()
    draw7Cards()