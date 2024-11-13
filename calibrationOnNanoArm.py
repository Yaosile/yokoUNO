import numpy as np
import serial
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2

from scipy import signal

thresh = 0

src = []
dst = []

def calibrationPoints():
    global thresh
    thresh = int(input('Enter the threshold: '))
    cx,cy = 0,0
    gx,gy = 0,0
    blur = np.ones((5,5))
    blur = blur/blur.sum()
    src = []
    dst = []
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
                frame[gy-1:gy+1,:,2] = 0
                frame[:,gx-1:gx+1,2] = 0
                

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

                elif key == ord('n'):
                    output = frame.astype(float)
                    output = myJazz.rgb2hsv(output,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    for i in range(3): 
                        output = signal.fftconvolve(output, blur, mode='same')
                    output = myJazz.threshHold(output, 254)
                    t,b,l,right = myJazz.boundingBox(output)
                    gx,gy = myJazz.midPoint(t,b,l,right)
                    print(gx,gy)

                elif key == ord('m'):
                    x,y = myJazz.pixelToCartesian(cx,cy,frame.shape[1],frame.shape[0])
                    l,r = myJazz.cartesianToScara(x,y)
                    test = f'{int((l*180/np.pi + 45)*1000)} {int((r*180/np.pi + 45)*1000)} 0'

                    ser = serial.Serial('/dev/ttyUSB0', 115200)
                    ser.write(test.encode())
                    ser.close()
                elif key == ord('o'):
                    test = '220000 60000 0'
                    ser = serial.Serial('/dev/ttyUSB0', 115200)
                    ser.write(test.encode())
                    ser.close()
                elif key == ord('e'):
                    src.append([cx,cy])
                    dst.append([gx,gy])

                elif key == ord('f'):
                    src = ana(src)
                    dst = ana(dst)

                    print(src)
                    print(dst)
                

                    
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print('Failed to open camera')

if __name__ == '__main__':
    calibrationPoints()