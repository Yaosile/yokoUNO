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
    temp = 0
    frameName = 'Undistorted Frame'
    print('Calculating Distortion')
    yu, xu = myJazz.distortionMap()
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yu,xu]
                temp = frame.copy()

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
            return temp
    else:
        print('Failed to open camera')

mouse_x, mouse_y = 0,0
def update_coordinates(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:  # Mouse is moving
        if x > 0 and y > 0:
            mouse_x, mouse_y = x, y

def calculatePoints(og: np.ndarray):
    points = [[0,0],[0,0],[0,0],[0,0]]
    global mouse_x, mouse_y
    ySize, xSize = og.shape[0], og.shape[1]
    windowWidth = 301
    windowHeight = 301
    frameName = 'Undistorted Frame'
    try:
        window_handle = cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(frameName, update_coordinates)
        while True:
            scaleY, scaleX = int((ySize/windowHeight)*mouse_y), int((xSize/windowWidth)*mouse_x)
            frame = og[scaleY:scaleY+windowHeight, scaleX:scaleX+windowWidth, :].copy()
            frame[150,:] = 255-frame[150,:]
            frame[:,150] = 255-frame[:,150]
            if cv2.getWindowProperty(frameName, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.imshow(frameName,frame)
            else:
                break
            key = cv2.waitKey(10) & 0xFF
            if key == ord('1'):
                points[0] = [scaleY+150,scaleX+150][:]
                print('top left')
            elif key == ord('2'):
                points[1] = [scaleY+150,scaleX+150][:]
                print('bottom left')
            elif key == ord('3'):
                points[2] = [scaleY+150,scaleX+150][:]
                print('bottom right')
            elif key == ord('4'):
                points[3] = [scaleY+150,scaleX+150][:]
                print('top right')
            elif key == ord('5'):
                for i in range(4):
                    points[i] = points[i][::-1]
                break
                
    finally:
        cv2.destroyAllWindows()
        return points
    
def finalFootage():
    yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')
    frameName = 'Final Board Frame'
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                print(frame.shape)
                if cv2.getWindowProperty(frameName, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(frameName,frame[::2, ::2])
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print('Failed to open camera')

if __name__ == '__main__':
    boardSize = (517*2, 605*2)
    print(myJazz.gstreamer_pipeline(flip_method=0))
    rawFootage()
    snap = unDistortedFootage()
    corners = calculatePoints(snap)
    yu,xu = myJazz.distortionMap()
    yw,xw = myJazz.unwarpMap(corners, *boardSize)
    yuw, xuw = myJazz.getFinalTransform(yw,xw,yu,xu)
    np.save('yMap.npy', yuw)
    np.save('xMap.npy', xuw)
    finalFootage()