import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2
from PIL import Image
import feedForwardybackwards as cnn

lookUp = ['0','1','2','3','4','5','6','7','8','9','+','r','s']

def mainFootage():
    weights = []
    for i in range(2):
        weights.append(np.load(f'imageWeights/{i}.npy'))
    boardFrame = 'Board'
    cardFrame = 'Card'
    blur = np.ones((5,5))
    blur /= blur.sum()
    thresh = int(input('Enter a threshold value: '))
    output = np.ones((10,10,3))

    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
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

                elif key == ord(' '):
                    print('sn',end='')
                    output = frame.astype(float)
                    output = myJazz.rgb2hsv(output,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    output = myJazz.convolveMultiplication(output, blur)
                    output = myJazz.threshHold(output, 254)
                    output = myJazz.isolateCard(output, frame)
                    img = myJazz.scaleImage(output, 100, 100)
                    img = myJazz.rgb2hsv(img)
                    img = img[...,1]*img[...,2]*255
                    img = [cnn.convolutionalSection(myJazz.histogram_equalization(img))]
                    value = lookUp[np.argmax(cnn.feedForward(img, weights))]
                    print(value)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
if __name__ == '__main__':
    mainFootage()