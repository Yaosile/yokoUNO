import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2
import feedForwardybackwards as cnn
from PIL import Image, ImageFilter
from scipy.signal import convolve2d as convolve
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error')
    exit()

kernelX = [
    [1,0,-1],
    [2,0,-2],
    [1,0,-1],
]
kernelY = [
    [1,2,1],
    [0,0,0],
    [-1,-2,-1],
]

edgeWidth = 1
blurKernel = np.ones((5,5))
blurKernel /= blurKernel.sum()

prev = None
output = None
background = None
trueBackground = None
movementBuffer = -1
movementValue = 0
change = 20
now = 0
guess = 0

possible = ['10.png', 'Checkers.png']
weights = []
for i in range(4):
    weights.append(list(np.load(f'Weights/{i}.npy')))

while True: #Initial camera verifications
    ret, frame = cap.read()
    if type(frame) != type(None):
        prev = frame
        output = frame
        background = frame
        trueBackground = frame
        break
while True:
    fps = str(int(1e9/(time.time_ns() - now)))
    now = time.time_ns()
    '''Main Loop'''
    ret, frame = cap.read()
    if type(frame) == type(None):
        continue
    # output = frame.astype(float)
    # output = myJazz.rgb2hsv(frame,Calculations='SV')
    # output = (output[:,:,1])*output[:,:,2]*255
    # output = Image.fromarray(output.astype(np.uint8))
    # output = output.filter(ImageFilter.BoxBlur(radius=3))
    # output = ana(output)

    # # frame = myJazz.rgb2hsv(frame, Calculations='SV')
    # # frame = frame[:,:,1]*frame[:,:,2]*255
    # # Gx,Gy = frame[:, edgeWidth:] - frame[:, :-edgeWidth], frame[edgeWidth:, :] - frame[:-edgeWidth, :]
    # # frame = ((Gx)[edgeWidth:,:]**2 +  (Gy)[:,edgeWidth:]**2)**0.5
    # # # Gx,Gy = convolve(frame, kernelX, mode='same'), convolve(frame, kernelY, mode='same')
    # # movement = frame-prev
    # # prev = frame
    

    # # movementValue = movement.max() - movement.min()
    # # if(movementValue > 350):
    # #     movementValue = 0
    # #     print('Movement')
    # #     movementBuffer = 0
    if movementBuffer > -1:
        movementBuffer -= 1
    if movementBuffer == 0:
        output = frame.astype(float)
        output = myJazz.rgb2hsv(output,Calculations='SV')
        output = output[:,:,1]*output[:,:,2]*255
        output = myJazz.convolveMultiplication(output, blurKernel)
        output = myJazz.threshHold(output, 75)
        output = myJazz.convolveMultiplication(output, blurKernel)
        output = myJazz.threshHold(output, 254)
        output = myJazz.isolateCard(output, frame)
        output = myJazz.scaleImage(output, 256,256)
        indata = ana([output[::8,::8,:].reshape(-1)])
        cv2.destroyWindow(f'{possible[guess]}')
        guess = np.argmax(cnn.feedForward(indata, weights))
        # if output.shape[0] > 0 and output.shape[1] > 0:
        #     output = frame
        # cx, cy = myJazz.midPoint(t,b,l,r)
        # r = myJazz.getRadius(output, cx, cy)
        # output = myJazz.getRotation(output, cx, cy, r)
        # print(theta)
        # output[cy,:] = 255
        # output[:,cx] = 255
        # output[t,:] = 255
        # output[b,:] = 255
        # output[:,l] = 255
        # output[:,r] = 255
        # Gx,Gy = output[:, edgeWidth:] - output[:, :-edgeWidth], output[edgeWidth:, :] - output[:-edgeWidth, :]
        # output = ((Gx)[edgeWidth:,:]**2 + (Gy)[:,edgeWidth:]**2)**0.5


        # c1,c2,c3,c4 = myJazz.boundingBox(output)
        # output = np.repeat(output[:,:,None], 3, axis=2)/3
        # output[c2] = [0,0,255]




        
    #     # output = output[:,:,1]*output[:,:,2]
    #     # # Gx,Gy = output[:, edgeWidth:] - output[:, :-edgeWidth], output[edgeWidth:, :] - output[:-edgeWidth, :]
    #     # # output = ((Gx)[edgeWidth:,:]**2 + (Gy)[:,edgeWidth:]**2)**0.5
    #     # # output = np.atan2((Gy)[:,edgeWidth:], (Gx)[edgeWidth:,:])*output
    #     # output = myJazz.threshHold(output, 100)
    #     # output = myJazz.threshHold(output, 10)
    #     # output = np.sum(output, axis=2)
    #     # output[output>255] = 255
    #     # output = myJazz.rgb2gray(output[:,:,::-1])
    #     # output = myJazz.convolveMultiplication(output, gaussiankernel)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(output, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
    cv2.imshow(f'{possible[guess]}', output.astype(np.uint8))
    cv2.imshow('Footage', frame.astype(np.uint8))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        '''ending logic'''
        break
    elif key == ord('w'):
        Image.fromarray(output.astype(np.uint8)).save('Images/Screenshot.png')
        print('Kachow')
    elif key == ord(' '):
        print('Space')
        movementBuffer = 1


# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()