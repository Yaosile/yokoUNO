import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2
from PIL import Image

import unoLogic as logic

from scipy import signal

yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')

drawLocation = 0
hand1Location = 0
hand2Location = 0
discardLocation = 0

hand1 = ['r+','y5','g4']
hand2 = ['yr','w','g8','rs']
discard = 'r6'

thresh = 50

blur = np.ones((5,5))
blur /= blur.sum()

def initilisation():
    global drawLocation
    global hand1Location
    global hand2Location
    global discardLocation
    
    global thresh

    global blur
    boardFrame = 'Board'
    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            cv2.namedWindow(boardFrame, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                if drawLocation != 0:
                    frame = myJazz.drawCircle(frame, *drawLocation)
                if hand1Location != 0:
                    frame = myJazz.drawCircle(frame, *hand1Location)
                if hand2Location != 0:
                    frame = myJazz.drawCircle(frame, *hand2Location)
                if discardLocation != 0:
                    frame = myJazz.drawCircle(frame, *discardLocation)


                if cv2.getWindowProperty(boardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(boardFrame,frame[::2,::2,:].astype(np.uint8))
                else:
                    break


                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    print('draw location')
                    output = frame.astype(float)
                    output = myJazz.rgb2hsv(output,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    for i in range(3): 
                        output = signal.fftconvolve(output, blur, mode='same')
                    output = myJazz.threshHold(output, 254)
                    t,b,l,right = myJazz.boundingBox(output)
                    drawLocation = myJazz.midPoint(t,b,l,right)
                    np.save('drawLoc.npy', ana(drawLocation))
                elif key == ord('2'):
                    print('hand1')

                    output = frame.astype(float)
                    output = myJazz.rgb2hsv(output,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    for i in range(3): 
                        output = signal.fftconvolve(output, blur, mode='same')
                    output = myJazz.threshHold(output, 254)
                    t,b,l,right = myJazz.boundingBox(output)
                    hand1Location = myJazz.midPoint(t,b,l,right)
                    np.save('h1Loc.npy', ana(hand1Location))
                elif key == ord('3'):
                    print('hand2')

                    output = frame.astype(float)
                    output = myJazz.rgb2hsv(output,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    for i in range(3): 
                        output = signal.fftconvolve(output, blur, mode='same')
                    output = myJazz.threshHold(output, 254)
                    t,b,l,right = myJazz.boundingBox(output)
                    hand2Location = myJazz.midPoint(t,b,l,right)
                    np.save('h2Loc.npy', ana(hand2Location))
                elif key == ord('4'):
                    print('discard location')

                    output = frame.astype(float)
                    output = myJazz.rgb2hsv(output,Calculations='SV')
                    output = (output[:,:,1])*output[:,:,2]*255
                    output = myJazz.threshHold(output, thresh)
                    for i in range(3): 
                        output = signal.fftconvolve(output, blur, mode='same')
                    output = myJazz.threshHold(output, 254)
                    t,b,l,right = myJazz.boundingBox(output)
                    discardLocation = myJazz.midPoint(t,b,l,right)
                    np.save('discardLoc.npy', ana(discardLocation))
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

def PlayUNO():
    cardBuffer = ['WW','WW','WW']
    turn = -1 #0 for Human 1 For Robot
    robotThought = 0
    timeout = 0
    ready = False
    drawLocation = np.load('drawLoc.npy')
    hand1Location = np.load('h1Loc.npy')
    hand2Location = np.load('h2Loc.npy')
    discardLocation = np.load('discardLoc.npy')
    boardFrame = 'Board'
    # cardFrame = 'Card'
    cx,cy = 0,0
    movement = 0

    prev = np.ones((1210,1034,3))
    prevChange = 0

    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')
    if video_capture.isOpened():
        try:
            cv2.namedWindow(boardFrame, cv2.WINDOW_AUTOSIZE)
            # cv2.namedWindow(cardFrame, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                if cv2.getWindowProperty(boardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(boardFrame,frame[::2,::2,:].astype(np.uint8))
                else:
                    break
                key = cv2.waitKey(1) & 0xFF
                if turn == -1: #Initial waiting period
                    print('waiting')
                    frame = myJazz.drawCircle(frame, *discardLocation, inverted=True)
                    change = (np.average(frame-prev))
                    if (np.abs(change-prevChange)) > 1:
                        movement = 30
                    if movement == 1:
                        movement = 0
                        turn = 0
                        print('Ready')
                    if movement > 0:
                        movement -= 1
                    prev = frame.copy()
                    prevChange = change
                elif turn == 0: #Humans turn
                    frame = myJazz.drawCircle(frame, *discardLocation, inverted=True)
                    change = (np.average(frame-prev))
                    if (np.abs(change-prevChange)) > 1:
                        movement = 10
                    if movement == 1:
                        movement = 0
                        turn = 1
                        print('Human played')
                    if movement > 0:
                        movement -= 1
                    prev = frame.copy()
                    prevChange = change

                    

                elif turn == 1: #Robots turn
                    if robotThought == 0:
                        frame = myJazz.drawCircle(frame, *discardLocation, inverted=True)
                        card = logic.getCardPlayed(frame)
                        cardBuffer.append(card)
                        cardBuffer.pop(0)
                        timeout += 1
                        print('.',end='')
                        if len(set(cardBuffer)) == 1:
                            print(f'looks like a human played a {card}')
                            robotThought = 1
                            discard = card
                            timeout = 0
                        if timeout > 30:
                            discard = input('cannot determine what human played, please enter the card: ')
                            robotThought = 1
                            timeout = 0


                    elif robotThought == 1:
                        print('Next Step')

                







                # if cv2.getWindowProperty(cardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                #     cv2.imshow(cardFrame,card.astype(np.uint8))
                # else:
                #     break
                if key == ord('q'):
                    break
                # elif key == ord(' '):
                #     card = frame.copy()
                #     card[card.shape[0]//2:,:] = 0
                #     lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
                #     # Split LAB channels
                #     l, a, b = cv2.split(lab)
                #     # Apply CLAHE to the L channel
                #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                #     cl = clahe.apply(l)
                #     # Merge channels and convert back to BGR color space
                #     limg = cv2.merge((cl, a, b))
                #     card = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                #     card = card - card.min()
                #     card = card/card.max()
                #     card = card * 255
                #     card, cx, cy = myJazz.isolateCard(card,card)
                #     card, col = myJazz.getCardColour(card)

                #     if col not in ['Wild', 'Wild+4']:
                #         print(col, myJazz.getCardValue(card))
                #     else:
                #         print(col)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    response = input('do initilisation: ')
    if response == 'y':
        initilisation()
    PlayUNO()