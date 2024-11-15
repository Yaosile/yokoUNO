import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2
import time
import robotCommands as rc
import unoLogic as logic
import MelMyBoy as audio
import serial

from scipy import signal

predefinedLocations = {
    'draw': 's210000 60000',
    'hand1': 's240000 100000', #TBD
    'hand2': 's220000 110000', #TBD
    'discard': '185000 85000', #TBD
    'playerDeal': 's200000 120000', #TBD
    'zd': 'zd',
    'zs': 'zs',
    'zu': 'zu',
}

hand1 = []
hand2 = []
deck = 101-3
discard = 0
topCard = 2.416013
bottomCard = 7.867078
travelTime = 5
drawPile = ['g1','y0','gr','g5','W','gs','b7','b5','b7','r+']

blur = np.ones((5,5))
blur = blur/blur.sum()
def determineLocations():
    rc.init()
    while True:
        location = input('please enter a location to calibrate: ')
        if location == 'q':
            break
        if location in rc.commands:
            ser = serial.Serial(rc.serialPort, 115200)
            ser.write(rc.commands[location].encode())
            ser.close()
def playUNO():
    hand1 = ['r+','y5','g4']
    hand2 = ['yr','w','g8','rs']
    discard = 'r6'
    card = 'W'
    cardBuffer = ['WW','WW','WW', 'WW']
    turn = -1 #0 for Human 1 For Robot
    robotThought = 0
    timeout = 0
    ready = False
    humanHand = 7
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
    rc.init()
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
                    if (np.abs(change-prevChange)) > 0.9:
                        movement = 20
                    if movement == 1:
                        movement = 0
                        turn = 0
                        print('Ready')
                    if movement > 0:
                        movement -= 1
                    prev = frame.copy()
                    prevChange = change


                elif turn == 0: #Humans turn
                    print('waiting on player move')
                    frame = myJazz.drawCircle(frame, *discardLocation, inverted=True)
                    change = (np.average(frame-prev))
                    if (np.abs(change-prevChange)) > 1:
                        print('detected Movement')
                        movement = 10
                    if movement == 1:
                        movement = 0
                        turn = 1
                        humanHand -= 1
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
                        # if len(set(cardBuffer)) == 1:
                        print(f'looks like a human played a {card}')
                        robotThought = 1
                        timeout = 0
                        if timeout > 30:
                            card = input('cannot determine what human played, please enter the card: ')
                            robotThought = 1
                            timeout = 0


                    elif robotThought == 1:
                        if logic.getMoveValid(discard, card):
                            discard = card
                            print('that is a valid move')
                            robotThought = 2
                            if card[1] in ['r', 's']:
                                turn = 0
                                robotThought = 0
                        else:
                            print('that move is invalid, please take it back')
                            robotThought = 0 #reset robot thought
                            turn = 0 #go back to Human turn
                            humanHand += 1

                    elif robotThought == 2:
                        if discard == 'w':
                            print('Please declare the colour')
                            declaredColour = audio.classify()
                            discard = declaredColour + '0'
                            hand1, hand2, cardToPlay = logic.getMoveToPlay(hand1, hand2, discard)
                            logic.makeMove(hand1, hand2, cardToPlay, drawLocation, frame)
                        elif discard == 'W':
                            print('please declare the colour')  
                            declaredColour = audio.classify()
                            for i in range(4):
                                topMostCard = drawPile.pop(0)
                                hand1, hand2 = logic.drawCard(hand1, hand2, drawLocation, frame, np.argmin([len(hand1), len(hand2)]), topMostCard)
                            robotThought = 0
                            turn = 0
                            rc.init()
                        elif discard[1] == '+':
                            for i in range(2):
                                topMostCard = drawPile.pop(0)
                                hand1, hand2 = logic.drawCard(hand1, hand2, drawLocation, frame, np.argmin([len(hand1), len(hand2)]), topMostCard)
                            robotThought = 0
                            turn = 0
                            rc.init()
                        else:
                            hand1, hand2, cardToPlay = logic.getMoveToPlay(hand1, hand2, discard)
                            logic.makeMove(hand1, hand2, cardToPlay, drawLocation, frame)
                            turn = 0
                            robotThought = 0
                            rc.init()

                







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
    determineLocations()
    playUNO()