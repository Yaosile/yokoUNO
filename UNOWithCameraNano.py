import myOwnLibrary as myJazz
import numpy as np

import cv2 

import unoLogic
import robotCommands

import MelMyBoy as audio
import time
import sys
from PIL import Image
cap = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0, framerate=1), cv2.CAP_GSTREAMER)
yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')

drawDeck = [
    'ys',
    'y+',
    'b0',
    'y1',
    'g0',
    'y3',
    'y2',
    'r8',
    'r5',
    'y7',
    'r7',
    'r0',
    'b+',
    'b6',
    'g+',
    'g4',
    'rs',
    'y4',
    'b4',
    'g1',
    'b7',
    'g3',
    'r+',
    'r3',
    'rr',
    'yr',
    'y0',
    'b3',
    'y9',
    'br',
    'g5',
    'g6',
    'bs',
    'r4',
    'b1',
    'g7',
    'b8',
    'gs'
]
hand0Deck = [
    'b2',
    'r9',
    'b9',
    'g8'
]
hand1Deck = [
    'r2',
    'g9',
    'y8'
]
handDeck = [hand0Deck, hand1Deck]
discardDeck = [
    'y6'
]
playerDeck = [
    'gr',
    'r6',
    'g2',
    'y5',
    'r1',
    'b5',
    'w'
]

#Global state reference, 0 is Human turn, 1 is Robot turn
turnState = 0
action = 'normal'

robotCommands.init()
print('You can turn on the robot now!')

#Creating a state machine
print("Let's play UNO!")
while True:
    if turnState == 0:
        #Human turn
        print()
        print("Your turn!")
        if action in ['r','y','g','b']:
            print(f'The colour has been changed to {unoLogic.colours[action]}')
            discardDeck[0] = f'{action}{discardDeck[0]}'

        elif action == 'skip':
            print('Your turn has been skipped!')

            turnState = 1
            action = 'normal'
            continue

        elif action == 'draw2':
            print('The player has to draw 2 cards and loses a turn!')
            
            playerDeck.append(drawDeck.pop(0))
            robotCommands.drawPlayer()
            playerDeck.append(drawDeck.pop(0))
            robotCommands.drawPlayer()

            turnState = 1
            action = 'normal'
            continue

        elif action == 'normal':
            print(f'The player has to play a card of the same colour or value as {discardDeck[0]}!')

        playedCard = unoLogic.getPlayableCards([playerDeck, []], discardDeck[0])
        if playedCard != 'draw':
            cardPlacedFlag = False
            prev = []
            if cap.isOpened():
                try:
                    time.sleep(3)
                    cv2.namedWindow('card', cv2.WINDOW_AUTOSIZE)
                    while True:
                        ret,frame = cap.read()
                        frame = frame[yuw,xuw]
                        trueCard = 0
                        if prev == []:
                            prev = frame
                        change = np.average(np.abs((prev.astype(float)-frame.astype(float))))
                        if change > 30:
                            cardPlacedFlag = True
                            print(f'card placed: {change}')
                        elif cardPlacedFlag:
                            cardPlacedFlag = False
                            guess, score, _ = myJazz.getCardValue(cap.read()[1][yuw,xuw])
                            trueCard = unoLogic.getMostLikelyPlayedCard(guess, score, playerDeck)
                            if trueCard == 0:
                                cardPlacedFlag = True
                        
                        if trueCard != 0:
                            playedCard = trueCard
                            break

                        cv2.imshow('card',frame)
                        prev = frame.copy()
                finally:
                    print('found the card played')
                    cap.release()
                    cv2.destroyAllWindows()
        if type(playedCard) == type([]):
            playedCard = playedCard[0]
        if playedCard == 'draw':
            print('The player has to draw a card!')
            print(drawDeck[0])
            playerDeck.append(drawDeck.pop(0))

            robotCommands.drawPlayer()

            turnState = 1
            action = 'normal'
            continue

        print(f'You played {playedCard}')
        if unoLogic.moveValid(playedCard, discardDeck[0]):
            #The player has played a valid move
            print('This move is valid!')
            discardDeck.insert(0, playedCard) # Add the played card to the discard deck
            playerDeck.remove(playedCard)
            action = unoLogic.getAction(playedCard)
            turnState = 1
        else:
            #The player has played an invalid move
            print('This move is not valid, please remove the card and try again!')

    elif turnState == 1:
        #Robot turn
        print()
        print('Robot turn!')
        if action in ['r','y','g','b']:
            print(f'The colour has been changed to {unoLogic.colours[action]}')
            discardDeck[0] = f'{action}{discardDeck[0]}'

        elif action == 'skip':
            print('The robot has been skipped!')
            
            turnState = 0
            action = 'normal'
            continue

        elif action == 'draw2':
            print('The robot has to draw 2 cards and loses a turn!')
            for i in range(2):
                unoLogic.drawCard(handDeck, drawDeck)
            turnState = 0
            action = 'normal'
            continue

        elif action == 'normal':
            print(f'The robot has to play a card of the same colour or value as {discardDeck[0]}!')
        
        playableCards = unoLogic.getPlayableCards(handDeck, discardDeck[0])
        #Now for robot response
        if playableCards == 'draw':
            print('The robot has to draw a card!')
            unoLogic.drawCard(handDeck, drawDeck)
            
            turnState = 0
            action = 'normal'
            continue
        else:
            print('The robot can play the following cards:')
            print(playableCards)
            bestMove = unoLogic.getBestMove(playableCards)
            playedCard = unoLogic.playCard(bestMove, handDeck)
            discardDeck.insert(0, playedCard)

            turnState = 0
            action = unoLogic.getAction(playedCard)
            continue