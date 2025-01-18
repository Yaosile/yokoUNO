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
    'b5',
    'y6',
    'r9',
    'g9',
    'b2',
    'y+',
    'r0',
    'y9',
    'gr',
    'b4',
    'y2',
    'r7',
    'g2',
    'b9',
    'y3',
    'r8',
    'g+',
    'b0',
    'ys',
    'r3',
    'g0',
    'b+',
    'y7',
    'r1',
    'g1',
    'b3',
    'y4',
    'r2',
    'g4',
    'b6',
    'y1',
    'r5',
    'g3',
    'b7',
    'y8',
    'rs',
    'g8',
    'yr'
]
hand0Deck = [
    'g7',
    'r4',
    'g6',
    'rr'
]
hand1Deck = [
    'bs',
    'y0',
    'br'
]
handDeck = [hand0Deck, hand1Deck]
discardDeck = [
    'r6'
]
playerDeck = [
    'gs',
    'r+',
    'y5',
    'b8',
    'b1',
    'g5'
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
            change = 0
            if cap.isOpened():
                try:
                    ret,frame = cap.read()
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
            turnState = 0

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