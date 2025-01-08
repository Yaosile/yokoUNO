import myOwnLibrary as myJazz
import numpy as np

import unoLogic
import robotCommands

import MelMyBoy as audio
import time
import sys

drawDeck = [
    'r6',
    'y6',
    'b4',
    'y2',
    'b0',
    'r0',
    'b+',
    'r3',
    'b3',
    'g4',
    'b7',
    'b5',
    'b1',
    'g5',
    'r5',
    'r1',
    'y9',
    'g3',
    'y0',
    'r4',
    'y4',
    'r2',
    'g9',
    'g7',
    'gr',
    'g2',
    'ys',
    'rr',
    'g+',
    'b9',
    'g0',
    'y3',
    'b2',
    'b6',
    'rs',
    'yr',
    'bs',
    'g6'
]
hand0Deck = [
    'br',
    'r+',
    'g1',
    'r7'
]
hand1Deck = [
    'r9',
    'y7',
    'y8'
]
handDeck = [hand0Deck, hand1Deck]
discardDeck = [
    'g8'
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
            
            drawDeck.pop(0)
            robotCommands.drawPlayer()
            drawDeck.pop(0)
            robotCommands.drawPlayer()

            turnState = 1
            action = 'normal'
            continue

        elif action == 'normal':
            print(f'The player has to play a card of the same colour or value as {discardDeck[0]}!')


        playedCard = input("What card are you going to play?: ")
        if playedCard == 'draw':
            print('The player has chosen to draw a card!')
            print(drawDeck.pop(0))
            robotCommands.drawPlayer()

            turnState = 1
            action = 'normal'
            continue

        print(f'You played {playedCard}')
        if unoLogic.moveValid(playedCard, discardDeck[0]):
            #The player has played a valid move
            print('This move is valid!')
            discardDeck.insert(0, playedCard) # Add the played card to the discard deck
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