import myOwnLibrary as myJazz
import cv2
import numpy as np
from numpy import asanyarray as ana
import robotCommands as rc
import MelMyBoy as audio
colours = {
    'r':'Red',
    'g':'Green',
    'y':'Yellow',
    'b':'Blue',
    'w':'Wild',
    'W':'Wild +4',
}
faceValues = {
    's':'Skip',
    'r':'Reverse',
    '+':'+2'
}
for i in range(10):
    faceValues[f'{i}'] = f'{i}'

def moveValid(playedCard, discardTopCard):
    if len(playedCard) == 1:
        return True
    elif playedCard[0] == discardTopCard[0]:
        return True
    elif playedCard[1] == discardTopCard[1]:
        return True
    return False

def getAction(playedCard):
    #does not include move validity check:
    if playedCard == 'w':
        return audio.classify()
    if playedCard[1] in ['r', 's']:
        return 'skip'
    if playedCard[1] == '+':
        return 'draw2'
    return 'normal'

def getPlayableCards(hand, discardTopCard):
    playableCards = []
    for card in hand[0]:
        if moveValid(card, discardTopCard):
            playableCards.append(card)
    for card in hand[1]:
        if moveValid(card, discardTopCard):
            playableCards.append(card)
    if len(playableCards) == 0:
        return 'draw'
    return playableCards

def getBestMove(playAbleCards):
    if 'w' in playAbleCards[0]:
        return playAbleCards[1]
    return playAbleCards[0]

def playCard(card, hand):
    cardIn = 0 if card in hand[0] else 1
    while hand[cardIn][0] != card:
        #Shuffle the deck till you find the card you are looking for
        print(f'shuffle from hand {cardIn} to hand {(cardIn+1)%2}')
        hand[(cardIn+1)%2].insert(0,hand[cardIn].pop(0))
        rc.shuffle(cardIn)

    print(f'The robot has played {card} from hand {cardIn}')
    rc.play(cardIn)
    return hand[cardIn].pop(0)


def drawCard(hand, drawDeck):
    if len(hand[0]) > len(hand[1]):
        print('The robot has drawn a card and placed it in deck 1!')
        hand[1].insert(0,drawDeck.pop(0))
        rc.drawRobot(1)
    else:
        print('The robot has drawn a card and placed it in deck 0!')
        hand[0].insert(0,drawDeck.pop(0))
        rc.drawRobot(0)