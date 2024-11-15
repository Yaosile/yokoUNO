import myOwnLibrary as myJazz
import cv2
import numpy as np
from numpy import asanyarray as ana
import robotCommands as rc

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

def canPlay(hand1, hand2, discard):
    pass

def getCardPlayed(frame):
    card = frame.copy()
    card[card.shape[0]//2:,:] = 0
    lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
    # Split LAB channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge channels and convert back to BGR color space
    limg = cv2.merge((cl, a, b))
    card = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    card = card - card.min()
    card = card/card.max()
    card = card * 255
    card, cx, cy = myJazz.isolateCard(card,card)
    card, col = myJazz.getCardColour(card)
    if col not in ['w', 'W']:
        return f'{col + myJazz.getCardValue(card)}'
    else:
        return col
    
def getMoveValid(discard,played):
    if len(played) == 1: #played a type of wild
        return True
    if played[0] == discard[0]:
        return True
    if played[1] == discard[1]:
        return True
    return False

def getMoveToPlay(hand1, hand2, discard):

    chosenCard = ''

    hand = hand1+hand2

    if 'W' in [i[0] for i in hand]: #wanna play wild +4 first
        chosenCard = 'W'
    elif 'w' in [i[0] for i in hand]: #wanna play wild
        chosenCard = 'w'
    elif discard[0] in [i[0] for i in hand]: #wanna play same colour
        chosenCard = np.random.choice([i for i in hand if i[0] == discard[0]])
    else:
        chosenCard = 'draw'
    return hand1, hand2, chosenCard

def makeMove(hand1, hand2, choice, drawLocation, frame):
    smallestHand = np.argmax([len(hand1), len(hand2)])
    if choice == 'draw':
        print(f'The robot has chosen to draw and added it to {smallestHand + 1}')
        return drawCard(hand1, hand2, drawLocation, frame, smallestHand)
    else:
        print(f'The robot is playing {choice}')
        return playCard(hand1, hand2, choice)
    


def drawCard(hand1, hand2, drawLocation, frame, handChoice, topCard):
    rc.moveTo('hand2')
    frame = frame.copy()
    frame = myJazz.drawCircle(frame, *drawLocation, inverted=True)
    # drawenCard = getCardPlayed(frame)
    drawenCard = topCard
    rc.drawCard()
    if handChoice == 0:
        hand1.append(drawenCard)
        rc.dropHand1()
    else:
        hand2.append(drawenCard)
        rc.dropHand2()
    return hand1, hand2

def playCard(hand1, hand2, choice):
    whatHand = 0
    if choice in hand2:
        whatHand = 1
    hands = [hand1, hand2]
    depth = hands[whatHand][::-1].index(choice)

    for i in range(depth):#shuffling cards around
        hands[1-whatHand].append(hands[whatHand].pop())

        if whatHand == 1:
            rc.pickUpHand2()
            rc.dropHand1()
        else:
            rc.pickUpHand1()
            rc.dropHand2()
    hands[whatHand].pop()

    if whatHand == 1:
        rc.pickUpHand2()
    else:
        rc.pickUpHand1()

    rc.playCard()

    for i in hands:
        print(hands)
    
    return hands[0], hands[1]