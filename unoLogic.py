import myOwnLibrary as myJazz
import cv2
import numpy as np
from numpy import asanyarray as ana

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
    