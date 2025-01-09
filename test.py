import numpy as np
from numpy import asanyarray as ana
import time

from scipy.io import wavfile
import scipy
import scipy.signal

from PIL import Image
# import pillow_heif

import MelMyBoy
import myOwnLibrary as myJazz
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import feedForwardybackwards as cnn
import os
import cv2
import unoLogic

import MelMyBoy as audio

template = ana(Image.open('templates/templateOG.png')).copy()
cut = ana(Image.open('templates/templateCut.png')).copy()
oneCardWidth = template.shape[1]//13
oneCardHeight = template.shape[0]

print(oneCardHeight/8, oneCardWidth/8)

# cards = []
# for i in range(13):
#     card = template[:,i*oneCardWidth:(i+1)*oneCardWidth] - cut
#     card[card < 255] = 0
#     cards.append(card)

# cards = np.hstack(cards)

# template = Image.fromarray(cards.astype(np.uint8))
# template.save('templates/template.png')


# # np.random.seed(0)

# # drawDeck = myJazz.deck.copy()
# # np.random.shuffle(drawDeck)

# # robotHand = [drawDeck.pop() for i in range(7)]
# # playerHand = [drawDeck.pop() for i in range(7)]
# # discard = [drawDeck.pop()]

# # print(unoLogic.getPlayableCards([playerHand,[]], discard[0]))


# # x,y = 0, 400
# # # x,y = myJazz.pixelToCartesian(x, y, 517, 605)

# # l1, r1 = myJazz.cartesianToScara(x,y)

# # l1,r1 = np.rad2deg(l1)+45, np.rad2deg(r1)+45
# # print()
# # print(f's{int(l1*1000)} {int(r1*1000)}')
# # print()



# # card = ana(Image.open('test.png'))[...,::-1]

# # lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
# # # Split LAB channels
# # l, a, b = cv2.split(lab)
# # # Apply CLAHE to the L channel
# # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# # cl = clahe.apply(l)
# # # Merge channels and convert back to BGR color space
# # limg = cv2.merge((cl, a, b))
# # card = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# # card = card - card.min()
# # card = card/card.max()
# # card = card * 255

# # card = myJazz.isolateCard(card)
# # card = myJazz.isolateValue(card)
# # guess = myJazz.compareTemplate(card)

# # print(myJazz.guessIndex[guess])

# # img = Image.fromarray(card.astype(np.uint8))
# # img.show()

# # mapX = np.load('xMap.npy')
# # mapY = np.load('yMap.npy')

# # xDiff = np.max(mapX) - np.min(mapX)
# # yDiff = np.max(mapY) - np.min(mapY)
# # print(mapX.shape, mapY.shape)
# # print(xDiff, yDiff)

# # dira = 'CardSnaps/'

# # cardCount = 13

# # guessIndex = ['+', *[f'{i}' for i in range(10)], 'r', 's']

# # template = ana(Image.open('templates/template.png'))

# # dimensions = (template.shape[1]//cardCount, template.shape[0])
# # print(dimensions)
# # cards = os.listdir(dira)

# # guesses = 0
# # correctness = 0
# # incorrect = []
# # for choice in cards:
# #     correct = choice[1]
# #     ref = ana(Image.open(dira + choice))
# #     ref = myJazz.scaleImage(ref,*dimensions)
# #     ref = myJazz.isolateValue(ref)
# #     guess = guessIndex[myJazz.compareTemplate(ref, template)]
# #     print(correct, guess)
# #     guesses += 1
# #     if correct == guess:
# #         correctness += 1
# #     else:
# #         incorrect.append(choice)

# # print(correctness, guesses, correctness/guesses)
# # print(incorrect)

# # # incorrect = ['b8.png', 'r3.png', 'r8.png', 'b4.png', 'b3.png', 'y8.png']
# # # choice = incorrect[2]
# # # choice = 'b0.png'
# # # ref = ana(Image.open(dira + choice))
# # # ref = myJazz.scaleImage(ref,*dimensions)

# # ref = myJazz.isolateValue(ref)

# # # print(myJazz.compareTemplate(ref, template))

# # img = Image.fromarray(ref.astype(np.uint8))
# # img.show()