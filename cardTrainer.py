import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import feedForwardybackwards as cnn
from PIL import Image

from matplotlib import pyplot as plt

import os

lookUp = ['0','1','2','3','4','5','6','7','8','9','+','r','s']
dira = 'HighResCards/'
inData = []
targetData =[]
files = os.listdir(dira)
# #No green Photos, didn't come out nice ig
# files = [i for i in files if i[0] == 'r']
# output in format of [0-9,+,r,s]
for choice in files:
    img = ana(Image.open(dira + choice))[6:41,10:45]
    img = img.sum(axis=2)/3
    img[img>254] = 1
    img[img<1] = 0
    img[img>1] = 0.5
    inData.append(cnn.convolutionalSection(img,kernelMax=2))
    output = [0]*len(lookUp)
    output[lookUp.index(choice[0])] = 1
    targetData.append(output)
# img = Image.fromarray(img.astype(np.uint8))
# img.show()

# for f in files:
#     output = [0]*len(lookUp)
#     output[lookUp.index(f[1])] = 1
#     targetData.append(output)
#     targetData.append(output)

#     img = ana(Image.open(f'CardSnaps/{f}'))[...,::-1]
#     img = myJazz.scaleImage(img, 100, 100)
#     img = myJazz.rgb2gray(img)
#     # img = myJazz.rgb2hsv(img)
#     # img = img[...,1]*img[...,2]*255
#     img = myJazz.histogram_equalization(img)

#     inData.append(cnn.convolutionalSection(img))
#     inData.append(cnn.convolutionalSection(myJazz.rot90(img,2)))

weights = cnn.generateWeights(len(inData[0]), int(len(inData[0])*1.15), len(targetData[0]), 3)


# weights = []
# for i in range(2):
#     weights.append(np.load(f'imageWeights/{i}.npy'))
L2, weights = cnn.backProp(inData, targetData, weights, 0.01, 100, 1)
for i in range(len(weights)):
    np.save(f'imageWeights/{i}.npy', weights[i])
plt.plot(L2)
plt.show()
# # img = ana(Image.open(f'CardSnaps/b0.png'))
# # img = myJazz.scaleImage(img, 100, 100)
# # img = myJazz.rgb2hsv(img)
# # img = img[...,1]*img[...,2]*255
# # img = myJazz.histogram_equalization(img)
# # img = [cnn.convolutionalSection(img)]

# # print(lookUp[np.argmax(cnn.feedForward(img, weights))])

# files = os.listdir('CardSnaps/')
# files = [i for i in files if i[0] == 'r']
# tot = len(files)
# testData = []
# correct = []
# for f in files:
#     img = ana(Image.open(f'CardSnaps/{f}'))[...,::-1]
#     img = myJazz.scaleImage(img, 100, 100)
#     img = myJazz.rgb2gray(img)
#     # img = myJazz.rgb2hsv(img)
#     # img = img[...,1]*img[...,2]*255
#     img = myJazz.histogram_equalization(img)

#     testData.append(cnn.convolutionalSection(img))

#     output = [0]*len(lookUp)
#     output[lookUp.index(f[1])] = 1
#     correct.append(output)

# guess = cnn.feedForward(testData, weights)
# guessSum = 0
# for i in range(len(guess)):
#     if np.argmax(guess[i]) == np.argmax(correct[i]):
#         guessSum += 1

# print(guessSum, tot, guessSum/tot)


# # ones = [i for i in files if i[1] == '1']

# # t = np.zeros((100,100))
# # ymax = 143
# # xmax = 93

# # for o in ones:
# #     img = ana(Image.open(f'CardSnaps/{o}'))
# #     img = myJazz.scaleImage(img, 100, 100)
# #     img = myJazz.rgb2hsv(img)
# #     img = img[...,1]*img[...,2]*255
# #     img = myJazz.histogram_equalization(img)
# #     t = np.hstack((t, img))

# # img = Image.fromarray(t.astype(np.uint8))

# # img.show()