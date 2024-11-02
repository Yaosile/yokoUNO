import numpy as np
from numpy import asanyarray as ana
import feedForwardybackwards as cnn
from PIL import Image

possible = ['10.png', 'Checkers.png', '10R.png', 'CheckersR.png']
weights = cnn.generateWeights(32*32*3, 32*32*3, 2, 5)
print('weights done')

inputs = []
for i in possible:
    temp = ana(Image.open(f'Images/{i}'))[::8,::8,:].reshape(-1)
    inputs.append(temp)

inputs = ana(inputs)
correct = np.eye(2)
correct = np.vstack((correct, correct))
print(correct)

L2, weights = cnn.backProp(inputs, correct, weights, 0.01, 100, 0.01, 1)
print('training done')
for i, weight in enumerate(weights):
   np.save(f'Weights/{i}.npy', weight)
print(cnn.feedForward(inputs, weights))