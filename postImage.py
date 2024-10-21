import numpy as np
from numpy import asanyarray as ana
from PIL import Image
import myOwnLibrary as myJazz

test = myJazz.rgb2gray(ana(Image.open('Images/Screenshot1.png')))

kernel = myJazz.gaussianKernelGenerator(7,1)
test = myJazz.convolveMultiplication(test, kernel)


test -= 127.5
test = myJazz.positive(test)
test = myJazz.threshHold(test, 10)
x,y = myJazz.onCardLines(test)
test[y,:] = 255
test[:,x] = 255



test = Image.fromarray(test.astype(np.uint8))
test.show()