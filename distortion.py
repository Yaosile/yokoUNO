import numpy as np
from numpy import asanyarray as ana
from PIL import Image
import myOwnLibrary as myLib
import time

img = Image.open('0.5515105986232713.png')
img = ana(img)

height,width = img.shape[0], img.shape[1]
dist = ana([-0.0639733628476694, -0.059022840140777, 0, 0, 0.0238818089164303])
mtx = ana([
    [1.734239392051136E3,0,1.667798059392088E3],
    [0,1.729637617052701E3,1.195682065165660E3],
    [0,0,1],
])
src = [[749,1246],[885,1774],[1682,1447],[1586, 1094]]
# dst = [[0,0], [0,250], [500,250], [500,0]]

yu,xu = myLib.distortionMap(dist, mtx, width, height)
yw,xw = myLib.unwarpMap(src, 500, 250, width, height)
yuw,xuw = myLib.getFinalTransform(yw,xw,yu,xu)

# now = time.time_ns()
output = img[yuw,xuw]
# print((time.time_ns()-now)/1e6)


output = Image.fromarray(output.astype(np.uint8))
output.show()