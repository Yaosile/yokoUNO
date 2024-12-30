import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import myOwnLibrary as myJazz
import sympy as sp

cardWidth = 55.84#mm
step = 10#mm
angle = 30#deg

steps = 0
angle = np.deg2rad(angle)
#we want the card to always be intercepting a 45º inclanation

x = np.arange(-100,100,step)
plt.plot(x,np.zeros_like(x))#Plotting a ground plane


y = np.tan(angle)*x#create first wall

# plt.plot(x,y)






# plt.gca().set_aspect('equal', adjustable='box') 
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.show()







# H = 0.01/np.cos(angle)
# y = np.tan(angle)*x
# delta = cardWidth-steps
# cardAngle = np.arccos((cardWidth**2 + delta**2 - H**2)/(2*cardWidth*delta))

# for i in range(10):
#     H = step/np.cos(angle)
#     y = np.tan(angle)*x
#     delta = cardWidth-steps
#     cardAngle = np.arccos((cardWidth**2 + delta**2 - H**2)/(2*cardWidth*delta))
#     angle += cardAngle
#     plt.plot(x,y)

#     steps += 0.01

# plt.show()