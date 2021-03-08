from typing import Pattern
from numpy.lib.npyio import load
from scipy.io import loadmat
import numpy as np
import cv2
img = cv2.imread('D:/study/PatternDL/python/examples/Noise patterns/rednoise_120_1.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x = np.array(gray)/255
Patterns = np.zeros( [9,1000,1000])
Patterns[0,:,:] = x
for i in range(2,10):
    img = cv2.imread('D:/study/PatternDL/python/examples/Noise patterns/rednoise_120_'+str(int(i))+'.bmp')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    Patterns[i-1,:,:] = x
np.save('PatternP22ink9',Patterns)

print(np.max(x))