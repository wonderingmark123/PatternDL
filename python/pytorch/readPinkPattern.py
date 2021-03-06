from numpy.lib.npyio import load
from scipy.io import loadmat
import numpy as np
import cv2
img = cv2.imread('D:/study/PatternDL/python/examples/Noise patterns/rednoise_120_1.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x = np.array(gray)/255
np.save('PatternPink',x)

print(np.max(x))