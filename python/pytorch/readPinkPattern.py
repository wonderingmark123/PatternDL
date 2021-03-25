from typing import Pattern
from numpy.lib.npyio import load
from scipy.io import loadmat
import numpy as np
import cv2
from tqdm import trange
num = 99
size = 256
img = cv2.imread('D:\study\DLpattern\PatternDL\python\data\PinkPatternOriginal/rednoise_120_1.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x = np.array(gray)/255
Patterns = np.zeros( [num,size,size])
Patterns[0,:,:] = x[0:size,0:size]
for i in trange(2,num):
    img = cv2.imread('D:\study\DLpattern\PatternDL\python\data\PinkPatternOriginal/rednoise_120_'+str(int(i))+'.bmp')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = np.array(gray)/255
    Patterns[i-1,:,:] = x[0:size,0:size]
np.save('PatternPink99',Patterns)

print(np.max(x))