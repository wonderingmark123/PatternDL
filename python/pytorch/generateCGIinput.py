import os
from pathlib import Path
from numpy.core.numeric import zeros_like
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageOps
from IPython.core.debugger import set_trace
from numpy.core.shape_base import block
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from generateCGI import norm_mat,generateCGI_func
import generateCGI

#ref: https://pytorch.org/tutorials/beginner/nn_tutorial.html
# https://course.fast.ai/videos/?lesson=1 for network
class NetMine(torch.nn.Module):
    def __init__(self):
        super(NetMine , self).__init__()
        self.conv1 = torch.nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 1200)
        self.fc2 = nn.Linear(1200, 2400)
        self.fc3 = nn.Linear(2400, 5292)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1 , 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
def loss_function(input, target):
        return -input[range(target.shape[0]),target].mean()
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
def ChangeSize(image_ori):
    """
    change the size of images to [sizex , sizey]
    """
    sizex = 54
    sizey = 98
    image_ori = np.array(image_ori)
    nn = np.size(image_ori,0)
    image_changed = np.zeros([nn,sizex*sizey])
    for i in range(nn):
        img = image_ori[i].reshape(28,28)
        img = Image.fromarray(img)
        target_size = (sizey, sizex)
        new_image = img.resize(target_size)
        # new_image.
        new_image=np.array(new_image)
        image_changed[i] = new_image.reshape(1,sizex*sizey)
    return image_changed
        

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)
URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

sizex = 54
sizey = 98
beta  = 0.3
# if not (PATH / FILENAME).exists():
#         content = requests.get(URL + FILENAME).content
#         (PATH / FILENAME).open("wb").write(content)
# with gzip.open((PATH / FILENAME).as_posix() , "rb") as f:
#         ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
# x_train = ChangeSize(x_train)
# show the images resized
# pyplot.imshow(x_train[0].reshape((sizex, sizey)))
# pyplot.show()
# print(x_train.shape)

# x_train, y_train, x_valid, y_valid = map(
#     torch.tensor, (x_train, y_train, x_valid, y_valid))

# gpu is used for DL
device=torch.device("cuda:0")

# x_train = x_train.to(device)


# n = np.size(x_train,0)

# print(x_train.shape)
# print(y_train.min(), y_train.max())

# weights = torch.randn(784, 10,device=device) / math.sqrt(784)
# weights.requires_grad_()
# bias = torch.zeros(10, requires_grad=True,device= device)

# INPUT = np.zeros([n,1,sizex,sizey])
# OUTPUT = np.zeros([n,1,sizex,sizey])


# txt = np.load('./data/pattern_rednoise.npy')
# y_train = x_train *0
# for i in range(n):
#     img_ori = x_train[i].reshape(sizex,sizey)
#     y_train[i] = generateCGI.generateCGI_func(img_ori , txt , beta , sizex=54 , sizey=98).reshape(1,sizex*sizey)
#     plt.subplot(2,1,1)
#     pyplot.imshow(y_train[i].reshape((sizex, sizey)))
#     plt.subplot(2,1,2)
#     pyplot.imshow(x_train[i].reshape((sizex, sizey)))
#     pyplot.show()
    
# y_train = np.load("./data/pink_CGI.npy")

# for i in range(n):
#     INPUT[i] = x_train[i].reshape(sizex,sizey)
#     OUTPUT[i] = y_train[i].reshape(sizex,sizey)
INPUT = np.load("./data/training_input.npy")
OUTPUT = np.load("./data/training_output.npy")
n = 10000
for i in range(n):
    INPUT[i] = generateCGI.norm_mat(INPUT[i])
    OUTPUT[i] = generateCGI.norm_mat(OUTPUT[i])
np.save("./data/training_ori.npy",INPUT[0:n])
np.save("./data/training_cgi.npy",OUTPUT[0:n])