from ast import Num
from typing import Pattern
from torch.utils.data import Dataset,dataloader,TensorDataset
from torch.autograd import Variable
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torchvision.datasets import MNIST
class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self,fileFolder="D:\\study\\PatternDL\\python\\data"
                    ,imsize=[54,98]):
        fileName = os.path.join(
        fileFolder,'training_input.npy')
        if not os.path.isfile(fileName):
            raise IOError(' This file doesn\'t exist {:} '.format(fileName))
        # loading data
        self.x_data = np.load(fileName)
        self.len = np.size(self.x_data,0)
        self.transform = trasnFcn(imsize)
        if len(imsize) == 1:
            imsize = [imsize,imsize]
     
        

    def __getitem__(self, index):
        img = Image.fromarray(self.x_data[index,0,:,:],mode='L')
        img = self.transform(img)
        return img

    def __len__(self):
        return self.len

def trasnFcn(imsize = [54,98],datamean = 0.5,
datastd = 0.5 ):
    # crop     = transforms.CenterCrop(imsize)
    resizeIMG  = transforms.Resize(size = imsize)
    # pad      = transforms.Pad((0, 0, 1, 1), fill=0)
    totensor   = transforms.ToTensor()
    # normalise  = transforms.Normalize(datamean , datastd )

    # transform = transforms.Compose([
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.Resize(50),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,))
    #         ])
    
    transform = transforms.Compose([
        resizeIMG,
        transforms.RandomRotation(360, resample=Image.BILINEAR, expand=False),
        totensor,
        # normalise,
    ])

    return transform
def ImportOriData(fileFolder="D:\\study\\PatternDL\\python\\data"
    ,imsize=[54,98]):
    fileName = os.path.join(
        fileFolder,'training_input.npy')
    if not os.path.isfile(fileName):
        raise IOError(' This file doesn\'t exist {:} '.format(fileName))
    # loading data
    training_input = np.load(fileName)

    return training_input
def imageShow(Image):
    plt.imshow(Image)
