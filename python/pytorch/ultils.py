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
from torch.utils.data.dataloader import DataLoader

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
    rotate     = transforms.RandomRotation(180)
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
        rotate,
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
def LoadModel(model,SaveModelFile):
    state  = torch.load(os.path.join(SaveModelFile,'Modelpara.pth'))
    model.load_state_dict(state['net'])
    epoch               = state['epoch']
    epochTrainingLoss   = state['TrainingLosses']
    MINloss             = state['MINloss']
    return model,epoch,epochTrainingLoss,MINloss
def npySave(FileName,tensor,SaveModelFile):
    np.save(os.path.join(SaveModelFile,FileName),tensor.to('cpu').detach().numpy())
def generateCGI_func_noise(img_ori, pattern , Number_Pattern , batch_size ,stdInputImg,Noise = 0):
    """
        generate ghost images from image_ori
        img_ori : Tensor [batch_size, in_channel, [imsize]]
        pattern : Tensor [batch_size, Number_Pattern, [imsize]]
        Number_Pattern : int 
        imsize : int or turple with length of 2
        CGIpic is normalized, and target is range from 0 to 255
        the CGIpic is normalized to mean value 0.25 0.2891
        Other variables in this function 
        I : intensity [bacth_size, Number_Pattern]
    """
    I = torch.sum( pattern * img_ori, (2,3))
    I = torch.rand_like(I) * 2 * Noise + I
    PI = torch.sum(
        I.view(batch_size,Number_Pattern,1,1)* pattern,
        1) / Number_Pattern
    Pmean = torch.sum(pattern,1)/Number_Pattern
    Imean = torch.sum(I,1)/Number_Pattern
    CGI_img = PI.view_as(img_ori) - Pmean.view_as(img_ori) * Imean.view([batch_size,1,1,1])
    # MAXCGIimg = torch.max(CGI_img,0)
    # CGI_img = CGI_img / torch.max(CGI_img,0)
    CGI_img = (CGI_img - torch.mean(CGI_img) + 0.5)/torch.std(CGI_img)* stdInputImg
    
    return CGI_img

    
def generateCGI_func(img_ori, pattern , Number_Pattern , batch_size ,stdInputImg):
    """
        generate ghost images from image_ori
        img_ori : Tensor [batch_size, in_channel, [imsize]]
        pattern : Tensor [batch_size, Number_Pattern, [imsize]]
        Number_Pattern : int 
        imsize : int or turple with length of 2
        CGIpic is normalized, and target is range from 0 to 255
        the CGIpic is normalized to mean value 0.25 0.2891

        Other variables in this function 
        I : intensity [bacth_size, Number_Pattern]
    """
    
    I = torch.sum( pattern * img_ori, (2,3))
    PI = torch.sum(
        I.view(batch_size,Number_Pattern,1,1)* pattern,
        1) / Number_Pattern
    Pmean = torch.sum(pattern,1)/Number_Pattern
    Imean = torch.sum(I,1)/Number_Pattern
    CGI_img = PI.view_as(img_ori) - Pmean.view_as(img_ori) * Imean.view([batch_size,1,1,1])
    # MAXCGIimg = torch.max(CGI_img,0)
    # CGI_img = CGI_img / torch.max(CGI_img,0)
    CGI_img = (CGI_img - torch.mean(CGI_img) + 0.5)/torch.std(CGI_img)* stdInputImg
    
    return CGI_img
def LoadData(MNISTsaveFolder,imsize=[54,98],train = True,batch_size=32,num_works=0,DataSetName = MNIST):
    # original image size is [28,28]
    # data_set = DealDataset(imsize=imsize)
    datamean      = 0.5
    datastd       = 0.5
    Trans    = trasnFcn(imsize,datamean = datamean, datastd = datastd)
    if train:
        data_set = DataSetName(root=MNISTsaveFolder, train=True, transform=Trans , download=False)
    else:
        data_set = DataSetName(root=MNISTsaveFolder, train=False, transform=Trans)
    
    dataLoader = DataLoader(dataset= data_set,batch_size=batch_size, shuffle = True, num_workers=num_works,drop_last=True)
    return dataLoader


def Training(trainingLoader,device,model):
    
    blursigma = (torch.tensor(1))
    optimizer = torch.optim.SGD( model.parameters() , lr=learning_rate, momentum=momentum, weight_decay=decay)
    for epoch in range(Epochs):
        for i , data in enumerate(trainingLoader):
            input_image = data.to(device)
            loss = blursigma * input_image
            loss.backwards()
            optimizer.step()
        print('Epoch: {:d}, Blursigma: {:}'.format(epoch,blursigma))
