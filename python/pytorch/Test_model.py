import numpy as np
import torch

from torch import nn
import matplotlib.pyplot as plt
from torch._C import device
from torch.autograd.variable import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision import transforms
from generateCGI import *
from ultils import *
from model import *
import torch.nn.functional as F
from torchvision.datasets import MNIST
#---------------------- parameters -----------------
batch_size    = 2                 # number of samples per mini-batch
Epochs        = 200                   # total epochs for training process
learning_rate = 1e-3
imsize        = [64]
beta          = 0.05                # sampling rate
momentum      = torch.tensor(8e-1)  # momentum for optimizer
decay         = torch.tensor(1e-6)  # weight decay for regularisation
num_works     = 0                   # setting in DataLoader Default: 0
random_seed   = 42
in_channels   = 1                   # 1 for grey, 3 for PIL
kernel_size   = 8                   # kenel_size for conv layers
ONEloss       = 'mean'                # reduce for loss function

saving_best   = True
NUM_savingBatch = 10
Load_model    = True
MNISTsaveFolder = 'D:\\study\\PatternDL\\python\\data'
SaveModelFile = 'D:\\study\\PatternDL\\python\\data\\model\\MoreNet_model_stat.pt'
datamean      = 0.5
datastd       = 0.5
#--------------------------------------------------

def LoadData(imsize=[54,98],train = True):

    data_set = DealDataset(imsize=imsize)
    
    Trans = trasnFcn(imsize,datamean = datamean, datastd = datastd)
    if train:
        data_set = MNIST(root=MNISTsaveFolder, train=True, transform=Trans , download=True)
    else:
        data_set = MNIST(root=MNISTsaveFolder, train=False, transform=Trans)
    if len(imsize) == 2:
        Number_Pattern = beta * imsize[0]*imsize[1]
    else:
        Number_Pattern = beta * imsize**2
    dataLoader = DataLoader(dataset= data_set,batch_size=batch_size, shuffle = True, num_workers=num_works)
    return dataLoader,int(Number_Pattern)

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
def DeviceChosen():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print("Device: ",device)
    return device  

def generateCGI_func(img_ori, pattern , Number_Pattern , batch_size ,stdInputImg):
    """
    generate ghost images from image_ori
    img_ori : Tensor [batch_size, in_channel, [imsize]]
    pattern : Tensor [batch_size, Number_Pattern, [imsize]]
    Number_Pattern : int 
    imsize : int or turple with length of 2
    CGIpic is normalized, and target is range from 0 to 255
    the CGIpic is normalized to [0,1]

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
    CGI_img = (CGI_img - torch.mean(CGI_img) )/torch.std(CGI_img)* stdInputImg
    return CGI_img

def main():
    global imsize
    imsize = imsize*2 if (len(imsize)==1) else imsize

    PatternWhite = torch.from_numpy(np.load('PatternWhite.npy'))[0:imsize[0],0:imsize[1]] * torch.ones([batch_size,1,imsize[0],imsize[1]])
    device = DeviceChosen()
    trainingLoader,Number_Pattern = LoadData(imsize=imsize , train = True)
    testingLoader,Number_Pattern = LoadData(imsize=imsize , train = False)
    # model = CONVPatternNetBASE(Number_Pattern ,in_channels= in_channels,kernel_size= kernel_size)
    model = CONVPatternNetMoreLayer(Number_Pattern,int(Number_Pattern/2) ,in_channels= in_channels,kernel_size= kernel_size)
    # load model
    if Load_model:
        model.load_state_dict(torch.load(SaveModelFile))
    model,PatternWhite = model.to(device),PatternWhite.float().to(device)
    optimizer = torch.optim.SGD( model.parameters() , lr=learning_rate, momentum=momentum, weight_decay=decay)

# -----------------------------------------------------------
#                       traning process
# -----------------------------------------------------------

    epochTrainingLoss,epochTestingLoss = [] ,[]
    print('Start training process :)')
    for epoch in range(Epochs):
        model.train()
        train_losses = []
        for batch , (data, target) in enumerate(testingLoader):
            model.zero_grad()
            input_image = data.to(device)
            Patterns = model(PatternWhite)
            stdInputImg = torch.std(input_image)
            CGI_image = generateCGI_func(input_image, Patterns, Number_Pattern,batch_size, stdInputImg)
            plt.subplot(2,1,1)
            plt.imshow(input_image.to('cpu').detach().numpy()[0,0,:,:])
            print(torch.max(input_image))
            print(torch.min(input_image))
            plt.subplot(2,1,2)
            plt.imshow(CGI_image.to('cpu').detach().numpy()[0,0,:,:])
            print(torch.max(CGI_image))
            print(torch.min(CGI_image))
            plt.show()
            showx = 5
            showy = 5
            for i in range(1,showx * showy+1):
                plt.subplot(showx,showy,i)
                plt.imshow(Patterns.to('cpu').detach().numpy()[0,i,:,:])
            plt.show()
        print('Epoch: {:d}, Training Loss: {:}.'.format(epoch,epochTrainingLoss[epoch]))




    return 0



if __name__ == '__main__':
    main()
    
    