import numpy as np
import torch
import os
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

from generateCGI import *
from ultils import *
from model import *
import torch.nn.functional as F
from torchvision.datasets import MNIST
#---------------------- parameters -----------------
batch_size    = 32                 # number of samples per mini-batch
Epochs        = 200                   # total epochs for training process
learning_rate = 5e-3
imsize        = [84]
beta          = 0.005                # sampling rate
momentum      = torch.tensor(8e-1)  # momentum for optimizer
decay         = torch.tensor(1e-6)  # weight decay for regularisation
num_works     = 6                   # setting in DataLoader Default: 0
random_seed   = 42
in_channels   = 1                   # 1 for grey, 3 for PIL
kernel_size   = 10                   # kenel_size for conv layers
ONEloss       = 'mean'                # reduce for loss function

saving_best   = True
Load_model    = False
MNISTsaveFolder = 'D:\\study\\PatternDL\\python\\data'
SaveModelFile = 'D:\\study\\PatternDL\\python\\data\\Net_Layers2_pink_beta003_kernel8x8'
PatternFileName= 'PatternPink.npy'
datamean      = 0.5
datastd       = 0.5
TestMODE      = False

#--------------------------------------------------

def LoadData(imsize=[54,98],train = True):
    # original image size is [28,28]
    # data_set = DealDataset(imsize=imsize)
    Trans    = trasnFcn(imsize,datamean = datamean, datastd = datastd)
    if train:
        data_set = MNIST(root=MNISTsaveFolder, train=True, transform=Trans , download=True)
    else:
        data_set = MNIST(root=MNISTsaveFolder, train=False, transform=Trans)
    
    dataLoader = DataLoader(dataset= data_set,batch_size=batch_size, shuffle = True, num_workers=num_works)
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
def BasicSettings():
    global imsize ,batch_size,num_works
    imsize = imsize*2 if (len(imsize)==1) else imsize
    if TestMODE:
        batch_size,num_works = 1,0
    PatternOrigin = torch.from_numpy(np.load(PatternFileName))[0:imsize[0],0:imsize[1]] * torch.ones([batch_size,1,imsize[0],imsize[1]])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    if len(imsize) == 2:
        Number_Pattern =int( beta * imsize[0]*imsize[1])
    else:
        Number_Pattern =int( beta * imsize**2)

    print("Device: ",device,' Pattern Number: {:d} Beta: {:f}'.format(int(Number_Pattern),beta))
    return device ,Number_Pattern,PatternOrigin,torch.tensor(Number_Pattern).float()

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
    npySave('Intensity.npy',I)
    npySave('Pmean.npy',Pmean)
    npySave('PI.npy',PI)
    npySave('CGI_img.npy',CGI_img)
    CGI_img = (CGI_img - torch.mean(CGI_img) + 0.5)/torch.std(CGI_img)* stdInputImg
    
    return CGI_img
def SavingModel(model,optimizer,epoch,TrainingLosses,MINloss):
    if not os.path.isdir(SaveModelFile):
        os.mkdir(SaveModelFile)
    state = {'net':model.state_dict(), 
        'optimizer':optimizer.state_dict(),
        'epoch':epoch,
        'TrainingLosses': TrainingLosses,
        'batch_size'    : batch_size   ,              # number of samples per mini-batch,
        'Epochs'        : Epochs ,                  # total epochs for training process
        'learning_rate' : learning_rate,
        'imsize'        : imsize,
        'beta'          : beta ,               # sampling rate
        'momentum'      : momentum.numpy  ,# momentum for optimizer
        'decay'         : decay.numpy  ,# weight decay for regularisation
        'num_works'     : num_works                   ,# setting in DataLoader Default: 0
        'random_seed'   : random_seed,
        'in_channels'   : in_channels,                 # 1 for grey, 3 for PIL
        'kernel_size'   : kernel_size, # kenel_size for conv layers
        'ONEloss'       : ONEloss   ,           
        'saving_best'   : saving_best,
        'Load_model'    : Load_model,
        'MNISTsaveFolder' : MNISTsaveFolder,
        'SaveModelFile' : SaveModelFile,
        'datamean'      : datamean,
        'datastd'       : datastd,
        'MINloss'       : MINloss
      }
    # shutil.copytree(os.path.abspath(__file__),SaveModelFile)
    torch.save(state,os.path.join(SaveModelFile,'Modelpara.pth'))
def LoadModel(model):
    state  = torch.load(os.path.join(SaveModelFile,'Modelpara.pth'))
    model.load_state_dict(state['net'])
    epoch               = state['epoch']
    epochTrainingLoss   = state['TrainingLosses']
    MINloss             = state['MINloss']
    return model,epoch,epochTrainingLoss,MINloss
def npySave(FileName,tensor):
    np.save(os.path.join(SaveModelFile,FileName),tensor.to('cpu').detach().numpy())

def main():
    
    device,Number_Pattern,PatternOrigin,NumPatternTensor = BasicSettings()
    trainingLoader  = LoadData(imsize=imsize , train = True)
    
    # model = CONVPatternNetBASE(Number_Pattern ,in_channels= in_channels,kernel_size= kernel_size)
    model = CONVPatternNetMoreLayer(Number_Pattern,int(Number_Pattern/2) ,in_channels= in_channels,kernel_size= kernel_size)
    # model = CONVPatternNet3kernel(Number_Pattern ,in_channels= in_channels,kernel_size= kernel_size)
    MINloss = 1e5
    MINtestLoss = 1e5
    epochTrainingLoss,epochTestingLoss = [] ,[]

    # load model
    if Load_model or TestMODE:
        model,epoch,epochTrainingLoss,MINloss = LoadModel(model)
    model,PatternOrigin = model.to(device),PatternOrigin.float().to(device)
    optimizer = torch.optim.SGD( model.parameters() , lr=learning_rate, momentum=momentum, weight_decay=decay)

    
    
    if TestMODE:
        # --------------------------------------------------------
        #                       testing process
        # --------------------------------------------------------
        plt.plot(range(len(epochTrainingLoss)),epochTrainingLoss)
        plt.show()
        with torch.no_grad():
            testingLoader   = LoadData(imsize=imsize , train = False)
            model.zero_grad()
            test_loss  = []
            for batchNum , (data, target) in enumerate(testingLoader):
                model.zero_grad()
                input_image = data.to(device)
                Patterns = model(PatternOrigin)
                stdInputImg = torch.std(input_image)
                CGI_image = generateCGI_func(input_image, Patterns, Number_Pattern,batch_size,stdInputImg)
                npySave('Patterns.npy',Patterns)
                npySave('PatternOrigin.npy',PatternOrigin)
                npySave('input_image.npy',input_image)
                
                loss = F.mse_loss(input_image,CGI_image,reduction=ONEloss)
                test_loss.append(loss.item())
                input_image = data.to(device)

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
            epochTestingLoss.append(np.mean(test_loss))
            return 0
    # --------------------------------------------------------
    #                       traning process
    # --------------------------------------------------------
    
    print('Start training process :)')
    for epoch in range(Epochs):
        model.train()
        train_losses = []
        for batch , (input_image, target) in enumerate(trainingLoader):
            model.zero_grad()
            input_image     = input_image.to(device)
            Patterns        = model(PatternOrigin)
            stdInputImg     = torch.std(input_image)
            CGI_image       = generateCGI_func(input_image, Patterns, Number_Pattern,batch_size,stdInputImg)
            loss            = F.mse_loss(input_image,CGI_image,reduction=ONEloss)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        epochTrainingLoss.append(np.mean(train_losses))
        if epochTrainingLoss[-1] < MINloss and saving_best and batch > 1000:
                MINloss = epochTrainingLoss[-1]
                print("Epoch: {:}, saving the model to {:}".format(epoch,SaveModelFile))
                SavingModel(model,optimizer,epoch,epochTrainingLoss,MINloss)
        
        
        print('Epoch: {:d}, Training Loss: {:}.'.format(epoch,epochTrainingLoss[epoch]))
    return 0



if __name__ == '__main__':
    main()
    
    