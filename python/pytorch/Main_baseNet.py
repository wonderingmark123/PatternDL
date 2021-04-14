# -----------------Directory settings ------------------------------------------------
MNISTsaveFolder = 'D:\\study\\DLpattern\\PatternDL\\python\\data'
SaveModelFile = 'D:\study\DLpattern\PatternDL\python\data\Kaggle_Layers10_pink_beta0005_imsize112_kernel10_Oripattern_oneKernel'
PatternFileName= '../../PatternsTrained.npy'
LoadModelFile = SaveModelFile
# ----------------------------------------------------------------------------------------------
from ast import Num
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm.std import tqdm
from generateCGI import *
from ultils import *
from model import *
import torch.nn.functional as F
from tqdm import trange
#---------------------- parameters -----------------
batch_size    = 8                 # number of samples per mini-batch
num_works     = 3                   # setting in DataLoader Default: 0
Epochs        = 200                # total epochs for training process
torch.backends.cudnn.benchmark = True

saving_best   = True
Load_model    = False
TestMODE      = False

imsize        = [112]
beta          = 0.005                # sampling rate
Noise         = 0                   # ratio of noise for intensity
learning_rate = 5e-3
momentum      = torch.tensor(8e-1)  # momentum for optimizer
decay         = torch.tensor(1e-6)  # weight decay for regularisation

Layers        = 10
in_channels   = 0                   # 1 for grey, 3 for PIL, 0 for all the npy patterns are inputed
kernel_size   = 10                   # kenel_size for conv layers
ONEloss       = 'mean'                # reduce for loss function
random_seed   = 42
#--------------------------------------------------
def BasicSettings():
    global imsize ,batch_size,num_works,in_channels
    imsize = imsize*2 if (len(imsize)==1) else imsize
    if len(imsize) == 2:
        Number_Pattern =int( beta * imsize[0]*imsize[1])
    else:
        Number_Pattern =int( beta * imsize**2)
    if in_channels == 0:
        in_channels = Number_Pattern
    if TestMODE:

        print('testing mode now!')
        batch_size,num_works = 1,0
    print("num_works: {:}, batch_size: {:}, kernel size: {:}".format(num_works,batch_size,kernel_size))
    print('loading pattern: {:}'.format(PatternFileName))

    PatternOrigin = np.load(PatternFileName)
    PatternShape = np.shape(PatternOrigin)
    if len(PatternShape)==3:
        PatternOrigin = torch.from_numpy(PatternOrigin)[0:in_channels,0:imsize[0],0:imsize[1]] * torch.ones([batch_size,in_channels,imsize[0],imsize[1]])
    else:
        PatternOrigin = torch.from_numpy(PatternOrigin)[0:imsize[0],0:imsize[1]] * torch.ones([batch_size,in_channels,imsize[0],imsize[1]])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print("Device: ",device,' Pattern Number: {:d} Beta: {:f}'.format(int(Number_Pattern),beta))
    return device ,Number_Pattern,PatternOrigin

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
        'MINloss'       : MINloss
      }
    # shutil.copytree(os.path.abspath(__file__),SaveModelFile)
    torch.save(state,os.path.join(SaveModelFile,'Modelpara.pth'))

def main():
    
    device,Number_Pattern,PatternOrigin = BasicSettings()
    trainingLoader  = LoadData(MNISTsaveFolder,imsize=imsize , train = True,batch_size=batch_size,num_works=num_works)
    
    # model = CONVPatternNetBASE(Number_Pattern ,in_channels= in_channels,kernel_size= kernel_size)
    # model = CONVPatternNetMoreLayer(Number_Pattern,int(Number_Pattern/2) ,in_channels= in_channels,kernel_size= kernel_size)
    model = CONVNetFC(Number_Pattern,int(Number_Pattern/2),batch_size,device ,in_channels= in_channels,kernel_size= kernel_size)
    # model = CONVPatternNetOnekernel(Number_Pattern ,kernel_size= kernel_size,NumLayers = Layers)
    # model = CONVPatternNet3kernel(Number_Pattern ,in_channels= in_channels,kernel_size= kernel_size)
    MINloss ,epochNow = 1e5,0
    MINtestLoss = 1e5
    epochTrainingLoss,epochTestingLoss = [] ,[]

    # load model
    if Load_model or TestMODE:
        model,epochNow,epochTrainingLoss,MINloss = LoadModel(model,LoadModelFile)
    model,PatternOrigin = model.to(device),PatternOrigin.float().to(device)
    optimizer = torch.optim.SGD( model.parameters() , lr=learning_rate, momentum=momentum, weight_decay=decay)

    
    
    if TestMODE:
        # --------------------------------------------------------
        #                       testing process
        # --------------------------------------------------------
        print("last training losses are {:}, and the epoch of training is {:}".format(epochTrainingLoss[-3:],epochNow))
        plt.plot(range(len(epochTrainingLoss)),epochTrainingLoss)
        plt.show()
        with torch.no_grad():
            testingLoader   = LoadData(imsize=imsize , train = False,MNISTsaveFolder = MNISTsaveFolder,batch_size=batch_size,num_works=num_works)
            model.zero_grad()
            test_loss  = []
            for batchNum , (data, target) in enumerate(testingLoader):
                model.zero_grad()
                input_image = data.to(device)
                Patterns = model(PatternOrigin)
                stdInputImg = torch.std(input_image)
                if Noise > 0:
                    CGI_image = generateCGI_func_noise(input_image, Patterns, Number_Pattern,batch_size,stdInputImg,Noise)
                else:
                    CGI_image = generateCGI_func(input_image, Patterns, Number_Pattern,batch_size,stdInputImg)
                npySave('Patterns.npy',Patterns,SaveModelFile)
                npySave('PatternOrigin.npy',PatternOrigin,SaveModelFile)
                npySave('input_image.npy',input_image,SaveModelFile)
                
                loss = F.mse_loss(input_image,CGI_image,reduction=ONEloss)
                test_loss.append(loss.item())
                input_image = data.to(device)

                plt.subplot(2,1,1)
                plt.imshow(input_image.to('cpu').detach().numpy()[0,0,:,:])
                plt.subplot(2,1,2)
                plt.imshow(CGI_image.to('cpu').detach().numpy()[0,0,:,:])
                plt.show()
                showx = 3
                showy = 3
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
    for epoch in range(epochNow,Epochs):
        model.train()
        train_losses = []
        for batch , (input_image, target) in enumerate(tqdm(trainingLoader)):
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
        if epochTrainingLoss[-1] < MINloss and saving_best:
                MINloss = epochTrainingLoss[-1]
                print("Epoch: {:}, saving the model to {:}".format(epoch,SaveModelFile))
                SavingModel(model,optimizer,epoch,epochTrainingLoss,MINloss)
        
        
        print('Epoch: {:d}, Training Loss: {:}.'.format(epoch,epochTrainingLoss[epoch]))
    return 0



if __name__ == '__main__':
    main()
    
    