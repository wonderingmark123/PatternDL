# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# [AMS - 200601] added lenet for comparison
# [AMS - 200601] added BaseCNN for comparison
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class CONVPatternNetBASE(nn.Module):
    def __init__(self, Number_Pattern ,kernel_size = 6, in_channels = 1 ):
        super(CONVPatternNetBASE , self).__init__()
        numREMOVE = int(kernel_size/2)
        if( kernel_size %2==0):
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE-1, numREMOVE, numREMOVE-1))
            
        else:
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE, numREMOVE, numREMOVE))
        self.conv1 = nn.Conv2d(in_channels= in_channels,kernel_size= kernel_size, out_channels= Number_Pattern )
        self.bn1 = nn.BatchNorm2d(num_features= Number_Pattern)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        """
            :param x: Input data of shape 
            [batch_size, in_channels, [imsize]]

            :return: Output data of shape
            [batch_size, Number_Pattern, [imsize]]
        """
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

class CONVPatternNetMoreLayer(nn.Module):
    def __init__(self, Number_Pattern, Hidden ,kernel_size = 10, in_channels = 1 ,numMove = None):
        super(CONVPatternNetMoreLayer , self).__init__()
        numREMOVE = math.floor(kernel_size) 
        if( kernel_size %2==0):
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE, numREMOVE, numREMOVE))
        else:
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE, numREMOVE, numREMOVE))
        if not (numMove is None):
            self.pad = nn.ReflectionPad2d(padding=numMove)
        self.conv1 = nn.Conv2d(in_channels= in_channels,kernel_size= kernel_size, out_channels= Hidden )
        self.bn1 = nn.BatchNorm2d(num_features= Hidden)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels= Hidden,kernel_size= kernel_size, out_channels= Number_Pattern )
        self.bn2 = nn.BatchNorm2d(num_features= Number_Pattern)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        """
            :param x: Input data of shape 
            [batch_size, in_channels, [imsize]]

            :return: Output data of shape
            [batch_size, Number_Pattern, [imsize]]
        """
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class CONVNetFC(nn.Module):
    def __init__(self, Number_Pattern:int, Hidden:int,BatchSize:int,device ,kernel_size = 6, in_channels = 1, LinearNum: int= 1 ):
        super(CONVNetFC , self).__init__()
        numREMOVE = math.floor(kernel_size) -1 
        if( kernel_size %2==0):
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE, numREMOVE, numREMOVE))
        else:
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE, numREMOVE, numREMOVE))
        self.conv1 = nn.Conv2d(in_channels= in_channels,kernel_size= kernel_size, out_channels= Hidden )
        self.bn1 = nn.BatchNorm2d(num_features= Hidden)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels= Hidden,kernel_size= kernel_size, out_channels= Number_Pattern )
        self.bn2 = nn.BatchNorm2d(num_features= Number_Pattern)
        self.relu = nn.ReLU()
        self.FC = nn.Linear(LinearNum,Number_Pattern)
        self.zero0 = torch.zeros([BatchSize,1],device=device)
    
    def forward(self,x):
        """
            :param x: Input data of shape 
            [batch_size, in_channels, [imsize]]

            :return: Output data of shape
            [batch_size, Number_Pattern, [imsize]]
        """
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        b = self.FC(self.zero0)
        x = x*b.view([*b.shape,1,1])
        return x


class CONVPatternNet3kernel(nn.Module):
    def __init__(self, Number_Pattern ,kernel_size = 3, in_channels = 1):
        """
            Layers must bigger than 3, otherwise CONVPatternNetMoreLayer class is recommanded.
        """
        super(CONVPatternNet3kernel , self).__init__()
        numREMOVE = int(kernel_size*2) - 1
        if( kernel_size %2==0):
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE, numREMOVE, numREMOVE))
        else:
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE, numREMOVE, numREMOVE))
        self.relu = nn.ReLU()
        self.bnMore,self.convMore = [],[]

        self.conv1 = nn.Conv2d(in_channels= in_channels,kernel_size= kernel_size, out_channels= int(Number_Pattern/4) )
        self.bn1 = nn.BatchNorm2d(num_features= int(Number_Pattern/4) )
        
        self.conv2 = nn.Conv2d(in_channels= int(Number_Pattern/4) ,kernel_size= kernel_size, out_channels= int(Number_Pattern/2)  )
        self.bn2 = nn.BatchNorm2d(num_features= int(Number_Pattern/2))

        self.conv3 = nn.Conv2d(in_channels= int(Number_Pattern/2) ,kernel_size= kernel_size, out_channels= int(Number_Pattern)  )
        self.bn3 = nn.BatchNorm2d(num_features= int(Number_Pattern))
        # for i in range(0,Layers - 3):
        #     self.convMore.append(nn.Conv2d(in_channels= Number_Pattern,kernel_size= kernel_size, out_channels= Number_Pattern ))
        #     self.bnMore.append(nn.BatchNorm2d(num_features= Number_Pattern ))
        self.conv4 = nn.Conv2d(in_channels= int(Number_Pattern) ,kernel_size= kernel_size, out_channels= int(Number_Pattern)  )
        self.bn4 = nn.BatchNorm2d(num_features= int(Number_Pattern))
        self.drop = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(in_channels= int(Number_Pattern) ,kernel_size= kernel_size, out_channels= int(Number_Pattern)  )
        self.bn5 = nn.BatchNorm2d(num_features= int(Number_Pattern))
        # self.drop = nn.Dropout(0.5)
    def forward(self,x):
        """
            :param x: Input data of shape 
            [batch_size, in_channels, [imsize]]

            :return: Output data of shape
            [batch_size, Number_Pattern, [imsize]]
        """
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        # for i in range(0,self.layers ):
        #     x = self.convMore[i](x)
        #     x = self.bnMore[i](x)
        #     x = self.relu(x)

        return self.drop(x)
class CONVPatternNetOnekernel(nn.Module):
    def __init__(self, Number_Pattern ,kernel_size = 3,NumLayers = 5):
        """
            Only one kernel is applied in each layer
        """
        super(CONVPatternNetOnekernel , self).__init__()
        numREMOVE = int((kernel_size - 1)*NumLayers/2)
        if( numREMOVE %2==0):
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE+1, numREMOVE, numREMOVE+1, numREMOVE))
        else:
            self.pad = nn.ReflectionPad2d(padding=(numREMOVE, numREMOVE, numREMOVE, numREMOVE))
        self.relu = nn.ReLU()
        layers = (nn.Conv2d(in_channels= Number_Pattern ,kernel_size= kernel_size, out_channels= Number_Pattern,groups=Number_Pattern ),
            nn.BatchNorm2d(num_features= Number_Pattern ),
            self.relu)*NumLayers
        # self.layers = nn.Sequential(
        #     nn.Conv2d(in_channels= Number_Pattern ,kernel_size= kernel_size, out_channels= Number_Pattern ),
        #     nn.BatchNorm2d(num_features= Number_Pattern ),
        #     nn.Conv2d(in_channels= Number_Pattern ,kernel_size= kernel_size, out_channels= Number_Pattern ),
        #     nn.BatchNorm2d(num_features= Number_Pattern ),
        #     nn.Conv2d(in_channels= Number_Pattern ,kernel_size= kernel_size, out_channels= Number_Pattern ),
        #     nn.BatchNorm2d(num_features= Number_Pattern ),
        #     nn.Conv2d(in_channels= Number_Pattern ,kernel_size= kernel_size, out_channels= Number_Pattern ),
        #     nn.BatchNorm2d(num_features= Number_Pattern ),
        # )
        self.layers = nn.Sequential(*layers)
        self.Channel = Number_Pattern
        # self.drop = nn.Dropout(0.5)
    def forward(self,x):
        """
            :param x: Input data of shape 
            [batch_size, in_channels, [imsize]]

            :return: Output data of shape
            [batch_size, Number_Pattern, [imsize]]
        """
        # x = self.pad(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)

        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        # for i in range(0,self.layers ):
        #     x = self.convMore[i](x)
        #     x = self.bnMore[i](x)
        #     x = self.relu(x)
        # z = torch.ones_like(x)
        x = self.pad(x)
        
        # for i in range(0,self.Channel):
        #     z[:,i:i+1,:,:] = self.layers(x[:,i:i+1,:,:])
        # x = self.pad(x)
        x = self.layers(x)
        return x


class BCNN(nn.Module):
    def __init__(self, in_chan, params, kernel_size=3,imsize=50):
        super(BCNN, self).__init__()

        c1_targets, c2_targets, out_chan = params
        imsize4 = int(imsize/4)
        imsize8 = int(imsize4/2)
        imsize16 = int(imsize8/2)
        self.convblock1 = ConvBlock(in_channels=in_chan, hidden=32, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, hidden=128, out_channels=128)
        self.coarse1    = CoarseBlock(in_features=128*imsize4*imsize4, hidden=128, out_features=c1_targets)
        
        self.convblock3 = ConvBlock(in_channels=128, hidden=256, out_channels=256)
        self.coarse2    = CoarseBlock(in_features=256*imsize8*imsize8, hidden=1024, out_features=c2_targets)
        
        self.convblock4 = ConvBlock(in_channels=256, hidden=512, out_channels=512)
        self.coarse3    = CoarseBlock(in_features=512*imsize16*imsize16, hidden=1024, out_features=out_chan)
        # self.coarse1    = CoarseBlock(in_features=128*12*12, hidden=128, out_features=c1_targets)
        # self.coarse2    = CoarseBlock(in_features=256*6*6, hidden=1024, out_features=c2_targets)
        # self.coarse3    = CoarseBlock(in_features=512*3*3, hidden=1024, out_features=out_chan)


    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)

        l1 = x.view(x.size()[0], -1)
        c1, c1_pred = self.coarse1(l1)

        x = self.convblock3(x)

        l2 = x.view(x.size()[0], -1)
        c2, c2_pred = self.coarse2(l2)

        x = self.convblock4(x)

        l3 = x.view(x.size()[0], -1)
        f1, f1_pred = self.coarse3(l3)

        return c1, c2, f1

# -----------------------------------------------------------------------------

class BaseCNN(nn.Module):
    def __init__(self, in_chan, params, kernel_size=3):
        super(BaseCNN, self).__init__()

        c1_targets, c2_targets, out_chan = params

        self.convblock1 = ConvBlock(in_channels=in_chan, hidden=32, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, hidden=128, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, hidden=256, out_channels=256)
        self.convblock4 = ConvBlock(in_channels=256, hidden=512, out_channels=512)
        self.coarse3    = CoarseBlock(in_features=512*3*3, hidden=1024, out_features=out_chan)

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.size()[0], -1)
        f1, f1_pred = self.coarse3(x)

        return f1, f1_pred


# -----------------------------------------------------------------------------


class LeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5):
        super(LeNet, self).__init__()

        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        self.drop  = nn.Dropout(p=0.5)

        self.init_weights()

    def init_weights(self):
        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = m.in_features
                nn.init.uniform_(m.weight, -np.sqrt(3./y), np.sqrt(3./y))
                nn.init.constant_(m.bias, 0)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x
