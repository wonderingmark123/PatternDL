
import torch 
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
Order = ["First","Second","Third"]
File = "../data"

for i in range(3):
    SaveModelFile ="".join( [File,"/B05_Layers3_white_beta001_imsize112_kernel10_" , Order[i]])
    state  = torch.load(os.path.join(SaveModelFile,'Modelpara.pth'))
    epochTrainingLoss   = state['TrainingLosses']
    scipy.io.savemat("TrainingLoss.mat",{'TrainingLoss':epochTrainingLoss})
    plt.plot(epochTrainingLoss)
# plt.show()
