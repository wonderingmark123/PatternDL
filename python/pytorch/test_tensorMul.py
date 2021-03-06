import torch
import numpy as np
import torch.nn.functional as nF
a= np.array(range(0,24))
a = np.reshape(a,[2,3,4])

a = torch.from_numpy(a).float()
b = nF.normalize(a,p=1,dim=0)
print(a)
print(b)
# print(b.view(3,1,1))
# print(c)