'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from prune import PruningModule, MaskedConv2d

number_class=10

def inverse_permutation(p):
    s=torch.empty(p.size(),dtype = torch.long)
    index=0
    for i in p:
        s[i]=index
        index+=1
    return s

def mask(in_weight,out_weight, partition_size,seed):
    #partition_size=8
    #seed=10

    row=out_weight
    col=in_weight

    np.random.seed(seed)
    row_temp= np.random.permutation(row) #{1,2,5,6,....}
    col_temp= np.random.permutation(col)

    row_permu= torch.from_numpy(row_temp).long()
    col_permu= torch.from_numpy(col_temp).long()

    row=row//partition_size
    col=col//partition_size
    a=np.full((row, col),1,dtype= int)
    binary_mask=np.kron(np.eye(partition_size),a)
    

    real_binary_mask=np.pad(binary_mask,((0,out_weight%partition_size),(0,in_weight%partition_size)),'constant', constant_values=(0,0))# to make it able to divide

    return row,col,row_permu,col_permu,torch.from_numpy(real_binary_mask)



class LeNet5_mask(PruningModule):
    def __init__(self, LeNet5_maskname, partitions,seed, mask_flag=False):
        super(LeNet5_mask, self).__init__()
        conv2d = MaskedConv2d if mask_flag else nn.Conv2d
        self.partition_size = partitions

        def fc1_hook(grad):
            return grad * self.mask1.float().cuda()

        def fc2_hook(grad):
            return grad * self.mask2.float().cuda()

        self.conv1 = conv2d(1, 20, kernel_size=(5, 5))
        self.conv2 = conv2d(20, 50, kernel_size=(5, 5))

        self.fc1= nn.Linear(50*4*4,500)
        self.block_row_size1,self.block_col_size1,self.rowp1,self.colp1,self.mask1= mask(50*4*4,500, int(partitions['fc1']),seed)
        self.invrow1=inverse_permutation(self.rowp1)
        self.invcol1=inverse_permutation(self.colp1)
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight*self.mask1.float())
        self.fc1.weight.register_hook(fc1_hook)
        
        self.fc2= nn.Linear(500,10)
        self.block_row_size2,self.block_col_size2,self.rowp2,self.colp2,self.mask2= mask(500,10, int(partitions['fc2']),seed)
        self.invrow2=inverse_permutation(self.rowp2)
        self.invcol2=inverse_permutation(self.colp2)
        self.fc2.weight = torch.nn.Parameter(self.fc2.weight*self.mask2.float())
        self.fc2.weight.register_hook(fc2_hook)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = out.view(out.size()[0], -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out