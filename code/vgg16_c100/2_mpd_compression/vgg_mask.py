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

number_class=100

def inverse_permutation(p):
    s=torch.empty(p.size(),dtype = torch.long)
    index=0
    for i in p:
        s[i]=index
        index+=1
    return s

def mask(in_weight,out_weight, partition_size):

    seed=10

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


    tensor_mask=torch.from_numpy(real_binary_mask)
    tensor_mask=torch.index_select(tensor_mask, 0, row_permu)
    tensor_mask=torch.index_select(tensor_mask, 1, col_permu)

    return row,col,row_permu,col_permu,tensor_mask

class VGG16(PruningModule):
    def __init__(self, vgg_name, partitions, mask_flag=False):
        super(VGG16, self).__init__()
        conv2d = MaskedConv2d if mask_flag else nn.Conv2d
        self.partition_size = partitions

        def fc1_hook(grad):
            return grad * self.mask1.float().cuda()

        def fc2_hook(grad):
            return grad * self.mask2.float().cuda()

        def fc3_hook(grad):
            return grad * self.mask3.float().cuda()

        self.conv1 = conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.conv8 = conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.conv9 = conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv10 = conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv10_bn = nn.BatchNorm2d(512)
        self.conv11 = conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv12_bn = nn.BatchNorm2d(512)
        self.conv13 = conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv13_bn = nn.BatchNorm2d(512)

        self.fc1= nn.Linear(512*1*1,4096)
        self.block_row_size1,self.block_col_size1,self.rowp1,self.colp1,self.mask1= mask(512*1*1,4096, int(partitions['fc1']))
        self.invrow1=inverse_permutation(self.rowp1)
        self.invcol1=inverse_permutation(self.colp1)
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight*self.mask1.float())
        self.fc1.weight.register_hook(fc1_hook)
        
        self.fc2= nn.Linear(4096,4096)
        self.block_row_size2,self.block_col_size2,self.rowp2,self.colp2,self.mask2= mask(4096,4096, int(partitions['fc2']))
        self.invrow2=inverse_permutation(self.rowp2)
        self.invcol2=inverse_permutation(self.colp2)
        self.fc2.weight = torch.nn.Parameter(self.fc2.weight*self.mask2.float())
        self.fc2.weight.register_hook(fc2_hook)

        self.fc3= nn.Linear(4096, number_class)
        self.block_row_size3,self.block_col_size3,self.rowp3,self.colp3,self.mask3= mask(4096,number_class, int(partitions['fc3']))
        self.invrow3=inverse_permutation(self.rowp3)
        self.invcol3=inverse_permutation(self.colp3)
        self.fc3.weight = torch.nn.Parameter(self.fc3.weight*self.mask3.float())
        self.fc3.weight.register_hook(fc3_hook)

        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv4(out)
        out = self.conv4_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = self.conv5(out)
        out = self.conv5_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv6(out)
        out = self.conv6_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv7(out)
        out = self.conv7_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = self.conv8(out)
        out = self.conv8_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv9(out)
        out = self.conv9_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv10(out)
        out = self.conv10_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = self.conv11(out)
        out = self.conv11_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv12(out)
        out = self.conv12_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv13(out)
        out = self.conv13_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)

        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out

