import torch
import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear, MaskedConv2d

class LeNet5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d
        self.conv1 = conv2d(1, 20, kernel_size=(5, 5))
        self.conv2 = conv2d(20, 50, kernel_size=(5, 5))
        self.fc1   = linear(50*4*4, 500)
        self.fc2   = linear(500, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = out.view(out.size()[0], -1)
        out = F.relu(self.fc1(out))
        out =  F.log_softmax(self.fc2(out), dim=1)

        return out
