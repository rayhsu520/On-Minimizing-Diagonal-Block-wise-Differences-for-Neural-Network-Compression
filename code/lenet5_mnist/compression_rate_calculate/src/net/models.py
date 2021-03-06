import torch
import torch.nn as nn
import torch.nn.functional as F

from net.prune import PruningModule, MaskedLinear, MaskedConv2d

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(28*28, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)


    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


        return x
class LeNet5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d
        self.conv1 = conv2d(1, 20, kernel_size=(5, 5))
        self.conv2 = conv2d(20, 50, kernel_size=(5, 5))
        self.fc1   = linear(50*4*4, 500)
        self.fc2   = linear(500, 10)
        #self.fc3   = linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = out.view(out.size()[0], -1)
        out = F.relu(self.fc1(out))
        out =  F.log_softmax(self.fc2(out), dim=1)
        #out = F.log_softmax(self.fc3(out), dim=1)
        return out
'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
'''
class VGG(PruningModule):
    def __init__(self, vgg_name, mask=False):
        super(VGG, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d
        #self.features = self._make_layers(cfg[vgg_name])
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
        self.fc1 = linear(512, 4096)
        self.fc2 = linear(4096, 4096)
        self.fc3 = linear(4096, 10)
        #self.classifier = linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = self.conv4_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = self.conv5(out)
        out = self.conv5_bn(out)
        out = F.relu(out)
        out = self.conv6(out)
        out = self.conv6_bn(out)
        out = F.relu(out)
        out = self.conv7(out)
        out = self.conv7_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = self.conv8(out)
        out = self.conv8_bn(out)
        out = F.relu(out)
        out = self.conv9(out)
        out = self.conv9_bn(out)
        out = F.relu(out)
        out = self.conv10(out)
        out = self.conv10_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2), stride=2)

        out = self.conv11(out)
        out = self.conv11_bn(out)
        out = F.relu(out)
        out = self.conv12(out)
        out = self.conv12_bn(out)
        out = F.relu(out)
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
