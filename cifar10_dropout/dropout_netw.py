import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from utils.continuous_dropouts import *


class DropoutConvNetw(nn.Module):
    def __init__(self, inputs, p=0.5, filter_perc=1.0, dropout_method='gaussian', initial_drop_prob=0.2):
        super(DropoutConvNetw, self).__init__()
        self.initial_drop_prob = initial_drop_prob
        self.filter_percentage = filter_perc
        self.dropout_method = dropout_method
        self.p = p

        # 32x32x3
        N1 = int(96 * self.filter_percentage)
        print("Filter 1: %d" % N1)
        self.conv1 = self.conv(inputs, N1, kernel_size=3)
        self.conv2 = self.conv(N1, N1, kernel_size=3)
        self.conv3 = self.conv(N1, N1, kernel_size=3, stride=2)
        #drp1_layer = nn.Conv2d(N1, N1, kernel_size=3, stride=2)
        #torch.nn.init.xavier_uniform(drp1_layer.weight)
        #self.drp1 = dropout(p, dropout_method, layer=drp1_layer, prior='log-normal')

        # 16x16x96
        N2 = int(192 * self.filter_percentage)
        print("Filter 2: %d" % N2)
        self.conv4 = self.conv(N1, N2, kernel_size=3)
        self.conv5 = self.conv(N2, N2, kernel_size=3)
        self.conv6 = self.conv(N2, N2, kernel_size=3, stride=2)
        drp2_layer = nn.Conv2d(N2, N2, kernel_size=3, stride=2)
        torch.nn.init.xavier_uniform(drp2_layer.weight)
        self.drp2 = dropout(p, dropout_method, layer=drp2_layer, prior='log-normal')

        # 8x8x192
        self.conv7 = self.conv(N2, N2, kernel_size=3)
        self.conv8 = self.conv(N2, N2, kernel_size=1)
        self.conv9 = self.conv(N2, 10, kernel_size=1)

    def forward(self, x):
        #out = F.dropout(x, self.initial_drop_prob)
        out = x
        out = F.softplus(self.conv1(out))
        out = F.softplus(self.conv2(out))
        #if self.dropout_method == 'information':
        #    out = self.drp1(out, activation=F.softplus, network_layer=self.conv3)
        #else:
        #    out = self.drp1(F.softplus(self.conv3(out)))
        out = F.relu(self.conv3(out))
        out = F.softplus(self.conv4(out))
        out = F.softplus(self.conv5(out))
        if self.dropout_method == 'information':
            out = self.drp2(out, activation=F.softplus, network_layer=self.conv6)
        else:
            out = self.drp2(F.softplus(self.conv6(out)))
        #out = F.relu(self.conv6(out))
        out = F.softplus(self.conv7(out))
        out = F.softplus(self.conv8(out))
        out = F.softplus(self.conv9(out))
        out = torch.mean(out, [-2,-1])
        return out

    def kl(self):
        kl = 0
        for name, module in self.named_modules():
            if isinstance(module, InformationDropout):
                kl += module.kl()
        return torch.mean(kl, dim=0)

    def conv(self, in_planes, out_planes, kernel_size=3, stride=1):
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)

        # in the initial implementation padding='same', but in pytorch it is impossible, so I remove it
        block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=True),
            nn.BatchNorm2d(out_planes, momentum=0.9)
        )
        block.apply(init_weights)
        return block

    def saveName(self):
        return 'ConvNet_drp'


class DropoutNetw(nn.Module):
    def __init__(self, p, dropout_method='gaussian'):
        super(DropoutNetw, self).__init__()

        self.dropout_method = dropout_method
        self.p = p

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        drp1_layer = nn.Linear(120, 84)
        self.drp1 = dropout(p, dropout_method, layer=drp1_layer, prior='log-uniform')

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))

        if self.dropout_method == 'information':
            out = self.drp1(out, activation=F.relu, network_layer=self.fc2)
        else:
            out = self.drp1(F.relu(self.fc2(out)))

        out = self.fc3(out)
        return out

    def representation(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))

        if self.dropout_method == 'information':
            out = self.drp1(out, activation=F.relu, network_layer=self.fc2)
        else:
            out = self.drp1(F.relu(self.fc2(out)))

        out = out.view(out.size(0), -1)
        return out

    def kl(self):
        kl = 0
        for name, module in self.named_modules():
            if isinstance(module, InformationDropout):
                kl += module.kl()
        return torch.mean(kl, dim=0)

    def saveName(self):
        return 'LeNet_drp'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, p, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.drp = dropout(p, 'gaussian')
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear0 = nn.Linear(512*block.expansion, 128)
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.drp(self.linear0(out))
        out = self.linear(out)
        return out

    def representation(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear0(out)

        return out.view(out.size(0), -1)

    def saveName(self):
        return "ResNet"


def ResNet18(num_classes, p):
    return ResNet(BasicBlock, [2, 2, 2, 2], p, num_classes)

def create_ResNet_model(p):
    model = ResNet18(num_classes=10, p=p)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model