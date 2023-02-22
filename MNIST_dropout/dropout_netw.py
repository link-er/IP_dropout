import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../utils')
from utils_local.continuous_dropouts import *

class DropoutNetw(nn.Module):
    def __init__(self, p, dropout_method='information'):
        super(DropoutNetw, self).__init__()

        self.dropout_method = dropout_method
        self.p = p

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)

        drp1_layer = nn.Linear(120, 84)
        self.drp1 = dropout(p, dropout_method, layer=drp1_layer, prior='log-normal')

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.softplus(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.softplus(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.softplus(self.fc1(out))

        if self.dropout_method == 'information':
            out = self.drp1(out, activation=F.softplus, network_layer=self.fc2)
        else:
            out = self.drp1(F.softplus(self.fc2(out)))

        out = self.fc3(out)
        return out

    def representation(self, x):
        out = F.softplus(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.softplus(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.softplus(self.fc1(out))

        no_noise_out = F.softplus(self.fc2(out))

        if self.dropout_method == 'information':
            out = self.drp1(out, activation=F.softplus, network_layer=self.fc2)
        else:
            out = self.drp1(no_noise_out)

        return no_noise_out.view(out.size(0), -1)

    def kl(self):
        kl = 0
        for name, module in self.named_modules():
            if isinstance(module, InformationDropout):
                kl += module.kl()
        return torch.mean(kl, dim=0)

    def saveName(self):
        return 'LeNet_drp'


class DropoutFCNetw(nn.Module):
    def __init__(self, p, dropout_method='information'):
        super(DropoutFCNetw, self).__init__()

        self.dropout_method = dropout_method
        self.p = p

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)

        drp1_layer = nn.Linear(128, 32)
        self.drp1 = dropout(p, dropout_method, layer=drp1_layer, prior='log-normal')

        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        out = F.softplus(self.fc1(x))
        out = F.softplus(self.fc2(out))

        if self.dropout_method == 'information':
            out = self.drp1(out, activation=F.softplus, network_layer=self.fc3)
        else:
            out = self.drp1(F.softplus(self.fc3(out)))

        out = self.fc4(out)
        return out

    def representation(self, x):
        out = F.softplus(self.fc1(x))
        out = F.softplus(self.fc2(out))

        no_noise_out = F.softplus(self.fc3(out))

        if self.dropout_method == 'information':
            out = self.drp1(out, activation=F.softplus, network_layer=self.fc3)
        else:
            out = self.drp1(no_noise_out)

        return no_noise_out.view(out.size(0), -1)

    def kl(self):
        kl = 0
        for name, module in self.named_modules():
            if isinstance(module, InformationDropout):
                kl += module.kl()
        return torch.mean(kl, dim=0)

    def saveName(self):
        return 'FC_drp'

