"""
EECS 445 - Introduction to Machine Learning
Winter 2021 - Project 2
Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Source(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 8, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(32, 8)
        ##

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc1]
        nn.init.normal_(self.fc1.weight, 0.0, 1 / 32.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        ##

    def forward(self, x):
        N, C, H, W = x.shape 
        #Initial
        #print(C,H,W, sep='X')

        #Conv1
        z = F.relu(self.conv1(x))
  
        #Pool1
        z = self.pool(z)

        #Conv2
        z = F.relu(self.conv2(z))

        #Pool2
        z = self.pool(z)

        #Conv3
        z = F.relu(self.conv3(z))

        z = z.view(N, -1)
        z = self.fc1(z)
        return z


