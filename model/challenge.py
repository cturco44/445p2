"""
EECS 445 - Introduction to Machine Learning
Winter 2021 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        ## Round 1
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # Round 2
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # Round 3
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        # Round 4
        self.conv7 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2)
        ##

        self.init_weights()

    def init_weights(self):
        l = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, 
        self.conv6, self.conv7, self.conv8]
        for conv in l:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        nn.init.normal_(self.fc1.weight, 0.0, 1 / 2048.0)
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.normal_(self.fc2.weight, 0.0, 1 / 2048.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        ##

    def forward(self, x):
        N, C, H, W = x.shape
        # Round 1
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = self.pool(z)

        # Round 2
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = self.pool(z)

        # Round 3
        z = F.relu(self.conv5(z))
        z = F.relu(self.conv6(z))
        z = self.pool(z)

        # Round 4
        z = F.relu(self.conv7(z))
        z = F.relu(self.conv8(z))
        z = self.pool(z)

        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        z = F.relu(self.fc1(z))
        z = self.dropout(z)

        z = F.relu(self.fc2(z))

        return z
