import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import sys 
sys.path.append('../')


from .sphere import SphereConv, SpherePool

class SphereCNN(nn.Module):
    def __init__(self):
        super(SphereCNN, self).__init__()
        self.conv1 = SphereConv(1, 32, stride=1)
        self.pool1 = SpherePool(stride=2)
        self.conv2 = SphereConv(32, 64, stride=1)
        self.pool2 = SpherePool(stride=2)
        self.fc = nn.Linear(64*15*15, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 64*15*15)
        x = self.fc(x)
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(64*13*13, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 64*13*13)
        x = self.fc(x)
        return x
