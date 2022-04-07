#--- model ---

import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import statistics

NUM_CLASSES=14

#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            #first conv
            nn.Conv2d(3, 64, kernel_size=3),
            nn.GroupNorm(16,64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.GroupNorm(16,64),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            #second conv
            nn.Conv2d(64,128, kernel_size=3),
            nn.GroupNorm(16,128),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3),
            nn.GroupNorm(16,128),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            #third conv
            nn.Conv2d(128,256, kernel_size=3),
            nn.GroupNorm(16,256),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3),
            nn.GroupNorm(16,256),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3),
            nn.GroupNorm(16,256),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            #fourth conv
            nn.Conv2d(256,512, kernel_size=3),
            nn.GroupNorm(16,512),
            nn.ReLU(),
            nn.Conv2d(512,512, kernel_size=3),
            nn.GroupNorm(16,512),
            nn.ReLU(),
            nn.Conv2d(512,512, kernel_size=3),
            nn.GroupNorm(16,512),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.33),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.33),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Dropout(0.33),
            nn.Linear(1000, NUM_CLASSES),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, x.shape[1:].numel())
        x = self.fc_layers(x)
        return x
