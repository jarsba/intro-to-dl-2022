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
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.GroupNorm(8,32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.GroupNorm(8,64),
            nn.LeakyReLU(),
            nn.Conv2d(64,128, kernel_size=5, stride=2),
            nn.GroupNorm(8,128),
            nn.LeakyReLU(),
            nn.Conv2d(128,16, kernel_size=1),
            nn.GroupNorm(8,16),
            nn.LeakyReLU(),

        )
        self.fc_layers = nn.Sequential(
            nn.Linear(2704, 2048),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(2048, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, x.shape[1:].numel())
        x = self.fc_layers(x)
        return x