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

from modules.dataset import ProjectDataset
from modules.model import CNN

from torch.utils.data import DataLoader, random_split

IMAGES_PATH = 'project/data/images'
ANNOTATIONS= 'project/data/annotations'

#--- hyperparameters ---
N_EPOCHS = 200
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 256
BATCH_SIZE_VALIDATION = 256
LR = 0.001

model = CNN()
tfs = transforms.Compose(transforms=[transforms.RandomVerticalFlip(0.2), transforms.RandomRotation(45), transforms.RandomSolarize(100,0.2), transforms.GaussianBlur(kernel_size=3)])
dataset = ProjectDataset(img_dir=IMAGES_PATH, annotations_dir=ANNOTATIONS, rebuildAnnotations=False, transform=tfs)
tr_len = round(len(dataset) * 0.7)
te_len = len(dataset) - tr_len
train_ind, test_ind = random_split(dataset, [tr_len, te_len] , generator=torch.Generator().manual_seed(42))
tr_dl = DataLoader(train_ind, batch_size=32, shuffle=True)
te_dl = DataLoader(test_ind, batch_size=32, shuffle=True)

#--- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.75)
loss_function = torch.nn.BCELoss()

training_losses = []
validation_losses = []

training_accuracies = []
validation_accuracies = []


epoch_stopped = 0

#--- training ---
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    for batch_num, (data, target) in enumerate(tr_dl):
        data, target = data.float().to(device), target.float().to(device)
        
        optimizer.zero_grad() 
        output = model(data)
        train_loss = loss_function(output, target)
        train_loss.backward()
        optimizer.step()
        
        pred = torch.round(output)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        total += pred.shape.numel()

        if batch_num % 25 == 0:
            print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                  (epoch, batch_num, len(tr_dl), train_loss / (batch_num + 1), 
                   100. * train_correct / total, train_correct, total))
    
    validation_loss = 0
    validation_correct = 0
    validation_total = 0
    
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(te_dl):
            data, target = data.float().to(device), target.float().to(device)

            output = model(data)
            validation_loss += loss_function(output, target).item()
            
            pred = torch.round(output)
            validation_correct += pred.eq(target.view_as(pred)).sum().item()
            validation_total += pred.shape.numel()

    
    print('Validation: Epoch %d: Loss: %.4f | Validation Acc: %.3f%% (%d/%d)' % 
          (epoch, validation_loss, (100. * validation_correct / validation_total), validation_correct, 
           validation_total))
    
    # Check if average of the last 5 validation losses is smaller than current validation loss
    if epoch > 5 and statistics.mean(validation_losses[-6:]) < validation_loss:
        epoch_stopped = epoch
        break
        
    training_losses.append(train_loss.detach().cpu())
    validation_losses.append(validation_loss)
        
    training_accuracies.append((100. * train_correct / total))
    validation_accuracies.append((100. * validation_correct / validation_total))
