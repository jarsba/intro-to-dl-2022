import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchinfo
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
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': np.round(precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0),3),
            'micro/recall': np.round(recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0),3),
            'micro/f1': np.round(f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0),3),
            }



IMAGES_PATH = 'project/data/images'
ANNOTATIONS= 'project/data/annotations'

#--- hyperparameters ---
N_EPOCHS = 250
BATCH_SIZE_TRAIN = 256 
BATCH_SIZE_TEST = 256 
BATCH_SIZE_VALIDATION = 256

model = CNN()
tfs = transforms.Compose(transforms=[transforms.RandomRotation(30), transforms.RandomHorizontalFlip(p=0.15), \
        transforms.ColorJitter(brightness=.5, hue=.3), transforms.RandomEqualize(p=0.15), transforms.RandomErasing(p=0.2)])
dataset = ProjectDataset(img_dir=IMAGES_PATH, annotations_dir=ANNOTATIONS, rebuildAnnotations=False, transform=tfs)
tr_len = round(len(dataset) * 0.7)
te_len = len(dataset) - tr_len
train_ind, test_ind = random_split(dataset, [tr_len, te_len] , generator=torch.Generator().manual_seed(42))
tr_dl = DataLoader(train_ind, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
te_dl = DataLoader(test_ind, batch_size=BATCH_SIZE_TEST, shuffle=True)

#--- set up ---
if torch.cuda.is_available():
    
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)
torchinfo.summary(model)

optimizer = optim.Adam(model.parameters())
loss_function = torch.nn.BCEWithLogitsLoss(weight=dataset.get_weights().to(device))

training_losses = []
validation_losses = []

training_accuracies = []
validation_accuracies = []


epoch_stopped = 0

#--- training ---
for epoch in range(N_EPOCHS):
    model.train(True)
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
        
        pred = torch.sigmoid(output)
        if batch_num % 10 == 0:

            metrics = calculate_metrics(pred.detach().cpu().numpy(), target.detach().cpu().numpy())
            print(f'TRAINING - Epoch {epoch } Batch {batch_num}: loss {train_loss}, f1 {metrics["micro/f1"]},  prec {metrics["micro/precision"]}, rec {metrics["micro/recall"]}')
        

    validation_loss = 0
    validation_correct = 0
    validation_total = 0
    
    model.train(False)
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(te_dl):
            data, target = data.float().to(device), target.float().to(device)

            output = model(data)
            validation_loss = loss_function(output, target).item()
            
            pred = torch.sigmoid(output)
            if batch_num % 10 == 0:
                metrics = calculate_metrics(pred.detach().cpu().numpy(), target.detach().cpu().numpy())

                print(f'VALIDATION - Epoch {epoch} Batch {batch_num}: loss {validation_loss}, f1 {metrics["micro/f1"]}, prec {metrics["micro/precision"]}, rec {metrics["micro/recall"]}')

    if epoch != 250 and epoch % 25 == 0:
        torch.save(model.state_dict(), f'model_state_{epoch}.bin')

torch.save(model.state_dict(), f'model_state_{epoch}.bin')


