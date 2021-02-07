#global imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import time
#local imports
from consts import _epochs, _dataset_png_path,_batch_size

# Implementing the model
def model_init(num_classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #read the model RESNET18
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device) #if cuda or nots
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return model, criterion, exp_lr_scheduler


def data_load():
    # Creating the train/test dataloaders from images
    root_data_dir = _dataset_png_path
    #transform the data
    transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    total_dataset = datasets.ImageFolder(root_data_dir, transform)
    #split train and test
    train_size = int(0.8 * len(total_dataset))
    test_size = len(total_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=True, num_workers=4)

    class_names = total_dataset.classes
    num_classes = len(class_names)
    return train_dataloader, test_dataloader, num_classes


def run():
    train_dataloader, test_dataloader, num_classes = data_load()
    model, criterion, exp_lr_scheduler = model_init(num_classes)
    for epoch in range(_epochs):
        print(f'=== EPOCH {epoch} / {_epochs} ===')
        train()
        test()
        exp_lr_scheduler.step()
    
    #save model
    if(not os.path.isdir('./models')):
        os.mkdir('./models')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_path = './models/model_'+timestr+'.h5'
    torch.save(model, model_path)

run()