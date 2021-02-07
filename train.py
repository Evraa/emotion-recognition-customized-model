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
from consts import _num_classes, _epochs

# Implementing the model
def model_init():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #read the model RESNET18
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, _num_classes)
    model = model.to(device) #if cuda or nots
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return model, criterion, exp_lr_scheduler


def run():
    model, criterion, exp_lr_scheduler = model_init()
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