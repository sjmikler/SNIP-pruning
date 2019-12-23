import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch

'''
One can flatten the data before sending to classifier
And thicken to ensure that it has depth to go through convolutional layers
'''
class LayerFlatten(torch.nn.Module):
    def forward(self, x): return x.view(x.shape[0], -1)
    
class LayerThicken(torch.nn.Module):
    def forward(self, x):
        if x.dim()==4:
            if x.shape[-1]==3 or x.shape[-1]==1:
                x = x.transpose(1,-1)
            return x
        if x.dim()==3:
            return x[:,None,:,:]
        if x.dim()==2:
            n, m = x.shape
            return x[n, None, int(m**0.5), m-int(m**0.5)]

        
def LeNet_300_100():
    model = torch.nn.Sequential(
        LayerFlatten(),
        torch.nn.Linear(784, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
        torch.nn.LogSoftmax(dim=-1))
    return model


def LeNet5(init=None):
    model = torch.nn.Sequential(
        LayerThicken(), 
        torch.nn.Conv2d(1, 6, kernel_size=5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(6, 16, kernel_size=5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        LayerFlatten(),
        torch.nn.Linear(400,120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 10),
        torch.nn.LogSoftmax(dim=-1))
    return model


def vgg16():
    vgg = models.vgg16_bn()
    vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    vgg.classifier = torch.nn.Sequential(
        torch.nn.Linear(512, 300),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(300),
        torch.nn.Linear(300,10),
        torch.nn.LogSoftmax(dim=-1))
    model = torch.nn.Sequential( LayerThicken(), *vgg.features , LayerFlatten(), *vgg.classifier )
    return model