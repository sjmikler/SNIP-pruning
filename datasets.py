import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch

DEVICE = 'cpu'

data = datasets.MNIST('Data', download=True)
mnist_full_x = data.data
mnist_full_y = data.targets

'''
Loader code
it splits the data into training and validation set
it returns two generators
'''
def RamLoad(dataset, labels, split=1, batch_size=1,
                   t_trans=transforms.Compose([]), v_trans=transforms.Compose([])):
    
    class RamLoader(torch.utils.data.DataLoader): 
        def __init__(self, data, labels, transform, **kwargs):
            super().__init__(torch.utils.data.TensorDataset(data,labels), **kwargs)
            self.dataset.transform = transform
    
    indices = torch.randperm(dataset.shape[0])
    split_point = int(split*dataset.shape[0])
    train_data = dataset[ indices[:split_point] ]; train_labels= labels[ indices[:split_point] ]
    valid_data = dataset[ indices[split_point:] ]; valid_labels= labels[ indices[split_point:] ]
    
    return {'train':RamLoader(train_data, train_labels, transform=t_trans, batch_size=batch_size, shuffle=True),
            'eval' :RamLoader(valid_data, valid_labels, transform=v_trans, batch_size=batch_size, shuffle=False)}

'''
Useful data transformations
'''
class TransformFlatten:
    def __call__(self, x):
        return x.view(x.shape[0], -1)
    
class TransformStandarize:
    def __init__(self, mean=0, std=1):
        self.mean, self.std = mean, std
    def __call__(self, x):
        return (x - torch.mean(x) + self.mean) / torch.std(x) * self.std
    
'''
Data is ready from here
'''
tmnist_x = TransformStandarize()(torch.tensor(mnist_full_x, device=DEVICE, dtype=torch.float))
tmnist_y = torch.tensor(mnist_full_y, device=DEVICE, dtype=torch.long)

RamLoader = RamLoad(tmnist_x, tmnist_y, split=0.85, batch_size=128)