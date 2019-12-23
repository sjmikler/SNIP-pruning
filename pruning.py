import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch


class Pruner:
    def __init__(self, model, loader=None, device='cpu', silent=False):
        self.device = device
        self.loader = loader
        self.model = model
        
        self.weights = list(model.parameters())
        self.indicators = [torch.ones_like(layer) for layer in model.parameters()]
        self.pruned = [0 for _ in range(len(self.indicators))]
 
        if not silent:
            print("number of weights to prune:", [x.numel() for x in self.indicators])

    def indicate(self):
        for weight, indicator in zip(self.weights, self.indicators):
            weight.data = weight * indicator
       
    
    def snip(self, sparsity, mini_batches=1, silent=False): # prunes due to SNIP method
        mini_batch=0
        self.indicate()
        self.model.zero_grad()
        grads = [torch.zeros_like(w) for w in self.weights]
        
        for x, y in self.loader:
            x = self.model.forward(x)
            L = torch.nn.CrossEntropyLoss()(x, y)
            grads = [g.abs()+ag.abs() for g, ag in zip(grads, torch.autograd.grad(L, self.weights))]
            mini_batch+=1
            if mini_batch>=mini_batches: break

        with torch.no_grad():
            saliences = [(grad * weight).view(-1).abs().cpu() for weight, grad in zip(self.weights, grads)]
            saliences = torch.cat(saliences)
            
            thresh = float( saliences.kthvalue( int(sparsity * saliences.shape[0] ) )[0] )
            
            for j, layer in enumerate(self.indicators):
                layer[ (grads[j] * self.weights[j]).abs() <= thresh ] = 0
                self.pruned[j] = int(torch.sum(layer == 0))
        self.model.zero_grad() 
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel()-pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100*pruned/self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])
            
            
    def snipR(self, sparsity, silent=False):
        with torch.no_grad():
            saliences = [torch.zeros_like(w) for w in self.weights]
            x, y = next(iter(self.loader))
            z = self.model.forward(x)
            L0 = torch.nn.CrossEntropyLoss()(z, y) # Loss

            for laynum, layer in enumerate(self.weights):
                if not silent: print("layer ", laynum, "...")
                for weight in range(layer.numel()):
                    temp = layer.view(-1)[weight].clone()
                    layer.view(-1)[weight] = 0

                    z = self.model.forward(x) # Forward pass
                    L = torch.nn.CrossEntropyLoss()(z, y) # Loss
                    saliences[laynum].view(-1)[weight] = (L-L0).abs()    
                    layer.view(-1)[weight] = temp
                
            saliences_bag = torch.cat([s.view(-1) for s in saliences]).cpu()
            thresh = float( saliences_bag.kthvalue( int(sparsity * saliences_bag.numel() ) )[0] )

            for j, layer in enumerate(self.indicators):
                layer[ saliences[j] <= thresh ] = 0
                self.pruned[j] = int(torch.sum(layer == 0))   
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel()-pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100*pruned/self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])