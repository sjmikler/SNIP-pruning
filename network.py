import time
import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch


class NN:
    def __init__(self, model, device): # nn = NN(model, prepr, 'cpu')
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.epoch = 0
        
        # DEFAULTS
        self.optim = torch.optim.SGD(params=model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)
    
    def fit(self, data_loader, loss, silent=False, s_report=100, v_report=1, max_epochs=float('inf'), loop_operation=[]):
        try:
            if not silent:
                print("TRAINING:\n{:<10} | {:<10} | {:<12} - {}\n".format\
                      ('epoch', 'time', 'error rate', 'mistakes...'))
            
            i, result = 0, 0
            self.stime = time.time()
            eval_length = data_loader['eval'].dataset.__len__()
            history = {'train':{'i':[], 'e':[], 'l':[]}, 'eval':{'i':[], 'e':[]}}
            self.scheduler.last_epoch = self.epoch-1
            
            while True:
                if not self.epoch % v_report:
                    res = self.evaluate(data_loader['eval'], eval_length, result, silent)
                    history['eval']['e'].append(float(res)); history['eval']['i'].append(i)
                    
                self.epoch+=1
                self.model.train()
                for x, y in data_loader['train']:
                    self.optim.zero_grad()
                    x = self.model.forward(x)
                    L = loss(x, y)
                    i+=1
                    
                    L.backward()
                    with torch.no_grad():
                        self.optim.step()
                        result = torch.sum(torch.argmax(x, dim=-1)==y, dtype=torch.float)/x.shape[0]
                        for op in loop_operation:
                            op() # You can run some additional, outside functions here running every iteration
                           
                    if not i%s_report:
                        history['train']['e'].append(float(100-100*result))
                        history['train']['l'].append(float(L)); history['train']['i'].append(i)
                        if not silent: self.report(i, L, result)
                
                self.scheduler.step()       
                if self.epoch>=max_epochs:
                    break
                    
        except KeyboardInterrupt:
            pass
        
        res = self.evaluate(data_loader['eval'], eval_length, result, silent)
        history['eval']['e'].append(float(res)); history['eval']['i'].append(i)
        if not silent:
            self.draw_history(history)
        return history
    
    def report(self, iteration, loss, error):
        print( "Iter {:>5} | err {:>5.2f}% | loss {:<5.3f}".format(iteration, 100-100*error, loss), end='\r' )
    
    def evaluate(self, data_loader, eval_length, train_error, silent=False):
        self.model.eval()
        with torch.no_grad():
            result = torch.tensor(0.)
            for x, y in data_loader:
                x = self.model.forward(x)
                predicts = torch.argmax(x, dim=-1)
                result += torch.sum(y==predicts)
        if not silent:
            print( "Epoch {:>4} | time {:0<5.3} | lr {:0<7.4} | tr {:>7.3f}% | eval {:>6.3f}% - {} mis".format( self.epoch,
                    time.time()-self.stime, self.optim.param_groups[0]['lr'],
                    100-100*train_error, 100-100*result/eval_length, int(eval_length-result)))
        self.stime = time.time()
        return 100 - 100*result/eval_length
        
    def predict(self, data, answers=None, silent=False):
        self.model.eval()
        with torch.no_grad():
            x = self.model.forward(data)
            x = torch.argmax(x, dim=-1)
        if answers is None:
            return x
        
        res = torch.sum(x==answers, dtype=torch.float) / x.shape[0]
        if not silent:
            print("Accuracy: {:>6.3f}%".format(100 * res))
            
        return float(res)
    
    def draw_history(self, history):
        print('\n'); plt.figure(figsize=(15,4)); plt.subplot(121)
        plt.plot(history['train']['i'], history['train']['l'])
        plt.xlabel('iterations'); plt.ylabel('training loss'); plt.grid(True, linewidth=0.2); plt.subplot(122)
        plt.plot(history['eval']['i'][1:], history['eval']['e'][1:], label='evaluation')
        plt.plot(history['train']['i'][1:], history['train']['e'][1:], label='training')
        plt.xlabel('iterations'); plt.ylabel('errors');
        plt.grid(True, linewidth=0.2); plt.legend()