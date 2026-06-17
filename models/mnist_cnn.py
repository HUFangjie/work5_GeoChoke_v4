import torch
from torch import nn
from models.registry import register_model

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,8,3,1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(16*5*5,32),nn.ReLU(),nn.Linear(32,10))
    def forward(self,x): return self.net(x)

@register_model('mnist_cnn')
def build(): return MNISTCNN()
