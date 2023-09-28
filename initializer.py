import torch 
from torch import nn 

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.BatchNorm2d)):
            if model.bias.data is not None:
                model.bias.daya.zero_()
            else:
                nn.init.kaiming_normal_(model.weight.data,mode='fan_in',nonlinearity='leaky_relu')
                
