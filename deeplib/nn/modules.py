"""
Implements basic neural network modules.
"""
import numpy as np
import deeplib
import deeplib.nn.functional as F
from deeplib import Tensor
from typing import List


class Module:
    def __init__(self):
        self.training = True
    
    def forward(self, input):
        raise NotImplementedError
    
    def __call__(self, input):
        return self.forward(input)
    
    def parameters(self) -> List[Tensor]:
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
    
class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = args
        
    def forward(self, input):
        for module in self.modules:
            input = module(input)
        return input
    
    def parameters(self) -> List[Tensor]:
        return [p for module in self.modules for p in module.parameters()]
    
class Linear(Module):
    def __init__(self, in_features, out_features, bais=True):
        super().__init__()
        limit = 1 / np.sqrt(in_features)
        self.weight = deeplib.uniform(-limit, limit, (in_features, out_features), requires_grad=True)
        if bais:
            self.bias = deeplib.uniform(-limit, limit, out_features, requires_grad=True)
        
    def forward(self, input):
        if self.bias is not None:
            return input @ self.weight + self.bias
        return input @ self.weight
    
    def parameters(self) -> List[Tensor]:
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
class ReLU(Module):
    def forward(self, input):
        return F.relu(input)
    
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, input: Tensor):
        return F.dropout(input, self.p, self.training)
    
class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = deeplib.ones(num_features, requires_grad=True)
        self.beta = deeplib.zeros(num_features, requires_grad=True)
        self.running_mean = deeplib.zeros(num_features)
        self.running_var = deeplib.ones(num_features)
        self.eps = eps
        self.momentum = momentum
        
    def forward(self, input: Tensor):
        if self.training:
            mean = input.mean(axis=0, keepdims=True)
            var = input.var(axis=0, keepdims=True)
            
            with deeplib.no_grad():
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
            
        xhat = (input - mean) / np.sqrt(var + self.eps)
        out = self.gamma * xhat + self.beta
            
        return out