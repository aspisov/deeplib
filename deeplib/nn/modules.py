"""
Implements basic neural network modules.
"""
import numpy as np
import deeplib
from typing import List


class Module:
    def forward(self, input):
        raise NotImplementedError
    
    def __call__(self, input):
        return self.forward(input)
    
    def parameters(self) -> List[deeplib.Tensor]:
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
class Sequential(Module):
    def __init__(self, *args):
        self.modules = args
        
    def forward(self, input):
        for module in self.modules:
            input = module(input)
        return input
    
    def parameters(self) -> List[deeplib.Tensor]:
        return [p for module in self.modules for p in module.parameters()]
    
class Linear(Module):
    def __init__(self, in_features, out_features, bais=True):
        limit = 1 / np.sqrt(in_features)
        self.weight = deeplib.uniform(-limit, limit, (in_features, out_features), requires_grad=True)
        if bais:
            self.bias = deeplib.uniform(-limit, limit, out_features, requires_grad=True)
        
    def forward(self, input):
        if self.bias is not None:
            return input @ self.weight + self.bias
        return input @ self.weight
    
    def parameters(self) -> List[deeplib.Tensor]:
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
class ReLU(Module):
    def forward(self, input):
        return input.relu()