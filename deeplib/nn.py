import numpy as np
from deeplib import Tensor
from typing import List


class Module:
    def forward(self, input):
        raise NotImplementedError
    
    def __call__(self, input):
        return self.forward(input)
    
    def parameters(self) -> List[Tensor]:
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
    
    def parameters(self) -> List[Tensor]:
        return [p for module in self.modules for p in module.parameters()]
    
class Linear(Module):
    def __init__(self, in_features, out_features):
        # Xavier initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)
        
    def forward(self, input):
        return input @ self.W + self.b
    
    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]
    
class ReLU(Module):
    def forward(self, input):
        return input.relu()