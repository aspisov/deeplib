"""
Implements basic neural network modules.
"""
import deeplib.nn.functional as F
from deeplib import Tensor
from typing import List

__all__ = ["Module"]

class Module:
    def __init__(self, *args, **kwargs):
        self.training = True
    
    def forward(self, input):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> List[Tensor]:
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
    
