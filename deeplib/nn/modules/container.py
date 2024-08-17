from typing import List
from .module import Module
from deeplib import Tensor

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