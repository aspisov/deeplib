from typing import List
from .module import Module
from deeplib import Tensor

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(self.layers):
            self.register_module(f'layer_{i}', layer)
    
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input