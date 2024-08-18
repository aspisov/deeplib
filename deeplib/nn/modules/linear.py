from typing import List
import deeplib
from .module import Module
import math

__all__ = ["Linear"]


class Linear(Module):
    def __init__(self, in_features, out_features, bais=True):
        super().__init__()
        limit = 1 / math.sqrt(in_features)
        self.weight = deeplib.uniform(-limit, limit, (in_features, out_features), requires_grad=True)
        if bais:
            self.bias = deeplib.uniform(-limit, limit, out_features, requires_grad=True)
        
    def forward(self, input):
        if self.bias is not None:
            return input @ self.weight + self.bias
        return input @ self.weight