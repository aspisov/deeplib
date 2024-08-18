from typing import List
import deeplib
import math
from deeplib.nn import functional as F, init
from deeplib.nn.parameter import Parameter

from .module import Module

__all__ = ["Linear"]

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=None):
        super().__init__()
        self.weight = Parameter(deeplib.empty((in_features, out_features), dtype=dtype))
        if bias:
            self.bias = Parameter(deeplib.empty(out_features), dtype=dtype)
        self.reset_parameters()
            
    def reset_parameters(self):
        init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            fan_in = init._calculate_fan(self.weight)[0]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        if self.bias is not None:
            return input @ self.weight + self.bias
        return input @ self.weight