import numpy as np
from deeplib import Tensor


class Module:
    def forward(self, input):
        pass
    
    def __call__(self, input):
        return self.forward(input)
    
class Sequential(Module):
    def __init__(self, *args):
        self.modules = args
        
    def forward(self, input):
        for module in self.modules:
            input = module(input)
        return input
    
class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features), requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)
        
    def forward(self, input):
        return input @ self.W + self.b