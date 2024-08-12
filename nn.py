import numpy as np
from autograd import Tensor

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
    
        
    
    

if __name__ == "__main__":
    model = Sequential(
        Linear(5, 10),
        Linear(10, 4),
        Linear(4, 1)
    )
    input = Tensor(np.random.randn(100, 5))
    out = model(input)
    print(out.shape)