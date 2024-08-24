import deeplib as dl
from deeplib.nn import init
from deeplib.nn.parameter import Parameter
from deeplib.tensor import Tensor
from .module import Module

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = Parameter(dl.empty(num_features))
        self.beta = Parameter(dl.empty(num_features))
        
        self.running_mean = dl.zeros(num_features)
        self.running_var = dl.ones(num_features)
        
        self.eps = eps
        self.momentum = momentum
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        
        self.running_mean.fill_(0)
        self.running_var.fill_(1)

    def forward(self, input: Tensor):
        if self.training:
            mean = input.mean(dim=0, keepdims=True)
            # use unbiased var estimation, however pytorch uses biased one
            var = input.var(dim=0, keepdims=True, unbiased=True)

            with dl.no_grad():
                self.running_mean = (
                    self.momentum * mean + (1 - self.momentum) * self.running_mean
                )
                self.running_var = (
                    self.momentum * var + (1 - self.momentum) * self.running_var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        xhat = (input - mean) / dl.sqrt(var + self.eps)
        out = self.gamma * xhat + self.beta

        return out
