import deeplib
from deeplib.tensor import Tensor
from .module import Module
import numpy as np


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = deeplib.ones(num_features, requires_grad=True)
        self.beta = deeplib.zeros(num_features, requires_grad=True)
        self.running_mean = deeplib.zeros(num_features)
        self.running_var = deeplib.ones(num_features)
        self.eps = eps
        self.momentum = momentum

    def forward(self, input: Tensor):
        if self.training:
            mean = input.mean(dim=0, keepdims=True)
            # use unbiased var estimation, however pytorch uses biased one
            var = input.var(dim=0, keepdims=True, unbiased=True)

            with deeplib.no_grad():
                self.running_mean = (
                    self.momentum * mean + (1 - self.momentum) * self.running_mean
                )
                self.running_var = (
                    self.momentum * var + (1 - self.momentum) * self.running_var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        xhat = (input - mean) / np.sqrt(var + self.eps)
        out = self.gamma * xhat + self.beta

        return out
