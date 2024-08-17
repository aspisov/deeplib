from .module import Module
from deeplib import Tensor
from deeplib.nn import functional as F

__all__ = ['MSELoss', 'CrossEntropyLoss']

class MSELoss(Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.mse_loss(input, target)
    
class CrossEntropyLoss(Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.cross_entropy_loss(input, target)