from deeplib.tensor import Tensor
from .module import Module
import deeplib.nn.functional as F


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, input: Tensor):
        return F.dropout(input, self.p, self.training)