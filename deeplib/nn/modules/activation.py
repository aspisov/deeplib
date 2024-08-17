from .module import Module
from deeplib.nn import functional as F


class ReLU(Module):
    def forward(self, input):
        return F.relu(input)