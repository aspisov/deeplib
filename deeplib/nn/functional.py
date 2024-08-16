import deeplib
from deeplib.tensor import Tensor
import numpy as np

def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training:
        return input
    
    mask = np.random.randn(*input.shape) > p
    return input * mask / (1 - p)

def relu(input: Tensor) -> Tensor:
    return input.relu()
