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

def batch_norm(input: Tensor, running_mean: Tensor, running_var: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5, momentum: float = 0.1) -> Tensor:
    return input