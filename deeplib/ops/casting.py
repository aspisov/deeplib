from deeplib import Tensor
import numpy as np

def float(tensor: Tensor):
    return Tensor(tensor.data.astype(np.float32))

def int(tensor: Tensor):
    return Tensor(tensor.data.astype(np.int32))

def bool(tensor: Tensor):
    return Tensor(tensor.data.astype(np.bool_))

def long(tensor: Tensor):
    return Tensor(tensor.data.astype(np.int64))

Tensor.float = float