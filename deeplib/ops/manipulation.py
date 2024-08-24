import deeplib as dl
import numpy as np

__all__ = ["stack"]

def stack(tensors, dim=0):
    return dl.Tensor(np.stack([t.data for t in tensors], axis=dim))