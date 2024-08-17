import numpy as np
from deeplib import Tensor

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)

def FloatTensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad, dtype=np.float32)


def LongTensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad, dtype=np.int64)

def ones_like(tensor, requires_grad=False):
    return Tensor(np.ones_like(tensor.data), requires_grad=requires_grad)


def zeros_like(tensor, requires_grad=False):
    return Tensor(np.zeros_like(tensor.data), requires_grad=requires_grad)


def ones(shape, requires_grad=False):
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def zeros(shape, requires_grad=False):
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def randn(shape, requires_grad=False):
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)


def uniform(low, high, shape, requires_grad=False):
    return Tensor(np.random.uniform(low, high, shape), requires_grad=requires_grad)


def rand_like(tensor: Tensor, requires_grad: bool = False):
    return Tensor(np.random.randn(*tensor.shape), requires_grad=requires_grad)