import numpy as np
from deeplib.tensor import Tensor


def max(tensor, dim=None, keepdims=False):
    return tensor.max(dim=dim, keepdims=keepdims)


def sum(tensor, dim=None, keepdims=False):
    return tensor.sum(dim=dim, keepdims=keepdims)


def mean(tensor, dim=None, keepdims=False):
    return tensor.mean(dim=dim, keepdims=keepdims)


def var(tensor, dim=None, keepdims=False, unbiased=False):
    return tensor.var(dim=dim, keepdims=keepdims, unbiased=unbiased)


def log(tensor):
    return tensor.log()


def exp(tensor):
    return tensor.exp()


