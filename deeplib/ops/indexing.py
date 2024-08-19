from deeplib.tensor import Tensor
import numpy as np


def getitem(X, key):
    if isinstance(key, Tensor):
        key = key.data
    sliced_data = X.data[key]
    out = Tensor(sliced_data, _children=(X,), requires_grad=X.requires_grad)

    def _backward():
        if X.requires_grad:
            grad = np.zeros_like(X.data)
            grad[key] = out.grad
            X.grad += grad

    out._backward = _backward
    return out


def gather(X, dim, index):
    # ensure index is a Tensor
    if not isinstance(index, Tensor):
        index = Tensor(index)

    # create a list of slice objects for indexing
    slices = [slice(None)] * X.dim()

    # replace the slice at the specified dimension with the index array
    slices[dim] = index.data

    # use advanced indexing to gather the values
    gathered_data = X.data[tuple(slices)]

    out = Tensor(gathered_data, _children=(X,), requires_grad=X.requires_grad)

    def _backward():
        if X.requires_grad:
            grad = np.zeros_like(X.data)
            np.add.at(grad, tuple(slices), out.grad)
            X.grad += grad

    out._backward = _backward
    return out


# overload
Tensor.__getitem__ = getitem
Tensor.gather = gather
