import numpy as np
from deeplib.tensor import Tensor

__all__ = ["exp", "log", "sum", "mean", "var", "max"]


def exp(tensor):
    out = Tensor(np.exp(tensor.data), _children=(tensor,), requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            tensor.grad += out.grad * out.data

    out._backward = _backward
    return out

def log(tensor):
    out = Tensor(np.log(tensor.data), _children=(tensor,), requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            tensor.grad += out.grad / tensor.data

    out._backward = _backward
    return out


def sum(tensor, dim=None, keepdims=False):
    out = Tensor(
        np.sum(tensor.data, axis=dim, keepdims=keepdims),
        _children=(tensor,),
        requires_grad=tensor.requires_grad,
    )
    
    def _backward():
        if tensor.requires_grad:
            grad = out.grad
            # if axis is None, the gradient is scalar and should be broadcasted to the original shape
            if dim is None:
                grad = np.ones_like(tensor.data) * grad
            else:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=dim)
                grad = np.broadcast_to(grad, tensor.shape)
            tensor.grad += grad
            
    out._backward = _backward
    return out

def mean(tensor, dim=None, keepdims=False):
    # TODO currently incorrect
    n = tensor.data.size if dim is None else tensor.data.shape[dim]
    return sum(tensor, dim=dim, keepdims=keepdims) / n
    
def var(tensor, dim=None, keepdims=False, unbiased=False):
    tensor_mean = mean(tensor, dim=dim, keepdims=keepdims)
    squared_diff = (tensor - tensor_mean) ** 2
    
    n = tensor.data.size if dim is None else tensor.data.shape[dim]
    if unbiased:
        return sum(squared_diff, dim=dim, keepdims=keepdims) / (n - 1)
    return sum(squared_diff, dim=dim, keepdims=keepdims) / n

def max(tensor, dim=None, keepdims=False):
    out = Tensor(
        np.max(tensor.data, axis=dim, keepdims=keepdims),
        _children=(tensor,),
        requires_grad=tensor.requires_grad,
    )
    
    def _backward():
        if tensor.requires_grad:
            tensor.grad += out.grad * (tensor.data == out.data)
            
    out._backward = _backward
    return out
        
# math operations
Tensor.exp = exp
Tensor.log = log
Tensor.sum = sum
Tensor.mean = mean
Tensor.var = var
Tensor.max = max
