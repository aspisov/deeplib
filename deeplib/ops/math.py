import numpy as np
from deeplib.tensor import Tensor

__all__ = [
    "add",
    "sub",
    "mul",
    "true_divide",
    "pow",
    "neg",
    "sum",
    "mean",
    "max",
    "matmul",
    "exp",
    "log",
    "sqrt",
    "var",
]


def add(tensor1, tensor2):
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)
    
    out = Tensor(
        tensor1.data + tensor2.data,
        _children=(tensor1, tensor2),
        requires_grad=tensor1.requires_grad or tensor2.requires_grad,
    )

    def _backward():
        if tensor1.requires_grad:
            grad = out.grad
            # sum over all broadcasted axes
            while grad.ndim > tensor1.grad.ndim:
                grad = grad.sum(axis=0)
            for i, dim in enumerate(tensor1.grad.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            tensor1.grad = tensor1.grad + grad
        if tensor2.requires_grad:
            grad = out.grad
            # sum over all broadcasted axes
            while grad.ndim > tensor2.grad.ndim:
                grad = grad.sum(axis=0)
            for i, dim in enumerate(tensor2.grad.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            tensor2.grad = tensor2.grad + grad

    out._backward = _backward
    return out

def mul(tensor1, tensor2):
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)
    
    out = Tensor(tensor1.data * tensor2.data,
                 _children=(tensor1, tensor2),
                 requires_grad=tensor1.requires_grad or tensor2.requires_grad)

    def _backward():
        if tensor1.requires_grad:
            grad = tensor2.data * out.grad
            # sum over all broadcasted axes
            while grad.ndim > tensor1.grad.ndim:
                grad = grad.sum(axis=0)
            for i, dim in enumerate(tensor1.grad.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            tensor1.grad = tensor1.grad + grad

        if tensor2.requires_grad:
            grad = tensor1.data * out.grad
            # sum over all broadcasted axes
            while grad.ndim > tensor2.grad.ndim:
                grad = grad.sum(axis=0)
            for i, dim in enumerate(tensor2.grad.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            tensor2.grad = tensor2.grad + grad

    out._backward = _backward
    return out
    
def matmul(tensor1, tensor2):
    out = Tensor(tensor1.data @ tensor2.data,
                 _children=(tensor1, tensor2),
                 requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    def _backward():
        if tensor1.requires_grad:
            tensor1.grad += out.grad @ tensor2.data.T
        if tensor2.requires_grad:
            tensor2.grad += tensor1.data.T @ out.grad
        
    out._backward = _backward
    return out

def pow(tensor, power):
    assert isinstance(power, (int, float))
    
    out = Tensor(tensor.data**power,
                 _children=(tensor,),
                 requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            tensor.grad += out.grad * power * tensor.data ** (power - 1)

    out._backward = _backward
    return out

def sqrt(tensor):
    return pow(tensor, 0.5)

def neg(tensor):
    return tensor*-1

def sub(tensor1, tensor2):
    return add(tensor1, neg(tensor2))

def true_divide(tensor1, tensor2):
    return mul(tensor1, tensor2**-1)


def exp(tensor):
    out = Tensor(np.exp(tensor.data), _children=(tensor,), requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            tensor.grad += out.grad * out.data

    out._backward = _backward
    return out

def log(X):
    out = Tensor(np.log(X.data), _children=(X,), requires_grad=X.requires_grad)

    def _backward():
        if X.requires_grad:
            X.grad += out.grad / X.data

    out._backward = _backward
    return out


def sum(X, dim=None, keepdims=False):
    out = Tensor(
        np.sum(X.data, axis=dim, keepdims=keepdims),
        _children=(X,),
        requires_grad=X.requires_grad,
    )
    
    def _backward():
        if X.requires_grad:
            grad = out.grad
            # if axis is None, the gradient is scalar and should be broadcasted to the original shape
            if dim is None:
                grad = np.ones_like(X.data) * grad
            else:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=dim)
                grad = np.broadcast_to(grad, X.shape)
            X.grad += grad
            
    out._backward = _backward
    return out

def mean(X, dim=None, keepdims=False):
    # TODO currently incorrect
    n = X.data.size if dim is None else X.data.shape[dim]
    return sum(X, dim=dim, keepdims=keepdims) / n
    
def var(X, dim=None, keepdims=False, unbiased=False):
    tensor_mean = mean(X, dim=dim, keepdims=keepdims)
    squared_diff = (X - tensor_mean) ** 2
    
    n = X.data.size if dim is None else X.data.shape[dim]
    if unbiased:
        return sum(squared_diff, dim=dim, keepdims=keepdims) / (n - 1)
    return sum(squared_diff, dim=dim, keepdims=keepdims) / n

def max(X, dim=None, keepdims=False):
    out = Tensor(
        np.max(X.data, axis=dim, keepdims=keepdims),
        _children=(X,),
        requires_grad=X.requires_grad,
    )
    
    def _backward():
        if X.requires_grad:
            X.grad += out.grad * (X.data == out.data)
            
    out._backward = _backward
    return out

def argmax(X: Tensor, dim: int = None):
    return Tensor(np.argmax(X.data, axis=dim))


Tensor.__add__ = add
Tensor.__iadd__ = add
Tensor.__radd__ = add
Tensor.__neg__ = neg
Tensor.__sub__ = sub
Tensor.__isub__ = sub
Tensor.__mul__ = mul
Tensor.__rmul__ = mul
Tensor.__truediv__ = true_divide
Tensor.__matmul__ = matmul
Tensor.__pow__ = pow
Tensor.sqrt = sqrt
        
Tensor.exp = exp
Tensor.log = log
Tensor.sum = sum
Tensor.mean = mean
Tensor.var = var
Tensor.max = max
Tensor.argmax = argmax
