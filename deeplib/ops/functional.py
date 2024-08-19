from deeplib import Tensor
import numpy as np

__all__ = ["tanh", "sigmoid", "relu"]

def tanh(X: Tensor) -> Tensor:
    out = Tensor(np.tanh(X.data), _children=(X,), requires_grad=X.requires_grad)
    
    def _backward():
        if X.requires_grad:
            X.grad += out.grad * (1 - out.data**2)
            
    out._backward = _backward
    return out

def sigmoid(X: Tensor) -> Tensor:
    out = Tensor(1 / (1 + np.exp(-X.data)), _children=(X,), requires_grad=X.requires_grad)
    
    def _backward():
        if X.requires_grad:
            X.grad += out.grad * out.data * (1 - out.data)
            
    out._backward = _backward
    return out

def relu(X: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, X.data), _children=(X,), requires_grad=X.requires_grad)
    
    def _backward():
        if X.requires_grad:
            X.grad += out.grad * (X.data > 0)
            
    out._backward = _backward
    return out

