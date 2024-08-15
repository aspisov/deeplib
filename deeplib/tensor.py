import numpy as np
from collections import deque
from typing import Optional

class NoGrad:
    _enabled = False

    def __enter__(self):
        self.prev = NoGrad._enabled
        NoGrad._enabled = True

    def __exit__(self, exc_type, exc_value, traceback):
        NoGrad._enabled = self.prev

def no_grad():
    return NoGrad()

class Tensor:
    def __init__(self, data, _children=(), requires_grad=False, dtype=np.float32):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        self.data = data
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        
        self.requires_grad = requires_grad and not NoGrad._enabled
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward = lambda: None
        self._children = set(_children)
        
    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0)
    
    def backward(self) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require gradients.")
        self.grad = np.ones_like(self.data)
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        for node in reversed(topo):
            node._backward()
        
    def __repr__(self):
        return f"tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, _children=(self, other), requires_grad=requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad
            if other.requires_grad:
                grad_other = out.grad
                while grad_other.ndim > other.grad.ndim:
                    grad_other = grad_other.sum(axis=0)
                for i, dim in enumerate(other.grad.shape):
                    if dim == 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad = other.grad + grad_other

        out._backward = _backward
        return out
    
    def __iadd__(self, other):
        return self + other
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, _children=(self, other), requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = other.data * out.grad
                while grad_self.ndim > self.grad.ndim:
                    grad_self = grad_self.sum(axis=0)
                for i, dim in enumerate(self.grad.shape):
                    if dim == 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad = self.grad + grad_self
            
            if other.requires_grad:
                grad_other = self.data * out.grad
                while grad_other.ndim > other.grad.ndim:
                    grad_other = grad_other.sum(axis=0)
                for i, dim in enumerate(other.grad.shape):
                    if dim == 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad = other.grad + grad_other
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __matmul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, _children=(self, other), requires_grad=requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
            
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        
        out = Tensor(self.data**power, _children=(self,), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * power * self.data**(power - 1)   
            
        out._backward = _backward
        return out
    
    def __gt__(self, other):
        assert isinstance(other, (int, float))
        
        out = Tensor(self.data > other, _children=(self,), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad if self.grad is not None else np.zeros_like(self.data)
                self.grad += out.grad * out.data
            
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data
            
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                    _children=(self,), 
                    requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                # if axis is None, the gradient is scalar and should be broadcasted to the original shape
                if axis is None:
                    grad = np.ones_like(self.data) * grad
                else:
                    if not keepdims:
                        grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)
                self.grad += grad
                
        out._backward = _backward
        return out

    
    def mean(self, axis=None, keepdims=False):
        return self.sum(axis=axis, keepdims=keepdims) / self.data.size
    
    def var(self, axis=None, keepdims=False):
        mean = self.mean(axis=axis, keepdims=True)
        out = ((self - mean)**2).mean(axis=axis, keepdims=keepdims)
        return out
    
    def sqrt(self):
        return self**0.5
    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)),
                     _children=(self,),
                     requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data * (1 - out.data)
            
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data),
                     _children=(self,),
                     requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)
            
        out._backward = _backward
        return out

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

