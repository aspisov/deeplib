import numpy as np
from collections import deque
from typing import Optional


class Tensor:
    def __init__(self, data, _children=(), requires_grad=False, dtype=np.float32):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        self.data = data
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None
        self._children = set(_children)
        self.requires_grad = requires_grad
        
    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0)
        else:
            self.grad = np.zeros_like(self.data)
    
    def backward(self):
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
        data = repr(self.data).replace("array(", "tensor(").replace(")", "")
        lines = data.split("\n")
        data = "\n".join([lines[0]] + [" " + line for line in lines[1:]])
        return f"{data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, _children=(self, other), requires_grad=requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad if self.grad is not None else np.zeros_like(self.data)
                self.grad += np.sum(out.grad, axis=0)
            if other.requires_grad:
                other.grad = other.grad if other.grad is not None else np.zeros_like(other.data)
                other.grad += np.sum(out.grad, axis=0)
        
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
            self.grad = self.grad if self.grad is not None else np.zeros_like(self.data)
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
            
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
            self.grad = self.grad if self.grad is not None else np.zeros_like(self.data)
            self.grad += out.grad @ other.data.T
            other.grad += out.grad @ self.data.T
            
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        
        requires_grad = self.requires_grad
        out = Tensor(self.data**power, _children=(self,), requires_grad=self.requires_grad)
        
        def _backward():
            self.grad = self.grad if self.grad is not None else np.zeros_like(self.data)
            self.grad += out.grad * power * self.data**(power - 1) 
            
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,), requires_grad=self.requires_grad)
        
        def _backward():
            self.grad = self.grad if self.grad is not None else np.zeros_like(self.data)
            self.grad += out.grad * out.data
            
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                     _children=(self,), 
                     requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad if self.grad is not None else np.zeros_like(self.data)
                if keepdims:
                    grad = out.grad
                else:
                    grad = out.grad.reshape(out.grad.shape + (1,) * (self.data.ndim - out.grad.ndim))
                self.grad += grad
                
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)),
                     _children=(self,),
                     requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += out.grad * out.data * (np.ones_like(out.data) - out.data)
            
        out._backward = _backward
        return out
    

        