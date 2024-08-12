import numpy as np
from collections import deque

class Tensor:
    def __init__(self, data, prev=(), func="", name="", requires_grad=False, dtype=np.float32):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data.astype(dtype)
        self.shape = self.data.shape
        
        self.grad = np.zeros_like(data).astype(dtype)
        self._backward = lambda: None
        self._prev = set(prev)
        self._func = func
        self.name = name
        self.requires_grad = requires_grad
        
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
        
    def detach(self):
        return Tensor(self.data, requires_grad=False)
    
    def backward(self):
        self.grad = np.ones_like(self.data)
        
        # bfs
        queue = deque([self])
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                # if we don't need gradient, skip this node
                if not node.requires_grad:
                    continue
                # propagate further
                node._backward()
                for child in node._prev:
                    queue.append(child)
        
    def __repr__(self):
        data = repr(self.data).replace("array(", "tensor(").replace(")", "")
        lines = data.split("\n")
        data = "\n".join([lines[0]] + [" " + line for line in lines[1:]])
        return f"{data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, prev=(self, other), func="+", requires_grad=requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                if other.grad.shape == out.grad.shape:
                    other.grad += out.grad
                else:
                    other.grad += np.sum(out.grad, axis=tuple(range(out.grad.ndim - other.grad.ndim)))
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, prev=(self, other), func="*", requires_grad=requires_grad)

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def _truediv_(self, other):
        return self * other**-1
    
    def __matmul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, prev=(self, other), func="matmul", requires_grad=requires_grad)
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += out.grad @ self.data.T
            
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        
        requires_grad = self.requires_grad
        out = Tensor(self.data**power, prev=(self,), func=f"^{power}", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += out.grad * power * self.data**(power - 1) 
            
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), prev=(self,), func="exp", requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += out.grad * out.data
            
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                     prev=(self,), 
                     func="sum",
                     requires_grad=self.requires_grad)
        
        def _backward():
            if keepdims:
                grad = out.grad
            else:
                grad = np.reshape(out.grad, out.grad.shape + (1,) * (self.data.ndim - out.grad.ndim))
            self.grad += np.ones_like(self.data) * grad
                
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)),
                     prev=(self,),
                     func="sigmoid",
                     requires_grad=self.requires_grad)
        
        def _backward():
            self.grad = out.grad * out.data * (np.ones_like(out.data) - out.data)
            
        out._backward = _backward
        return out
        
    
    def draw_computation_graph(self):
        from graphviz import Digraph

        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"}) # LR = left to right
        
        nodes, edges = set(), set()
        def traverse(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((v, child))
                    traverse(child)
                    
        traverse(self)
        
        for v in nodes:
            uid = str(id(v))
            
            label = f"{v.name} | data: {v.data}"
            if v.requires_grad:
                label +=  f" | grad: {v.grad}"
            dot.node(name=uid, label=label, shape="record")
            if v._func:
                dot.node(name=uid + v._func, label=v._func)
                dot.edge(uid + v._func, uid)
        
        for v, child in edges:
            # here we want to connect children to node's operation
            dot.edge(str(id(child)), str(id(v)) + v._func)
            
        return dot