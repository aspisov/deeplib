from deeplib.tensor import Tensor
import deeplib
import numpy as np


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training:
        return input

    mask = np.random.randn(*input.shape) > p
    return input * mask / (1 - p)


def relu(input: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, input.data), _children=(input,), requires_grad=input.requires_grad)
    
    def _backward():
        if input.requires_grad:
            input.grad += out.grad * (input.data > 0)

    out._backward = _backward
    return out

def sigmoid(input: Tensor) -> Tensor:
    out = Tensor(1 / (1 + np.exp(-input.data)), _children=(input,), requires_grad=input.requires_grad)
    def _backward():
        if input.requires_grad:
            input.grad += out.grad * out.data * (1 - out.data)
    out._backward = _backward
    return out

def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return deeplib.mean((input - target) ** 2)

def cross_entropy_loss(logits: Tensor, target: Tensor) -> Tensor:
    N, C = logits.shape
    probs = softmax(logits, dim=1)
    nll = -deeplib.log(probs[np.arange(N), target.data.astype(int)])
    loss = deeplib.mean(nll)
    return loss

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable implementation of log_softmax"""
    x_diff = x - deeplib.max(x, dim=dim, keepdims=True)
    exp_tensor = deeplib.exp(x_diff)
    sum_exp_x_diff = deeplib.sum(exp_tensor, dim=dim, keepdims=True)
    return x_diff - deeplib.log(sum_exp_x_diff)

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable implementation of softmax"""
    exp_tensor = deeplib.exp(x - deeplib.max(x, dim=dim, keepdims=True))
    return exp_tensor / deeplib.sum(exp_tensor, dim=dim, keepdims=True)
