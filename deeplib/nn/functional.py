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
    return ((input - target) ** 2).mean()


def cross_entropy_loss(input: Tensor, target: Tensor) -> Tensor:
    log_probs = log_softmax(input)
    return nll_loss(log_probs, target)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable implementation of log_softmax"""
    max_x = deeplib.max(x, dim=dim, keepdims=True)
    x_diff = x - max_x
    exp_x_diff = deeplib.exp(x_diff)
    sum_exp_x_diff = deeplib.sum(exp_x_diff, dim=dim, keepdims=True)
    return x_diff - deeplib.log(sum_exp_x_diff)


def nll_loss(input: Tensor, target: Tensor) -> Tensor:
    assert input.dim() > 1, "Input should be at least 2-dimensional"
    assert target.dim() == 1, "Target should be 1-dimensional"

    gather_indices = Tensor(
        target.data[:, None], requires_grad=False
    )  # Convert target indices to a column tensor
    log_probs = input.gather(dim=1, index=gather_indices)

    loss = -log_probs.mean()
    return loss
