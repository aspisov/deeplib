from deeplib.tensor import Tensor


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
