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