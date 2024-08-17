from .tensor import *
from .tensor_ops import *
from .autograd import *
from . import nn
from . import optim
from . import utils

__version__ = "0.0.1"

__all__ = ["tensor", "autograd", "nn", "optim", "utils"]

# overload basic operations
Tensor.__add__ = add
Tensor.__iadd__ = add
Tensor.__radd__ = add
Tensor.__neg__ = neg
Tensor.__sub__ = sub
Tensor.__mul__ = mul
Tensor.__rmul__ = mul
Tensor.__truediv__ = true_divide
Tensor.__matmul__ = matmul
Tensor.__pow__ = pow
Tensor.sqrt = sqrt

# math operations
Tensor.exp = exp
Tensor.log = log
Tensor.sum = sum
Tensor.mean = mean
Tensor.var = var
Tensor.max = max