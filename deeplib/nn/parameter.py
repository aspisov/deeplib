from deeplib import Tensor

__all__ = ["Parameter"]

class Parameter(Tensor):
    
    def __init__(self, data, dtype=None):
        super().__init__(data, requires_grad=True, dtype=dtype)
        
    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()
    
    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0)