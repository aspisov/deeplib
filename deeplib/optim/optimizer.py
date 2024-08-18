import deeplib

class Optimizer:
    def __init__(self, params, defaults):
        self.params = list(params)
        self.defaults = defaults

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0):
        super().__init__(params, {'lr': lr, 'momentum': momentum})
        self.momentum_buffers = [deeplib.zeros_like(p) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            momentum_buffer = self.momentum_buffers[i]

            if self.defaults['momentum'] > 0:
                momentum_buffer = momentum_buffer * self.defaults['momentum'] + grad
                grad = momentum_buffer
            
            param -= self.defaults['lr'] * grad
            
            self.momentum_buffers[i] = momentum_buffer