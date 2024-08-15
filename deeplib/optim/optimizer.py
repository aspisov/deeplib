class Optimizer:
    def __init__(self, params, defaults):
        self.params = params
        self.defaults = defaults
        
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
            
    def step(self):
        raise NotImplemented
    

class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, {'lr': lr})
    
    def step(self):
        for param in self.params:
            param.data -= self.defaults['lr'] * param.grad