from deeplib.nn.parameter import Parameter
from typing import Iterator, Tuple

__all__ = ["Module"]

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
        
    def register_parameter(self, name: str, param: Parameter):
        if not isinstance(param, Parameter):
            raise TypeError(f"Parameter must be an instance of Parameter. Got {type(param)}")
        self._parameters[name] = param
        
    def register_module(self, name: str, module: "Module"):
        if not isinstance(module, Module):
            raise TypeError(f"Module must be an instance of Module. Got {type(module)}")
        self._modules[name] = module
    
    def forward(self, input):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> Iterator[Parameter]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param
            
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
        
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        object.__setattr__(self, name, value)