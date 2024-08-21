import deeplib
from deeplib.nn.parameter import Parameter
from .module import Module
from deeplib.nn import init

__all__ = ["Embedding"]

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter(deeplib.empty((num_embeddings, embedding_dim)))
        self.reset_parameters()
        
    def reset_parameters(self):
        init.normal_(self.weight)
        
    def forward(self, input):
        return self.weight[input]