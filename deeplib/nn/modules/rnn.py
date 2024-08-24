import deeplib as dl
from .module import Module
from .linear import Linear
import deeplib.nn.functional as F

__all__ = ["RNN"]

class RNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # input to hidden
        self.wx = Linear(input_size, hidden_size)
        
        # hidden to hidden
        self.wh = Linear(hidden_size, hidden_size)
        
        # hidden to output
        self.wy = Linear(hidden_size, output_size)
        
        self.activation = F.tanh
                
    def forward(self, x, h_0=None):
        batch_size, sequence_length, input_size = x.shape
        
        if h_0 is None:
            h_0 = dl.zeros(batch_size, self.hidden_size)
        
        outputs = []
        h_t = h_0
        
        for t in range(sequence_length):
            # input at current time step
            x_t = x[:, t, :]
            
            # new hidden state
            print(f"h_t shape: {h_t.shape}")
            print(f"x_t shape: {x_t.shape}")
            h_t = self.activation(self.wh(h_t) + self.wx(x_t))

            # output at current time step
            y_t = self.wy(h_t)
            
            outputs.append(y_t)
            
        outputs = dl.stack(outputs, dim=1)
        return outputs, h_t