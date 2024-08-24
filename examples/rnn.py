import deeplib as dl
import deeplib.nn as nn

print(dl.__version__)

# Example usage
input_size = 10
hidden_size = 20
output_size = 5
sequence_length = 15
batch_size = 32

model = nn.RNN(input_size, hidden_size, output_size)

x = dl.randn(batch_size, sequence_length, input_size)

# Forward pass
outputs, final_hidden = model(x)

print(f"Output shape: {outputs.shape}")
print(f"Final hidden state shape: {final_hidden.shape}")