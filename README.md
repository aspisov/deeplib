# DeepLib

DeepLib is a deep learning library built from scratch that closely follows PyTorch's API. It provides a lightweight framework for creating and training neural networks.

## Installation

To install DeepLib, run:

```bash
pip install git+https://github.com/aspisov/deeplib.git
```

## Currently Implemented Features

### Tensor Operations
- Most of PyTorch's tensor operations are supported. For details, see [deeplib/tensor.py](deeplib/tensor.py) & [deeplib/ops](deeplib/ops).

### Autograd
- Automatic differentiation for differentiable operations
- Computation graph building and backpropagation

### Layers (nn module)
- `Linear`: Fully connected layer
- `BatchNorm1d`: 1D batch normalization layer
- `Dropout`: Dropout layer for regularization
- `Embedding`: Embedding layer
- `RNN`: RNN layer

### Activation Functions (nn module)
- `ReLU`: Rectified Linear Unit activation
- `Sigmoid`: Sigmoid activation
- `Tanh`: Hyperbolic tangent activation

### Loss Functions (nn module)
- `MSELoss`: Mean Squared Error loss
- `CrossEntropyLoss`: Cross-entropy loss for classification tasks

### Optimizers (optim module)
- `SGD`: Stochastic Gradient Descent optimizer

### Initializers (nn.init module)
Most of the initializers in PyTorch are supported. For details, see [deeplib/nn/init.py](deeplib/nn/init.py).


## Quick Start

```python
import deeplib as dl
import deeplib.nn as nn
import deeplib.optim as optim

# Create a model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example training loop
for epoch in range(100):
    # Assuming we have inputs and targets
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

For more examples, see the [examples](examples) directory.

## Contributing

Contributions to DeepLib are welcome! If you have suggestions for improvements or have found a bug, please open an issue or submit a pull request.

## Future Developments

I am continuously working on expanding DeepLib's capabilities. Some planned features include:
- Additional layer types (e.g., Conv2d, MaxPool2d)
- More optimization algorithms (e.g., Adam)
- Data loading and preprocessing utilities

## License

DeepLib is released under the [MIT License](LICENSE).