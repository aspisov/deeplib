import numpy as np
import deeplib
import deeplib.nn as nn
from sklearn.datasets import load_wine
    
data = load_wine()
X = deeplib.Tensor(data.data, dtype=np.float32)
y = deeplib.Tensor(data.target.reshape(-1, 1), dtype=np.float32)
    
model = nn.Sequential(
    nn.Linear(13, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
    
optim = deeplib.optim.SGD(model.parameters(), lr=0.01)
for i in range(100):
    pred = model(X)
    loss = ((y - pred) ** 2).mean()

    loss.backward()
    
    optim.step()
    optim.zero_grad()

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.data}")
