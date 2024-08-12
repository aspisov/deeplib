import numpy as np
import deeplib
import deeplib.nn as nn
from sklearn.datasets import load_wine

data = load_wine()
X, y = deeplib.Tensor(data.data, requires_grad=True), deeplib.Tensor(data.target)

model = nn.Sequential(
    nn.Linear(13, 10),
    nn.Linear(10, 4),
    nn.Linear(4, 1)
)

for i in range(1):
    pred = model(X)
    loss = ((y - pred)**2).sum()
    print(loss.data)
    loss.backward()