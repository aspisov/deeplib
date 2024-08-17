import numpy as np
import deeplib
import deeplib.nn as nn
from sklearn.datasets import load_wine
    
    
if __name__ == "__main__":
    data = load_wine()
    X_train = deeplib.Tensor(data.data, dtype=np.float32)
    y_train = deeplib.Tensor(data.target.reshape(-1, 1), dtype=np.float32)
        
    model = nn.Sequential(
        nn.Linear(13, 8),
        nn.ReLU(),
        nn.BatchNorm1d(8),
        nn.Dropout(0.5),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
        
    criterion = nn.MSELoss()
    optimizer = deeplib.optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        # forward pass 
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
