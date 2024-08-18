import numpy as np
import deeplib
import deeplib.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    
    
if __name__ == "__main__":
    data = load_wine()
    X, y = data.data, data.target.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = deeplib.FloatTensor(X_train)
    y_train = deeplib.FloatTensor(y_train)
    X_test = deeplib.FloatTensor(X_test)
    y_test = deeplib.FloatTensor(y_test)
        
    model = nn.Sequential(
        nn.Linear(13, 8),
        nn.ReLU(),
        # nn.BatchNorm1d(8),
        # nn.Dropout(0.5),
        nn.Linear(8, 1),
    )
        
    criterion = nn.MSELoss()
    optimizer = deeplib.optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        # forward pass 
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
    
        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    # evaluation
    model.eval()
    with deeplib.no_grad():
        predicted = model(X_test)
        loss = criterion(predicted, y_test)
        print(f'Loss: {loss.item():.4f}')
