import numpy as np
import deeplib
import deeplib.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def check_param_update(model, old_params):
    updated = False
    for name, param in model.named_parameters():
        if not np.allclose(old_params[name], param.data):
            print(f"Parameter {name} updated")
            updated = True
    if not updated:
        print("No parameters were updated!")

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Gradient for {name} is None!")
        elif np.all(param.grad == 0):
            print(f"Gradient for {name} is all zeros!")
        else:
            print(f"Gradient for {name}: min={param.grad.min():.6f}, max={param.grad.max():.6f}, mean={param.grad.mean():.6f}")

if __name__ == "__main__":
    do_checks = False
    
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
        nn.Linear(8, 1),
    )
        
    criterion = nn.MSELoss()
    optimizer = deeplib.optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 100
    prev_loss = float('inf')
    for epoch in range(num_epochs):
        # Store old parameter values
        old_params = {name: param.data.copy() for name, param in model.named_parameters()}

        # Forward pass 
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
    
        # Backward pass
        optimizer.zero_grad()  # Move zero_grad before backward
        if do_checks:
            print(f"\nEpoch {epoch+1} - Checking gradients after zero_grad:")
            check_gradients(model)
        loss.backward()
        
        # Check gradients
        if do_checks:
            print(f"\nEpoch {epoch+1} - Checking gradients after backward:")
            check_gradients(model)
        
        optimizer.step()
        
        # Check if parameters were updated
        if do_checks:
            print(f"\nEpoch {epoch+1} - Checking parameter updates:")
            check_param_update(model, old_params)
        
        # Check if loss is changing
        if do_checks:
            if np.isclose(loss.item(), prev_loss):
                print(f"Warning: Loss is not changing significantly. Current: {loss.item():.6f}, Previous: {prev_loss:.6f}")
            prev_loss = loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    # Evaluation
    model.eval()
    with deeplib.no_grad():
        predicted = model(X_test)
        loss = criterion(predicted, y_test)
        print(f'Test Loss: {loss.item():.4f}')

    # Final model inspection
    print("\nFinal model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: min={param.data.min():.6f}, max={param.data.max():.6f}, mean={param.data.mean():.6f}")