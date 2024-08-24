import deeplib as dl
import deeplib.nn as nn
import deeplib.optim as optim
from deeplib.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_test = dl.FloatTensor(X_test)
y_test = dl.LongTensor(y_test)

class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = dl.FloatTensor(X)
        self.y = dl.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
batch_size = 4

dataset = IrisDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batchnorm(x)
        x = self.fc2(x)
        return x

model = IrisClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (X_batch, y_batch) in enumerate(loader):
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch: {epoch + 1}, Step: {i + 1}, Loss: {epoch_loss / (i + 1):.4f}")

# evaluation
model.eval()
with dl.no_grad():
    predicted = model(X_test)
    predicted = predicted.argmax(dim=1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")

# print classification report
from sklearn.metrics import classification_report

print(classification_report(y_test.data, predicted.data))
