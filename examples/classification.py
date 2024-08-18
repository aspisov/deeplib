import deeplib
import deeplib.nn as nn
import deeplib.optim as optim
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

X_train = deeplib.FloatTensor(X_train)
y_train = deeplib.LongTensor(y_train)
X_test = deeplib.FloatTensor(X_test)
y_test = deeplib.LongTensor(y_test)


class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IrisClassifier()
for name, param in model.named_parameters():
    print(name, param.shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted = model(X_train)
    loss = criterion(predicted, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# evaluation
model.eval()
with deeplib.no_grad():
    predicted = model(X_test)
    predicted = predicted.argmax(dim=1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")

# print classification report
from sklearn.metrics import classification_report

print(classification_report(y_test.data, predicted.data))
