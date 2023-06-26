import torch
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self):
        super.__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(400, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        X = self.flatten(X)
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.softmax(self.fc3(X))
        return X

