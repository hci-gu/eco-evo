import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(54, 200)
        self.fc2 = nn.Linear(200, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # use softmax to get probabilities
        x = torch.softmax(x, dim=1)
        return x