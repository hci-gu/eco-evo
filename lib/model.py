import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, chromosome=None):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(54, 200)
        self.fc2 = nn.Linear(200, 10)
        
        # If chromosome (weights and biases) is passed, initialize with those
        if chromosome:
            self.set_weights(chromosome)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x[:, :5] = torch.softmax(x[:, :5], dim=1)
        x[:, 5:] = torch.softmax(x[:, 5:], dim=1)
        return x
    
    def get_weights(self):
        # Retrieve the weights and biases as chromosome (dict)
        weights = {
            'W0': self.fc1.weight.detach().cpu().numpy(),
            'b0': self.fc1.bias.detach().cpu().numpy(),
            'W1': self.fc2.weight.detach().cpu().numpy(),
            'b1': self.fc2.bias.detach().cpu().numpy(),
        }
        return weights
    
    def set_weights(self, chromosome):
        # Set weights and biases from the chromosome (dict)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(chromosome['W0']))
            self.fc1.bias.copy_(torch.tensor(chromosome['b0']))
            self.fc2.weight.copy_(torch.tensor(chromosome['W1']))
            self.fc2.bias.copy_(torch.tensor(chromosome['b1']))