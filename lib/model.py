import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, layer_sizes=[81, 10, 10], chromosome=None):
        """
        layer_sizes: List of integers where each element represents the number of neurons in each layer,
                     including the input and output layers. For example, [54, 200, 10] means:
                     - 54 input neurons
                     - 200 neurons in the first hidden layer
                     - 10 output neurons
        chromosome: Optional dictionary containing pre-initialized weights and biases.
        """
        super(Model, self).__init__()
        
        self.layers = nn.ModuleList()  # Create a ModuleList to store layers
        
        # Dynamically create layers based on layer_sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        # If chromosome (weights and biases) is passed, initialize with those
        if chromosome:
            self.set_weights(chromosome)

    def forward(self, x, species):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        
        # Pass through the last layer without activation
        x = self.layers[-1](x)
        
        # Assuming the output layer is split for softmax activation for two different parts
        x[:, :5] = torch.softmax(x[:, :5], dim=1)
        x[:, 5:] = torch.softmax(x[:, 5:], dim=1)
        
        if species == 1:
            return x[:, :5]
        else:
            return x[:, 5:]
    
    def get_weights(self):
        """
        Retrieve the weights and biases as a chromosome (state_dict).
        This uses PyTorch's built-in `state_dict()` method, which returns all model parameters.
        """
        return self.state_dict()
    
    def set_weights(self, chromosome):
        """
        Set weights and biases from the chromosome (state_dict).
        This uses PyTorch's built-in `load_state_dict()` method, which loads parameters into the model.
        """
        self.load_state_dict(chromosome)

    def save_to_file(self, filename):
        """
        Save the model weights and biases to a file.
        """
        torch.save(self.state_dict(), filename)