import torch
import torch.nn as nn
import lib.constants as const
import numpy as np

class Model(nn.Module):
    def __init__(self, layer_sizes=[const.NETWORK_INPUT_SIZE, const.NETWORK_HIDDEN_SIZE, const.NETWORK_OUTPUT_SIZE], chromosome=None):
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
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        
        x = self.layers[-1](x)
        
        # Depending on species, compute softmax only on the required outputs
        if species == 1:
            output = torch.softmax(x[:, :5], dim=1)
        else:
            output = torch.softmax(x[:, 5:], dim=1)
        
        return output
        
    def debug_move_down_and_eat(self, x, species):

        # random value between 0 and 1
        # eat_value = np.random.rand()
        eat_value = 0.75
        move_value = 1 - eat_value

        # random direction (index between 0 and 3)
        random_direction = np.random.randint(0, 4)

        base_tensor = torch.tensor([[0, 0, 0, 0, eat_value]])

        # set move index to move value
        base_tensor[0, random_direction] = move_value

        return base_tensor
        

        
    
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