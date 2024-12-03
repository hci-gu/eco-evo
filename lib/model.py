import torch
import torch.nn as nn
import lib.constants as const

class Model(nn.Module):
    def __init__(self, input_size=const.NETWORK_INPUT_SIZE, hidden_size=const.NETWORK_HIDDEN_SIZE, output_size=const.NETWORK_OUTPUT_SIZE, chromosome=None):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        if chromosome:
            self.set_weights(chromosome)

    def forward(self, x: torch.Tensor, species_key: str) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        if species_key == "anchovy":
            output = torch.softmax(x[:, :5], dim=1)
        else:
            output = torch.softmax(x[:, 5:], dim=1)
        return output
        
    def set_weights(self, chromosome):
        self.load_state_dict(chromosome)