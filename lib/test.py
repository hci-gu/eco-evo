import torch
import torch.nn as nn
import lib.constants as const

def create_test_world_3x3():
    """
    Creates a 3x3 test world with predefined values for biomass, energy, and terrain.
    Useful for testing the simulation logic with small, manageable data.
    """
    # Create a tensor to represent the 3x3 world
    world_tensor = torch.zeros(3, 3, 9)  # Assuming 9 channels as before

    # Set predefined terrain (1 for water, 0 for land)
    # Let's say the center is land, and the rest is water
    world_tensor[0, 0, :3] = torch.tensor([0, 1, 0])  # Water
    world_tensor[0, 1, :3] = torch.tensor([0, 1, 0])  # Water
    world_tensor[0, 2, :3] = torch.tensor([0, 1, 0])  # Water
    world_tensor[1, 0, :3] = torch.tensor([0, 1, 0])  # Water
    world_tensor[1, 1, :3] = torch.tensor([0, 1, 0])  # Water
    world_tensor[1, 2, :3] = torch.tensor([0, 1, 0])  # Land
    world_tensor[2, 0, :3] = torch.tensor([0, 1, 0])  # Water
    world_tensor[2, 1, :3] = torch.tensor([0, 1, 0])  # Water
    world_tensor[2, 2, :3] = torch.tensor([0, 1, 0])  # Land

    for x in range(3):
        for y in range(3):
            world_tensor[x, y, const.OFFSETS_ENERGY_ANCHOVY] = 50
            world_tensor[x, y, const.OFFSETS_BIOMASS_ANCHOVY] = 50

    # Set predefined biomass and energy for each species in the world
    # Assume some arbitrary values for plankton, anchovy, and cod
    world_tensor[1, 1, const.OFFSETS_BIOMASS_ANCHOVY] = 100
    world_tensor[1, 1, const.OFFSETS_ENERGY_ANCHOVY] = 50
    world_tensor[0, 1, const.OFFSETS_BIOMASS_ANCHOVY] = 100
    world_tensor[0, 1, const.OFFSETS_ENERGY_ANCHOVY] = 50
    world_tensor[1, 0, const.OFFSETS_BIOMASS_ANCHOVY] = 10
    world_tensor[1, 0, const.OFFSETS_ENERGY_ANCHOVY] = 100
    
    world_tensor[1, 1, const.OFFSETS_ENERGY_COD] = 75
    world_tensor[1, 1, const.OFFSETS_BIOMASS_COD] = 50

    # world_tensor[0, 1, const.OFFSETS_BIOMASS_ANCHOVY] = 50
    # world_tensor[0, 1, const.OFFSETS_ENERGY_ANCHOVY] = 100

    # # Similar for other cells as needed
    # # Filling some default biomass and energy for other species and cells
    # world_tensor[1, 0, const.OFFSETS_BIOMASS_PLANKTON] = 5
    # world_tensor[1, 0, const.OFFSETS_ENERGY_PLANKTON] = 60
    # world_tensor[2, 2, const.OFFSETS_BIOMASS_COD] = 15
    # world_tensor[2, 2, const.OFFSETS_ENERGY_COD] = 90

    return world_tensor


# direction_to_tensor(direction):
#     switch(direction):


class MockModel(nn.Module):
    def __init__(self, layer_sizes=[81, 18, 10]):
        super(MockModel, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x, species):
        # for i, layer in enumerate(self.layers[:-1]):
        #     x = torch.relu(layer(x))
        
        # # Pass through the last layer without activation
        # x = self.layers[-1](x)
        
        print('species:', species)
        if species == 1:
            return torch.tensor([[1, 0, 0, 0, 0.0]])
        else:
            return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        

