from enum import Enum
import opensimplex
import math
from random import random
import torch.nn as nn
import torch

class Terrain(Enum):
    LAND = 0
    WATER = 1
    OUT_OF_BOUNDS = 2

class Species(Enum):
    PLANKTON = 0
    ANCHOVY = 1
    COD = 2

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

delta_for_action = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1)
}

MOVEMENT_COST = 0.1

class Cell(object):
    def __init__(self, terrain):
        self.terrain = terrain
        self.biomass = {
            Species.PLANKTON: 0,
            Species.ANCHOVY: 0,
            Species.COD: 0
        }

    def perform_action(self, species, action, action_value, x, y, world):
        if self.biomass[species] == 0:
            return
        # species performing, action and world
        if action == Action.EAT:
            eat_amount = self.biomass[species] * action_value
            if species == Species.COD:
                anchovy_amount = self.biomass[Species.ANCHOVY]
                self.biomass[Species.ANCHOVY] -= eat_amount
                anchovy_eaten = max(anchovy_amount - self.biomass[Species.ANCHOVY], 0)
                self.biomass[Species.COD] += anchovy_eaten
            elif species == Species.ANCHOVY:
                plankton_amount = self.biomass[Species.PLANKTON]
                self.biomass[Species.PLANKTON] -= eat_amount
                plankton_eaten = max(plankton_amount - self.biomass[Species.PLANKTON], 0)
                self.biomass[Species.ANCHOVY] += plankton_eaten
        else:
            dx, dy = delta_for_action[action]
            new_x, new_y = x + dx, y + dy
            if new_x < 0 or new_x >= 100 or new_y < 0 or new_y >= 100:
                return
            new_cell = world[new_x * 100 + new_y]
            if new_cell.terrain == Terrain.LAND:
                return
            
            self.biomass[species] -= action_value
            self.biomass[species] = max(self.biomass[species], 0)
            new_cell.biomass[species] += (action_value - MOVEMENT_COST)

    def to_tensor(self):
        terrain_tensor = torch.tensor(self.terrain.value)
        land = nn.functional.one_hot(terrain_tensor, num_classes=3)
        
        biomass_values = [self.biomass[Species.PLANKTON], self.biomass[Species.ANCHOVY], self.biomass[Species.COD]]
        biomass_tensor = torch.tensor(biomass_values, dtype=torch.float32)

        return torch.cat((land, biomass_tensor))
    
    def toTensor3x3(self, x, y, world):
        tensor = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_x, new_y = x + i, y + j
                if new_x < 0 or new_x >= 100 or new_y < 0 or new_y >= 100:
                    tensor.append(torch.zeros(6))
                else:
                    cell = world[new_x * 100 + new_y]
                    tensor.append(cell.to_tensor())
        return torch.stack(tensor)

def create_world():
    size = 100
    opensimplex.seed(105)

    world = []
    center_x, center_y = size // 2, size // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)
    for x in range(size):
        for y in range(size):
            # Calculate the distance from the center
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            distance_factor = distance / max_distance

            # Adjust noise scaling for smoother transitions and a larger central water body
            noise_value = opensimplex.noise2(x * 0.15, y * 0.15)
            
            # Modify threshold to create a bay shape with a larger water body
            # Stronger influence of distance_factor for a clear bay shape
            threshold = 0.6 * distance_factor + 0.1 * noise_value
            
            # More water in the center, transitioning to land at the edges
            terrain = Terrain.WATER if threshold < 0.5 else Terrain.LAND
            cell = Cell(terrain)
            if terrain == Terrain.WATER:
                if random() < 0.01:
                    cell.biomass[Species.COD] = 1
                if random() < 0.025:
                    cell.biomass[Species.ANCHOVY] = 1
                if random() < 0.1:
                    cell.biomass[Species.PLANKTON] = 1
            
            world.append(cell)
    
    return world


            