from enum import Enum
import opensimplex
import math
from random import random
import torch.nn as nn
import torch
from lib.constants import BASE_BIOMASS_LOSS, PLANKTON_GROWTH_RATE, MAX_PLANKTON_IN_CELL, WORLD_SIZE, STARTING_BIOMASS_ANCHOVY, STARTING_BIOMASS_COD, STARTING_BIOMASS_PLANKTON, MIN_PERCENT_ALIVE, EAT_AMOUNT

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

class Cell(object):
    def __init__(self, terrain):
        self.terrain = terrain
        self.biomass = {
            Species.PLANKTON: 0,
            Species.ANCHOVY: 0,
            Species.COD: 0
        }

    def plankton_growth(self):
        self.biomass[Species.PLANKTON] *= (1 + PLANKTON_GROWTH_RATE)
        self.biomass[Species.PLANKTON] = min(self.biomass[Species.PLANKTON], MAX_PLANKTON_IN_CELL)

    def perform_action(self, species, action, action_value, x, y, world):
        # species performing, action and world
        if action == Action.EAT:
            eat_amount = (self.biomass[species] * EAT_AMOUNT) * action_value
            if species == Species.COD:
                anchovy_available = self.biomass[Species.ANCHOVY]
                eat_amount = min(eat_amount, anchovy_available)
                self.biomass[Species.ANCHOVY] -= eat_amount
                self.biomass[species] += eat_amount
            elif species == Species.ANCHOVY:
                plankton_available = self.biomass[Species.PLANKTON]
                eat_amount = min(eat_amount, plankton_available)
                self.biomass[Species.PLANKTON] -= eat_amount
                self.biomass[species] += eat_amount
        else:
            dx, dy = delta_for_action[action]
            new_x, new_y = x + dx, y + dy
            if new_x < 0 or new_x >= WORLD_SIZE or new_y < 0 or new_y >= WORLD_SIZE:
                self.biomass[species] *= (1 - BASE_BIOMASS_LOSS)
                return
            new_cell = world[new_x * WORLD_SIZE + new_y]
            if new_cell.terrain == Terrain.LAND:
                self.biomass[species] *= (1 - BASE_BIOMASS_LOSS)
                return
            
            # biomass movement
            move_amount = self.biomass[species] * action_value
            self.biomass[species] -= move_amount
            self.biomass[species] = max(self.biomass[species], 0)
            
            new_cell.biomass[species] += (move_amount * (1 - BASE_BIOMASS_LOSS))


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
                if new_x < 0 or new_x >= WORLD_SIZE or new_y < 0 or new_y >= WORLD_SIZE:
                    tensor.append(torch.zeros(6))
                else:
                    cell = world[new_x * WORLD_SIZE + new_y]
                    tensor.append(cell.to_tensor())
        return torch.stack(tensor)

def create_world():
    NOISE_SCALING = 4.5
    # # random seed each time
    opensimplex.seed(int(random() * 100000))

    world = []
    water_cells = []  # List to store water cells and their coordinates for biomass distribution
    center_x, center_y = WORLD_SIZE // 2, WORLD_SIZE // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)

    # Define total biomass for each species
    total_biomass_cod = STARTING_BIOMASS_COD  # Example: total biomass for cod
    total_biomass_anchovy = STARTING_BIOMASS_ANCHOVY  # Example: total biomass for anchovy
    total_biomass_plankton = STARTING_BIOMASS_PLANKTON  # Example: total biomass for plankton

    # Variables to store the noise sum for each species
    noise_sum_cod = 0
    noise_sum_anchovy = 0
    noise_sum_plankton = 0

    # First pass: create world and calculate the noise sum for normalization
    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            distance_factor = distance / max_distance
            noise_value = opensimplex.noise2(x * 0.15, y * 0.15)
            threshold = 0.6 * distance_factor + 0.1 * noise_value
            terrain = Terrain.WATER if threshold < 0.5 else Terrain.LAND
            cell = Cell(terrain)
            
            if terrain == Terrain.WATER:
                # Store the water cell and its coordinates
                water_cells.append((cell, x, y))

                # Calculate noise values for clustering species
                # Modify noise function to make it more clustered and steeper
                noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2  # Scale noise to range [0, 1]
                noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
                noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

                # Apply thresholds to zero-out biomass in certain regions
                if noise_cod < 0.4:  # Low noise means no cod
                    noise_cod = 0
                if noise_anchovy < 0.3:  # Low noise means no anchovy
                    noise_anchovy = 0
                if noise_plankton < 0.35:  # Low noise means no plankton
                    noise_plankton = 0
                
                # Apply more aggressive scaling to create larger biomass in high-noise areas
                noise_cod = noise_cod ** NOISE_SCALING  # Steeper scaling for more clustering
                noise_anchovy = noise_anchovy ** NOISE_SCALING
                noise_plankton = noise_plankton ** NOISE_SCALING

                # Sum noise values for normalization in the next step
                noise_sum_cod += noise_cod
                noise_sum_anchovy += noise_anchovy
                noise_sum_plankton += noise_plankton
                
            world.append(cell)

    # Second pass: distribute biomass based on the ratio of noise values to total noise sum
    for cell, x, y in water_cells:
        # Recalculate noise values for each species
        noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2
        noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
        noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

        # Apply thresholds again
        if noise_cod < 0.4:
            noise_cod = 0
        if noise_anchovy < 0.3:
            noise_anchovy = 0
        if noise_plankton < 0.35:
            noise_plankton = 0

        # Steeper scaling for higher concentration in certain areas
        noise_cod = noise_cod ** NOISE_SCALING
        noise_anchovy = noise_anchovy ** NOISE_SCALING
        noise_plankton = noise_plankton ** NOISE_SCALING

        # Ensure the total biomass is distributed proportionally based on noise values
        if noise_sum_cod > 0:
            cell.biomass[Species.COD] = (noise_cod / noise_sum_cod) * total_biomass_cod
        if noise_sum_anchovy > 0:
            cell.biomass[Species.ANCHOVY] = (noise_anchovy / noise_sum_anchovy) * total_biomass_anchovy
        if noise_sum_plankton > 0:
            cell.biomass[Species.PLANKTON] = (noise_plankton / noise_sum_plankton) * total_biomass_plankton

    return world


def world_is_alive(world):
    cod_alive = sum([cell.biomass[Species.COD] for cell in world])
    if cod_alive < (STARTING_BIOMASS_COD * MIN_PERCENT_ALIVE):
        return False
    anchovy_alive = sum([cell.biomass[Species.ANCHOVY] for cell in world])
    if anchovy_alive < (STARTING_BIOMASS_ANCHOVY * MIN_PERCENT_ALIVE):
        return False
    plankton_alive = sum([cell.biomass[Species.PLANKTON] for cell in world])
    if plankton_alive < (STARTING_BIOMASS_PLANKTON * MIN_PERCENT_ALIVE):
        return False
    
    return True