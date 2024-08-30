from lib.world import create_world, Species, Action
from lib.visualize import init_pygame, draw_world
from lib.model import Model
from time import sleep
from random import random
import math
import torch

species_order = [Species.PLANKTON, Species.ANCHOVY, Species.COD]

if __name__ == "__main__":
    world = create_world()
    screen = init_pygame()
    draw_world(screen, world)

    model = Model()

    draw_world(screen, world)
    sleep(3)

    steps = 0

    while True:
        species_order = sorted(species_order, key=lambda x: random())

        for species in species_order:
            cell = world[math.floor(random() * 10000)]

            # get tensor from cell
            cell_index = world.index(cell)
            x, y = cell_index // 100, cell_index % 100
            tensor = cell.toTensor3x3(x, y, world).view(1, -1)
            # get action probabilities
            action_values = model.forward(tensor)
            
            for action in Action:
                action_index = action.value
                action_probability = action_values[0, action_index].item()
                cell.perform_action(species, action, action_probability, x, y, world)
            
        steps += 1
        draw_world(screen, world)

        num_cod_alive = sum([cell.biomass[Species.COD] for cell in world])
        num_anchovy_alive = sum([cell.biomass[Species.ANCHOVY] for cell in world])
        num_plankton_alive = sum([cell.biomass[Species.PLANKTON] for cell in world])
        print(steps, num_cod_alive, num_anchovy_alive, num_plankton_alive)
            # # move entity to random adjacent cell
            # cell_index = world.index(cell)
            # x, y = cell_index // 100, cell_index % 100
            # dx, dy = math.floor(random() * 3) - 1, math.floor(random() * 3) - 1
            # new_x, new_y = x + dx, y + dy
            # if new_x < 0 or new_x >= 100 or new_y < 0 or new_y >= 100:
            #     continue
            # new_cell = world[new_x * 100 + new_y]

            # # can't move to land
            # if new_cell.terrain == Terrain.LAND:
            #     continue

            # if not new_cell.entity:
            #     new_cell.entity = cell.entity
            #     cell.entity = None
            
        

        # draw_world(screen, world)
    


