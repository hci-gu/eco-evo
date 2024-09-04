from lib.world import create_world, world_is_alive, Species, Action
from lib.visualize import init_pygame, draw_world, plot_biomass
from lib.model import Model
from lib.constants import WORLD_SIZE
from lib.runner import Runner
import time
import random
import math
import torch


if __name__ == "__main__":
    runner = Runner()
    
    runner.run()
            
        # batch_steps += 1
        # # if batch_steps % 10 == 0:
        # plot_biomass(screen, world, batch_steps)
        # draw_world(screen, world)

        # time_ended = time.time()
        # seconds_elapsed = time_ended - time_started
        # print(f"Time taken: {seconds_elapsed} seconds")
        # draw_world(screen, world)
    


