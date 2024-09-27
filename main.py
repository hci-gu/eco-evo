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
    with torch.no_grad():
        runner.run()
        # runner.run_test_case()
    


