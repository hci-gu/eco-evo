import argparse
import threading
import os
from lib.data_manager import data_loop, update_generations_data, process_data
from lib.constants import override_from_file
import lib.constants as const
from lib.runner import Runner
import time
import torch

if __name__ == "__main__":
    import pygame
    from lib.visualize import init_pygame, plot_generations, draw_world, plot_biomass

def load_config_files(config_folder):
    """Load all config files from the given folder."""
    config_files = []
    for file_name in os.listdir(config_folder):
        if file_name.endswith(".txt"):
            config_files.append(os.path.join(config_folder, file_name))
    return config_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ecosystem simulation with neural networks.")
    parser.add_argument("--config_folder", type=str, required=False,
                        help="Path to the folder containing config files.")
    parser.add_argument("--agent_file", type=str, required=False,
                        help="Path to the .pt file containing the agent model.")
    
    # Parse arguments
    args = parser.parse_args()

    if args.agent_file:
        print(f"Loading agent from file: {args.agent_file}")
        with torch.no_grad():
            runner = Runner()
            screen = init_pygame()
            def visualize(world, world_data, fitness):
                agents_data = process_data({
                    'agent_index': 0,
                    'eval_index': 0,
                    'step': fitness,
                    'world': world
                })
                draw_world(screen, world, world_data)
                plot_biomass(agents_data)
                pygame.display.flip()
                pygame.time.wait(1)
            runner.simulate(agent_file=args.agent_file, visualize=visualize)
        
    # Load config files
    config_files = load_config_files(args.config_folder)

    print("Running simulation with the following config files:" + str(config_files))

    with torch.no_grad():
        for config_file in config_files:
            override_from_file(config_file)
            time.sleep(1)
            runner = Runner()
            simulation_thread = threading.Thread(target=runner.run)
            simulation_thread.start()
            
            running = True
            while running and runner.current_generation < const.GENERATIONS_PER_RUN:
                # Handle Pygame events
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         running = False
                #         break
                # Process visualization data
                if hasattr(runner, 'data_queue'):
                    data_loop(runner.data_queue)

                if runner.generation_finished.is_set():
                    runner.next_generation()
                    generations_data = update_generations_data(runner.current_generation)
                    plot_generations(generations_data)

                    evaluation_thread = threading.Thread(target=runner.run)
                    evaluation_thread.start()

            simulation_thread.join()
            # cProfile.run('Runner().run(True)')

            
            # create_world()
            # start_time = time.time()
            # runs = 5
            # for i in range(runs):
            #     print(f"Run {i+1}/{runs}")
            #     Runner().run(True)
            # print(f"Average time: {(time.time() - start_time) / runs}")

    


