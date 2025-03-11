import argparse
import threading
import os
from lib.visualize import plot_generations, init_pygame, draw_world, plot_biomass
import pygame
from lib.data_manager import data_loop, update_generations_data, process_data
from lib.constants import override_from_file
# from lib.runners.single_agent_gym import SingleAgentGymRunner
# from lib.runners.petting_zoo_reinforcement_old import RLRunner
import lib.environments.gym
from lib.runners.petting_zoo import PettingZooRunner
from lib.runners.reinforcement import RLRunner
from lib.runners.single_agent_gym import SingleAgentGymRunner
import lib.constants as const
from lib.runner import Runner


def load_config_files(config_folder):
    """Load all config files from the given folder."""
    config_files = []
    for file_name in os.listdir(config_folder):
        if file_name.endswith(".txt"):
            config_files.append(os.path.join(config_folder, file_name))
    config_files.sort()
    return config_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ecosystem simulation with neural networks.")
    parser.add_argument("--config_folder", type=str, required=False,
                        help="Path to the folder containing config files.")
    parser.add_argument("--agent_file", type=str, required=False,
                        help="Path to the .pt file containing the agent model.")
    
    # Parse arguments
    args = parser.parse_args()

    folder = "results/petting_zoo_19/agents"
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[2].split(".")[0]), reverse=True)
    species = {}
    for f in files:
        s = f.split("_")[1].split(".")[0]
        # remove $ from species name
        s = s[1:] if s[0] == "$" else s
        if s not in species:
            species[s] = f
    
    model_paths = []
    for s, f in species.items():
        model_paths.append({ 'path': os.path.join(folder, f), 'species': s })

    print("START EVAL")
    runner = PettingZooRunner(render_mode="human")
    runner.evaluate(model_paths)
    print("EVAL DONE")

    # if args.agent_file:
    #     print(f"Loading agent from file: {args.agent_file}")
    #     runner = Runner()
    #     screen = init_pygame()
    #     def visualize(world, world_data, fitness):
    #         agents_data = process_data({
    #             'agent_index': 0,
    #             'eval_index': 0,
    #             'step': fitness,
    #             'world': world
    #         })
    #         draw_world(screen, world, world_data)
    #         plot_biomass(agents_data)
    #         pygame.display.flip()
    #         pygame.time.wait(1)
    #     runner.simulate(agent_file=args.agent_file, visualize=visualize)

    # Load config files
    config_files = load_config_files(args.config_folder)

    print("Running simulation with the following config files:" + str(config_files))

    for config_file in config_files:
        override_from_file(config_file)

        if const.RUNNER == "rl_runner":
            runner = RLRunner()
            runner.train()
        elif const.RUNNER == "petting_zoo":
            runner = PettingZooRunner()
            runner.train()
        elif const.RUNNER == "gym":
            runner = SingleAgentGymRunner()
            runner.train()
        elif const.RUNNER == "old":
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



        


    

