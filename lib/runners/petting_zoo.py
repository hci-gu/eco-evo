import lib.constants as const
from lib.model import Model
import lib.evolution as evolution  # Placeholder for your evolution functions
import gymnasium as gym
import torch
import copy
import random
from lib.environments.petting_zoo import env

class PettingZooRunner():
    def __init__(self):
        # Create the environment (with render_mode="human" for visualization)
        self.env = env(render_mode="none")
        self.env.reset()
        self.current_generation = 0
        
        # For training, we assume that only non-hardcoded species take actions.
        # Create a population (list of models) for each such species.
        self.species_list = [
            species for species, props in const.SPECIES_MAP.items()
            if not props["hardcoded_logic"]
        ]
        # Use a placeholder population size (NUM_AGENTS) per species.
        self.population = {
            species: [Model() for _ in range(const.NUM_AGENTS)]
            for species in self.species_list
        }
        
        # Dictionaries to track the best fitness and best model for each species.
        self.best_fitness = {species: -float('inf') for species in self.species_list}
        self.best_agent = {species: None for species in self.species_list}

    def run_generation(self):
        # Reset environment at the start of each generation.
        self.env.reset()
        
        # Placeholder fitness accumulators for each species.
        fitnesses = {species: 0 for species in self.species_list}
        
        # Iterate over agents in the environment.
        for agent in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()
            
            # If the agent is terminated or truncated, do nothing.
            if termination or truncation:
                action = None
                break
            else:
                # For species that are trained via the population,
                # choose a model from the corresponding population.
                if agent in self.population:
                    # In a real setup, you would process the observation with your model.
                    # For now, we use a placeholder random action.
                    # Example: action = model.forward(obs["observation"], agent)
                    action = self.env.action_space(agent).sample()
                else:
                    # For species with hardcoded logic (like plankton), use a fixed rule or random action.
                    action = self.env.action_space(agent).sample()
            
            # Step the environment using the batched action and positions from the observation.
            self.env.step({
                "action": action["action"] if action is not None else None,
                "positions": obs["positions"] if action is not None else None
            })
            
            # Optionally, render the environment (can be disabled during long training runs)
            self.env.render()
            
            # Accumulate rewards for fitness evaluation.
            # Here we assume the reward is a scalar; adjust as needed.
            if reward is not None and agent in fitnesses:
                fitnesses[agent] += reward
        
        # End of generation – update generation counter.
        self.current_generation += 1
        
        # Placeholder: Update the best fitness and perform evolutionary updates.
        for species in self.species_list:
            # If the current fitness is better, update best fitness and best agent.
            if fitnesses[species] > self.best_fitness[species]:
                self.best_fitness[species] = fitnesses[species]
                # In practice, you would store a copy of the best-performing model.
                self.best_agent[species] = random.choice(self.population[species])
            
            # Placeholder: Perform evolutionary operations (selection, crossover, mutation).
            # For example, you might call:
            #   new_population = evolution.evolve(self.population[species], fitnesses[species])
            # Here we simply reinitialize the population as a placeholder.
            self.population[species] = [Model() for _ in range(const.NUM_AGENTS)]
        
        print(f"Generation {self.current_generation} complete. Fitnesses: {fitnesses}")
    
    def train(self, generations=const.GENERATIONS_PER_RUN):
        for _ in range(generations):
            self.run_generation()
            # Optionally, save the best agents, log statistics, etc.

