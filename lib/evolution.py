from lib.model import Model
from typing import List, Tuple
import torch
import numpy as np
import random

def elitism_selection(agents: List[Tuple[Model, int]], num_individuals: int) -> List[Tuple[Model, int]]:
    individuals = sorted(agents, key=lambda x: x[1], reverse=True)
    return individuals[:num_individuals]

def tournament_selection(agents: List[Tuple[Model, int]], num_individuals: int, tournament_size: int) -> List[Tuple[Model, int]]:
    selected_individuals = []
    
    for _ in range(num_individuals):
        tournament = random.sample(agents, tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        selected_individuals.append(winner)
    
    return selected_individuals

def crossover(p1_weights, p2_weights):
    """
    Perform single-point crossover on two sets of weights and biases (parents).
    p1_weights and p2_weights should be dictionaries (state_dict format).
    """
    def crossover_layer(p1_W, p2_W):
        num_elements = p1_W.size
        crossover_point = np.random.randint(0, num_elements)
        
        c1_W = np.concatenate((p1_W.flatten()[:crossover_point], p2_W.flatten()[crossover_point:])).reshape(p1_W.shape)
        c2_W = np.concatenate((p2_W.flatten()[:crossover_point], p1_W.flatten()[crossover_point:])).reshape(p1_W.shape)
        return c1_W, c2_W
    
    c1_weights = {}
    c2_weights = {}
    
    # Perform crossover for each layer in the parent weights
    for key in p1_weights.keys():
        c1_layer, c2_layer = crossover_layer(p1_weights[key].cpu().numpy(), p2_weights[key].cpu().numpy())
        c1_weights[key] = torch.tensor(c1_layer)
        c2_weights[key] = torch.tensor(c2_layer)
    
    return c1_weights, c2_weights


def mutation(weights, mutation_rate, mutation_scale):
    """
    Apply Gaussian mutation to a set of weights and biases.
    weights should be a dictionary (state_dict format).
    mutation_rate: Probability of mutating each weight/bias element.
    mutation_scale: Standard deviation of the Gaussian noise added during mutation.
    """
    def mutate_layer(W):
        mutation_mask = np.random.rand(*W.shape) < mutation_rate
        noise = np.random.normal(0, mutation_scale, W.shape)
        W[mutation_mask] += noise[mutation_mask]
    
    # Mutate each layer in the weights dictionary
    for key in weights.keys():
        W = weights[key].cpu().numpy()  # Convert to numpy for mutation
        mutate_layer(W)
        weights[key] = torch.tensor(W)  # Convert back to tensor
    
    return weights
