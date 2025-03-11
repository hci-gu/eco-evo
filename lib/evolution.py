from lib.model import Model
from typing import List, Tuple
import numpy as np
import random

def elitism_selection(agents: List[Tuple[Model, int]], num_individuals: int) -> List[Tuple[Model, int]]:
    """Select the top individuals based on fitness."""
    individuals = sorted(agents, key=lambda x: x[1], reverse=True)
    return individuals[:num_individuals]

def tournament_selection(agents: List[Tuple[Model, int]], num_individuals: int, tournament_size: int) -> List[Tuple[Model, int]]:
    """Perform tournament selection and return the winners."""
    selected_individuals = []
    for _ in range(num_individuals):
        tournament = random.sample(agents, tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        selected_individuals.append(winner)
    return selected_individuals

def crossover(p1_weights: dict, p2_weights: dict) -> Tuple[dict, dict]:
    """
    Perform single-point crossover on two sets of weights and biases (parents).
    p1_weights and p2_weights should be dictionaries (state_dict format) containing NumPy arrays.
    """
    def crossover_layer(p1_W: np.ndarray, p2_W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_elements = p1_W.size
        crossover_point = np.random.randint(0, num_elements)
        
        flat1 = p1_W.flatten()
        flat2 = p2_W.flatten()
        c1_flat = np.concatenate((flat1[:crossover_point], flat2[crossover_point:]))
        c2_flat = np.concatenate((flat2[:crossover_point], flat1[crossover_point:]))
        c1_W = c1_flat.reshape(p1_W.shape)
        c2_W = c2_flat.reshape(p1_W.shape)
        return c1_W, c2_W

    c1_weights = {}
    c2_weights = {}
    # Perform crossover for each layer.
    for key in p1_weights.keys():
        c1_layer, c2_layer = crossover_layer(p1_weights[key], p2_weights[key])
        c1_weights[key] = c1_layer
        c2_weights[key] = c2_layer

    return c1_weights, c2_weights

def mutation(weights: dict, mutation_rate: float, mutation_scale: float) -> dict:
    """
    Apply Gaussian mutation to a set of weights and biases.
    weights should be a dictionary (state_dict format) of NumPy arrays.
    mutation_rate: Probability of mutating each element.
    mutation_scale: Standard deviation of the Gaussian noise added during mutation.
    """
    def mutate_layer(W: np.ndarray):
        mutation_mask = np.random.rand(*W.shape) < mutation_rate
        noise = np.random.normal(0, mutation_scale, W.shape)
        W[mutation_mask] += noise[mutation_mask]
    
    for key in weights.keys():
        mutate_layer(weights[key])
    
    return weights
