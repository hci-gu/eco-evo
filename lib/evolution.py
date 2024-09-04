from lib.model import Model
from typing import List, Tuple
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
    # Single-point crossover for weights and biases
    def crossover_layer(p1_W, p2_W):
        num_elements = p1_W.size
        crossover_point = np.random.randint(0, num_elements)
        
        c1_W = np.concatenate((p1_W.flatten()[:crossover_point], p2_W.flatten()[crossover_point:])).reshape(p1_W.shape)
        c2_W = np.concatenate((p2_W.flatten()[:crossover_point], p1_W.flatten()[crossover_point:])).reshape(p1_W.shape)
        return c1_W, c2_W

    c1_W0, c2_W0 = crossover_layer(p1_weights['W0'], p2_weights['W0'])
    c1_W1, c2_W1 = crossover_layer(p1_weights['W1'], p2_weights['W1'])
    c1_b0, c2_b0 = crossover_layer(p1_weights['b0'], p2_weights['b0'])
    c1_b1, c2_b1 = crossover_layer(p1_weights['b1'], p2_weights['b1'])
    
    return c1_W0, c2_W0, c1_b0, c2_b0, c1_W1, c2_W1, c1_b1, c2_b1

def mutation(weights, mutation_rate, mutation_scale):
    # Apply Gaussian mutation to weights and biases
    def mutate_layer(W):
        mutation_mask = np.random.rand(*W.shape) < mutation_rate
        noise = np.random.normal(0, mutation_scale, W.shape)
        W[mutation_mask] += noise[mutation_mask]
    
    mutate_layer(weights['W0'])
    mutate_layer(weights['W1'])
    mutate_layer(weights['b0'])
    mutate_layer(weights['b1'])