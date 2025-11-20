import numpy as np
from lib.model import OUTPUT_SIZE, MODEL_OFFSETS

def get_heuristic_action(agent, world, species_map, settings):
    """
    Generates a heuristic action grid for the given agent (species).
    Heuristic: If there is prey in the cell, EAT. Otherwise, move randomly.
    
    Args:
        agent: The species name (str).
        world: The padded world state (H, W, C).
        species_map: Dictionary of species properties.
        settings: Settings object.
        
    Returns:
        actions_grid: (H-2, W-2, OUTPUT_SIZE) action grid matching the inner world size.
                      Wait, the env expects (W, W, OUTPUT_SIZE) which matches settings.world_size.
                      The world passed in is padded (W+2, W+2).
    """
    
    # 1. Identify prey channels
    prey_list = species_map[agent].prey
    prey_channels = [MODEL_OFFSETS[p]["biomass"] for p in prey_list]
    
    # 2. Check for prey presence in the inner world grid
    # world is (W+2, W+2, C)
    # inner_world is (W, W, C)
    inner_world = world[1:-1, 1:-1]
    
    # Sum biomass of all prey types
    if prey_channels:
        prey_biomass = np.sum(inner_world[..., prey_channels], axis=-1)
        can_eat = prey_biomass > 0.1 # Threshold to bother eating
    else:
        can_eat = np.zeros((settings.world_size, settings.world_size), dtype=bool)
    
    # 3. Construct action grid
    # We generate actions for the inner grid size (W, W)
    heuristic_actions = np.zeros((settings.world_size, settings.world_size, OUTPUT_SIZE), dtype=np.float32)
    
    # --- Simplified Logic: Sustainable Eating + Smell Navigation ---
    
    # Thresholds
    # Lower hunger threshold to prevent over-eating
    HUNGRY_THRESHOLD = settings.max_energy * 0.4 
    EAT_THRESHOLD = 0.1 # Eat anything available
    
    # Agent State
    energy_offset = MODEL_OFFSETS[agent]["energy"]
    agent_energy = inner_world[..., energy_offset]
    is_hungry = agent_energy < HUNGRY_THRESHOLD
    
    # Food Map
    if prey_channels:
        prey_map = np.sum(inner_world[..., prey_channels], axis=-1)
        full_prey_map = np.sum(world[..., prey_channels], axis=-1)
    else:
        prey_map = np.zeros((settings.world_size, settings.world_size), dtype=np.float32)
        full_prey_map = np.zeros((settings.world_size + 2, settings.world_size + 2), dtype=np.float32)

    # Smell Map
    if prey_channels:
        prey_smell_channels = [MODEL_OFFSETS[p]["smell"] for p in prey_list]
        full_smell_map = np.sum(world[..., prey_smell_channels], axis=-1)
    else:
        full_smell_map = np.zeros((settings.world_size + 2, settings.world_size + 2), dtype=np.float32)

    # --- Movement ---
    
    # 1. Identify Valid Moves (Water)
    water_channel = MODEL_OFFSETS["terrain"]["water"]
    full_water_map = world[..., water_channel] # (W+2, W+2)
    
    up_water = full_water_map[0:-2, 1:-1]
    down_water = full_water_map[2:, 1:-1]
    left_water = full_water_map[1:-1, 0:-2]
    right_water = full_water_map[1:-1, 2:]
    
    # Stack valid masks: (W, W, 4)
    # 1.0 means water (valid), 0.0 means land/oob (invalid)
    valid_moves = np.stack([up_water, down_water, left_water, right_water], axis=-1) > 0.5
    
    # 2. Calculate Gradients
    up_food = full_prey_map[0:-2, 1:-1]
    down_food = full_prey_map[2:, 1:-1]
    left_food = full_prey_map[1:-1, 0:-2]
    right_food = full_prey_map[1:-1, 2:]
    
    neighbor_food = np.stack([up_food, down_food, left_food, right_food], axis=-1)
    
    up_smell = full_smell_map[0:-2, 1:-1]
    down_smell = full_smell_map[2:, 1:-1]
    left_smell = full_smell_map[1:-1, 0:-2]
    right_smell = full_smell_map[1:-1, 2:]
    
    neighbor_smell = np.stack([up_smell, down_smell, left_smell, right_smell], axis=-1)
    
    # Mask invalid moves (set value to -1 so they aren't picked)
    neighbor_food[~valid_moves] = -1.0
    neighbor_smell[~valid_moves] = -1.0
    
    # 3. Determine Best Directions
    max_food = np.max(neighbor_food, axis=-1)
    best_food_dir = np.argmax(neighbor_food, axis=-1)
    
    max_smell = np.max(neighbor_smell, axis=-1)
    best_smell_dir = np.argmax(neighbor_smell, axis=-1)
    
    has_food_nearby = max_food > 0.1
    
    # 4. Random Valid Move
    # This is tricky to vectorize efficiently without picking invalid.
    # We can pick a random index 0..3, check if valid. If not, pick another?
    # Or just pick argmax of random * valid_mask?
    random_scores = np.random.rand(settings.world_size, settings.world_size, 4)
    random_scores[~valid_moves] = -1.0
    random_best_dir = np.argmax(random_scores, axis=-1)
    
    # 5. Decision Logic
    # Default: Random Valid
    final_moves = random_best_dir
    
    # If Hungry:
    # - Follow Smell (if valid smell exists)
    # - Override with Food (if valid food exists)
    
    has_smell = max_smell > 0.001 # Arbitrary small threshold
    
    final_moves = np.where(np.logical_and(is_hungry, has_smell), best_smell_dir, final_moves)
    final_moves = np.where(np.logical_and(is_hungry, has_food_nearby), best_food_dir, final_moves)
    
    # Apply Movement
    rows, cols = np.indices((settings.world_size, settings.world_size))
    heuristic_actions[rows, cols, final_moves] = 1.0
    
    # --- Eating ---
    # Always eat if food is present (Aggressive)
    should_eat = prey_map > EAT_THRESHOLD
    
    # Overwrite movement with EAT
    EAT_INDEX = 4
    heuristic_actions[should_eat] = 0
    heuristic_actions[should_eat, EAT_INDEX] = 1.0
    
    return heuristic_actions
