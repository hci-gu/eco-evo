import os
from enum import Enum

class Terrain(Enum):
    LAND = 0
    WATER = 1
    OUT_OF_BOUNDS = 2

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

WORLD_SIZE = 50
NOISE_SCALING = 4.5
STARTING_BIOMASS_COD = 3000
STARTING_BIOMASS_ANCHOVY = 6400
STARTING_BIOMASS_PLANKTON = 14800
MIN_PERCENT_ALIVE = 0.2
MAX_PERCENT_ALIVE = 3
MAX_STEPS = 2000

# Define species properties in a map
SPECIES_MAP = {
    "plankton": {
        "index": 0,
        "starting_biomass": STARTING_BIOMASS_PLANKTON,
        "max_in_cell": STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE) * 1.5,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": 0,
        "max_biomass_in_cell": (STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE)) * 1.5,
        "hardcoded_logic": True,
        "hardcoded_rules": {
            "growth_rate": 0.0075,
            "respawn_delay": 150,
            "base_spawn_rate": STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE) / 20,
        },
        "visualization": {
            "color": [0, 255, 0]
        }
    },
    "anchovy": {
        "index": 1,
        "starting_biomass": STARTING_BIOMASS_ANCHOVY,
        "max_in_cell": STARTING_BIOMASS_ANCHOVY / 2,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": STARTING_BIOMASS_ANCHOVY / (WORLD_SIZE * WORLD_SIZE) / 20,
        "max_biomass_in_cell": STARTING_BIOMASS_ANCHOVY / 2,
        "hardcoded_logic": False,
        "visualization": {
            "color": [255, 0, 0]
        }
    },
    "cod": {
        "index": 2,
        "starting_biomass": STARTING_BIOMASS_COD,
        "max_in_cell": STARTING_BIOMASS_COD / 2,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": STARTING_BIOMASS_COD / (WORLD_SIZE * WORLD_SIZE) / 20,
        "max_biomass_in_cell": STARTING_BIOMASS_COD / 2,
        "hardcoded_logic": False,
        "visualization": {
            "color": [0, 0, 0]
        }
    }
}

EATING_MAP = {
    "plankton": {
        "anchovy": {"success_rate": 0.0, "nutrition_amount": 0.0},
        "cod": {"success_rate": 0.0, "nutrition_amount": 0.0},
    },
    "anchovy": {
        "plankton": {"success_rate": 0.5, "nutrition_amount": 0.25},
        "cod": {"success_rate": 0.0, "nutrition_amount": 0.0},
    },
    "cod": {
        "plankton": {"success_rate": 0.25, "nutrition_amount": 0.25},
        "anchovy": {"success_rate": 0.25, "nutrition_amount": 0.25},
    }
}

HUNT_SUCCESS_RATE_ANCHOVY = 0.5
HUNT_SUCCESS_RATE_COD = 0.25
EAT_AMOUNT_ANCHOVY = 0.25
EAT_AMOUNT_COD = 0.25
BASE_BIOMASS_LOSS = 0.05
BIOMASS_GROWTH_RATE = 0.075
PLANKTON_GROWTH_RATE = 0.0075

BASE_PLANKTON_SPAWN_RATE = STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE) / 20
PLANKTON_RESPAWN_DELAY = 150
ENERGY_REWARD_FOR_EATING = 25
MAX_ENERGY = 100.0

SMELL_EMISSION_RATE = 0.05
SMELL_DECAY_RATE = 0.2

NUM_AGENTS = 24
AGENT_EVALUATIONS = 4
ELITISM_SELECTION = 8
TOURNAMENT_SELECTION = 6
BASE_ENERGY_COST = 0.5
GENERATIONS_PER_RUN = 150

INITIAL_MUTATION_RATE = 0.15
MIN_MUTATION_RATE = 0.01
MUTATION_RATE_DECAY = 0.99

# Calculate offsets dynamically based on SPECIES_MAP
OFFSETS_TERRAIN_LAND = 0
OFFSETS_TERRAIN_WATER = 1
OFFSETS_TERRAIN_OUT_OF_BOUNDS = 2

OFFSETS_BIOMASS = OFFSETS_TERRAIN_OUT_OF_BOUNDS + 1

offset = OFFSETS_BIOMASS
for species in SPECIES_MAP.keys():
    SPECIES_MAP[species]["biomass_offset"] = offset
    offset += 1

OFFSETS_ENERGY = offset
for species in SPECIES_MAP.keys():
    SPECIES_MAP[species]["energy_offset"] = offset
    offset += 1

OFFSETS_SMELL = offset
for species in SPECIES_MAP.keys():
    SPECIES_MAP[species]["smell_offset"] = offset
    offset += 1

# Calculate the total required number of values in the world tensor
TOTAL_TENSOR_VALUES = OFFSETS_SMELL + len(SPECIES_MAP)

NETWORK_INPUT_SIZE = TOTAL_TENSOR_VALUES * 9
AVAILABLE_ACTIONS = len(Action)
NETWORK_HIDDEN_SIZE = 10

# number of species with hardcoded = False
ACTION_TAKING_SPECIES = sum(1 for species in SPECIES_MAP.values() if not species["hardcoded_logic"])
NETWORK_OUTPUT_SIZE = AVAILABLE_ACTIONS * ACTION_TAKING_SPECIES

CURRENT_FOLDER = "results/run"

def override_from_file(file_path):
    global WORLD_SIZE
    global STARTING_BIOMASS_COD
    global STARTING_BIOMASS_ANCHOVY
    global STARTING_BIOMASS_PLANKTON
    global MIN_PERCENT_ALIVE
    global MAX_STEPS
    global EAT_AMOUNT_ANCHOVY
    global EAT_AMOUNT_COD
    global BASE_BIOMASS_LOSS
    global BIOMASS_GROWTH_RATE
    global PLANKTON_GROWTH_RATE
    global MAX_PLANKTON_IN_CELL
    global MAX_ENERGY
    global NUM_AGENTS
    global ELITISM_SELECTION
    global TOURNAMENT_SELECTION
    global BASE_ENERGY_COST
    global GENERATIONS_PER_RUN
    
    global CURRENT_FOLDER
    file_name = os.path.basename(file_path).split(".")[0]

    CURRENT_FOLDER = f"results/{file_name}"
    if not os.path.exists(CURRENT_FOLDER):
        os.makedirs(CURRENT_FOLDER)
        if not os.path.exists(f"{CURRENT_FOLDER}/agents"):
            os.makedirs(f"{CURRENT_FOLDER}/agents")

    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            if key in globals():
                globals()[key] = type(globals()[key])(value)
    
    print(f"Overridden constants from file {file_path}")
    print(f"NUM_AGENTS: {NUM_AGENTS}")



