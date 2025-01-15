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
    REST = 5

SPEED_MULTIPLIER = 3
EAT_REWARD_BOOST = 10

MAP_METER_SIZE = 300 * 1000
WORLD_SIZE = 50
# meters per second
FISH_SWIM_SPEED = 0.025
SECONDS_IN_DAY = 86400
DAYS_TO_CROSS_MAP = MAP_METER_SIZE / (FISH_SWIM_SPEED * SECONDS_IN_DAY)
DAYS_PER_STEP = (DAYS_TO_CROSS_MAP / WORLD_SIZE) * SPEED_MULTIPLIER

NOISE_SCALING = 4.5
STARTING_BIOMASS_COD = 387941
STARTING_BIOMASS_HERRING = 759392
STARTING_BIOMASS_SPRAT = 1525100
STARTING_BIOMASS_PLANKTON = 5000000
MIN_PERCENT_ALIVE = 0.2
MAX_PERCENT_ALIVE = 3
MAX_STEPS = 2500

EVAL_AGENT = './agents/test.pt'

# Define species properties in a map
MAX_PLANKTON_IN_CELL = (STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE)) * 1.5
SPECIES_MAP = {
    "plankton": {
        "index": 0,
        "starting_biomass": STARTING_BIOMASS_PLANKTON,
        "max_in_cell": STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE) * 1.5,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": 0,
        "max_biomass_in_cell": MAX_PLANKTON_IN_CELL,
        "hardcoded_logic": True,
        "hardcoded_rules": {
            "growth_rate_constant": 0.15,
            "growth_threshold": MAX_PLANKTON_IN_CELL * 0.75,
            "respawn_delay": 400,
        },
        "visualization": {
            "color": [0, 255, 0]
        }
    },
    "herring": {
        "index": 1,
        "starting_biomass": STARTING_BIOMASS_HERRING,
        "max_in_cell": STARTING_BIOMASS_HERRING / 2,
        "average_weight": 7,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": STARTING_BIOMASS_HERRING / (WORLD_SIZE * WORLD_SIZE) / 20,
        "max_biomass_in_cell": STARTING_BIOMASS_HERRING / 2,
        "activity_metabolic_rate": 0.00055901699 * DAYS_PER_STEP,
        "standard_metabolic_rate": 0.00011180339 * DAYS_PER_STEP,
        "max_consumption_rate": 0.00132174564 * DAYS_PER_STEP,
        "natural_mortality_rate": 0.001604815 * DAYS_PER_STEP,
        "fishing_mortality_rate": 0.002651024 * DAYS_PER_STEP,
        "hardcoded_logic": False,
        "visualization": {
            "color": [255, 0, 0]
        }
    },
    "spat": {
        "index": 2,
        "starting_biomass": STARTING_BIOMASS_SPRAT,
        "max_in_cell": STARTING_BIOMASS_SPRAT / 2,
        "average_weight": 7,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": STARTING_BIOMASS_SPRAT / (WORLD_SIZE * WORLD_SIZE) / 20,
        "max_biomass_in_cell": STARTING_BIOMASS_SPRAT / 2,
        "activity_metabolic_rate": 0.00032237098 * DAYS_PER_STEP,
        "standard_metabolic_rate": 0.00006447419 * DAYS_PER_STEP,
        "max_consumption_rate": 0.00073367438 * DAYS_PER_STEP,
        "natural_mortality_rate": 0.001056525 * DAYS_PER_STEP,
        "fishing_mortality_rate": 0.002651024 * DAYS_PER_STEP,
        "hardcoded_logic": False,
        "visualization": {
            "color": [255, 100, 0]
        }
    },
    "cod": {
        "index": 3,
        "starting_biomass": STARTING_BIOMASS_COD,
        "max_in_cell": STARTING_BIOMASS_COD / 2,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": STARTING_BIOMASS_COD / (WORLD_SIZE * WORLD_SIZE) / 20,
        "max_biomass_in_cell": STARTING_BIOMASS_COD / 2,
        "activity_metabolic_rate": 0.000215 * DAYS_PER_STEP,
        "standard_metabolic_rate": 0.000043 * DAYS_PER_STEP,
        "max_consumption_rate": 0.00526 * DAYS_PER_STEP,
        "natural_mortality_rate": 0.003 * DAYS_PER_STEP,
        "fishing_mortality_rate": 0.005413783 * DAYS_PER_STEP,
        "hardcoded_logic": False,
        "visualization": {
            "color": [0, 0, 0]
        }
    }
}

EATING_MAP = {
    "plankton": {},
    "herring": {
        "plankton": {}
    },
    "spat": {
        "plankton": {}
    },
    "cod": {
        "herring": {},
        "spat": {},
    }
}

FISHING_AMOUNT = 0.1
FISHING_OCCURRENCE = 100

BASE_BIOMASS_LOSS = 0.05
BIOMASS_GROWTH_RATE = 0.075

SMELL_EMISSION_RATE = 0.05
SMELL_DECAY_RATE = 0.2

NUM_AGENTS = 24
AGENT_EVALUATIONS = 4
ELITISM_SELECTION = 8
TOURNAMENT_SELECTION = 6
GENERATIONS_PER_RUN = 500

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

# OFFSETS_ENERGY = offset
# for species in SPECIES_MAP.keys():
#     SPECIES_MAP[species]["energy_offset"] = offset
#     offset += 1

OFFSETS_SMELL = offset
for species in SPECIES_MAP.keys():
    SPECIES_MAP[species]["smell_offset"] = offset
    offset += 1

# Calculate the total required number of values in the world tensor
TOTAL_TENSOR_VALUES = OFFSETS_SMELL + len(SPECIES_MAP)


NETWORK_INPUT_SIZE = TOTAL_TENSOR_VALUES * 9
AVAILABLE_ACTIONS = len(Action)
NETWORK_HIDDEN_SIZE = 12

# number of species with hardcoded = False
ACTION_TAKING_SPECIES = sum(1 for species in SPECIES_MAP.values() if not species["hardcoded_logic"])
NETWORK_OUTPUT_SIZE = AVAILABLE_ACTIONS * ACTION_TAKING_SPECIES

CURRENT_FOLDER = "results/run"

def override_from_file(file_path):
    global WORLD_SIZE
    global STARTING_BIOMASS_COD
    global STARTING_BIOMASS_PLANKTON
    global MIN_PERCENT_ALIVE
    global MAX_STEPS
    global BASE_BIOMASS_LOSS
    global BIOMASS_GROWTH_RATE
    global NUM_AGENTS
    global ELITISM_SELECTION
    global TOURNAMENT_SELECTION
    global GENERATIONS_PER_RUN
    global SPEED_MULTIPLIER
    global EAT_REWARD_BOOST
    
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



