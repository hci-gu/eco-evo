import os
import torch
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUNNER = "rl_runner"

# SPEED_MULTIPLIER = 2
# EAT_REWARD_BOOST = 10
# SURVIVAL_BOOST = 3
SPEED_MULTIPLIER = 1
EAT_REWARD_BOOST = 5
COD_EAT_REWARD_BOOST = 6
SURVIVAL_BOOST = 4

MAP_METER_SIZE = 300 * 1000
WORLD_SIZE = 48
# meters per second
FISH_SWIM_SPEED = 0.025
SECONDS_IN_DAY = 86400
DAYS_TO_CROSS_MAP = MAP_METER_SIZE / (FISH_SWIM_SPEED * SECONDS_IN_DAY)
DAYS_PER_STEP = (DAYS_TO_CROSS_MAP / WORLD_SIZE) * SPEED_MULTIPLIER
SCALE_FISHING = 0

NOISE_SCALING = 4.5
STARTING_BIOMASS_COD = 387941
STARTING_BIOMASS_HERRING = 759392
STARTING_BIOMASS_SPRAT = 1525100
STARTING_BIOMASS_PLANKTON = 5000000
MIN_PERCENT_ALIVE = 0.05
MAX_PERCENT_ALIVE = 3
MAX_STEPS = 7500

EVAL_AGENT = './agents/test.pt'

# Define species properties in a map
MAX_PLANKTON_IN_CELL = (STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE)) * 1.5
SPECIES_MAP = {
    "plankton": {
        "index": 0,
        "original_starting_biomass": STARTING_BIOMASS_PLANKTON,
        "starting_biomass": STARTING_BIOMASS_PLANKTON,
        "max_in_cell": STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE) * 2,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": 0,
        "max_biomass_in_cell": MAX_PLANKTON_IN_CELL,
        "hardcoded_logic": True,
        "hardcoded_rules": {
            "growth_rate_constant": 0.2,
            "growth_threshold": MAX_PLANKTON_IN_CELL * 0.8,
            "respawn_delay": 40,
        },
        "visualization": {
            "color": [0, 255, 0]
        }
    },
    "herring": {
        "index": 1,
        "original_starting_biomass": STARTING_BIOMASS_HERRING,
        "starting_biomass": STARTING_BIOMASS_HERRING,
        "max_in_cell": STARTING_BIOMASS_HERRING / 2,
        "average_weight": 7,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": STARTING_BIOMASS_HERRING / (WORLD_SIZE * WORLD_SIZE) / 20,
        "max_biomass_in_cell": STARTING_BIOMASS_HERRING / 2,
        "activity_metabolic_rate": 0.022360679760000002 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "standard_metabolic_rate": 0.00447213596 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "max_consumption_rate": 0.0529 * DAYS_PER_STEP * EAT_REWARD_BOOST,
        "natural_mortality_rate": 0.001604815 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "fishing_mortality_rate": 0.002651024 * DAYS_PER_STEP * SCALE_FISHING,
        "hardcoded_logic": False,
        "visualization": {
            "color": [255, 0, 0]
        }
    },
    "spat": {
        "index": 2,
        "original_starting_biomass": STARTING_BIOMASS_SPRAT,
        "starting_biomass": STARTING_BIOMASS_SPRAT,
        "max_in_cell": STARTING_BIOMASS_SPRAT / 2,
        "average_weight": 7,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": STARTING_BIOMASS_SPRAT / (WORLD_SIZE * WORLD_SIZE) / 20,
        "max_biomass_in_cell": STARTING_BIOMASS_SPRAT / 2,
        "activity_metabolic_rate": 0.02686424833333333 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "standard_metabolic_rate": 0.005372849666666666 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "max_consumption_rate": 0.0611 * DAYS_PER_STEP * EAT_REWARD_BOOST,
        "natural_mortality_rate": 0.001056525 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "fishing_mortality_rate": 0.002651024 * DAYS_PER_STEP * SCALE_FISHING,
        "hardcoded_logic": False,
        "visualization": {
            "color": [255, 100, 0]
        }
    },
    "cod": {
        "index": 3,
        "original_starting_biomass": STARTING_BIOMASS_COD,
        "starting_biomass": STARTING_BIOMASS_COD,
        "max_in_cell": STARTING_BIOMASS_COD / 2,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": STARTING_BIOMASS_COD / (WORLD_SIZE * WORLD_SIZE) / 20,
        "max_biomass_in_cell": STARTING_BIOMASS_COD / 2,
        "activity_metabolic_rate": 0.014535768421428572 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "standard_metabolic_rate": 0.0029071536857142857 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "max_consumption_rate": 0.0376 * DAYS_PER_STEP * COD_EAT_REWARD_BOOST,
        "natural_mortality_rate": 0.003 * DAYS_PER_STEP / SURVIVAL_BOOST,
        "fishing_mortality_rate": 0.005413783 * DAYS_PER_STEP * SCALE_FISHING,
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

SMELL_EMISSION_RATE = 0.05
SMELL_DECAY_RATE = 0.2

NUM_AGENTS = 16
AGENT_EVALUATIONS = 4
ELITISM_SELECTION = 2
TOURNAMENT_SELECTION = 2
GENERATIONS_PER_RUN = 100

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
NETWORK_HIDDEN_SIZE = 24

# number of species with hardcoded = False
ACTION_TAKING_SPECIES = sum(1 for species in SPECIES_MAP.values() if not species["hardcoded_logic"])
NETWORK_OUTPUT_SIZE = AVAILABLE_ACTIONS * ACTION_TAKING_SPECIES
NETWORK_OUTPUT_SIZE_SINGLE_SPECIES = AVAILABLE_ACTIONS

CURRENT_FOLDER = "results/run"

def override_from_file(file_path):
    global RUNNER
    global WORLD_SIZE
    global STARTING_BIOMASS_COD
    global STARTING_BIOMASS_PLANKTON
    global MIN_PERCENT_ALIVE
    global MAX_STEPS
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

def override_from_options(options):
    global STARTING_BIOMASS_COD
    global STARTING_BIOMASS_HERRING
    global STARTING_BIOMASS_SPRAT
    global MAX_STEPS
    global MIN_PERCENT_ALIVE
    global EVAL_AGENT

    MIN_PERCENT_ALIVE = 0.01

    if "agent" in options:
        EVAL_AGENT = "./agents/" + options["agent"]

    for species, amount in options["fishingAmounts"].items():
        SPECIES_MAP[species]["fishing_mortality_rate"] = (amount / 100) * DAYS_PER_STEP

    for species, amount in options["initialPopulation"].items():
        biomass = amount * 1000
        SPECIES_MAP[species]["starting_biomass"] = biomass
        SPECIES_MAP[species]["max_in_cell"] = biomass / 2
        SPECIES_MAP[species]["min_biomass_in_cell"] = biomass / (WORLD_SIZE * WORLD_SIZE) / 20

    MAX_STEPS = options["maxSteps"]

    
    




