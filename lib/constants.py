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
    # REST = 5

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUNNER = "petting_zoo_single"

# SPEED_MULTIPLIER = 2
# EAT_AMOUNT_BOOST = 10
# MULTIPLY_DEATH_RATE = 3
SPEED_MULTIPLIER = 1
EAT_AMOUNT_BOOST = 5
MULTIPLY_DEATH_RATE = 1

MAP_METER_SIZE = 300 * 1000
WORLD_SIZE = 48
# meters per second
FISH_SWIM_SPEED = 0.025
SECONDS_IN_DAY = 86400
DAYS_TO_CROSS_MAP = MAP_METER_SIZE / (FISH_SWIM_SPEED * SECONDS_IN_DAY)
DAYS_PER_STEP = (DAYS_TO_CROSS_MAP / WORLD_SIZE) * SPEED_MULTIPLIER
SCALE_FISHING = 1
FIXED_BIOMASS = True
WORLD_SIZE = 48

NOISE_SCALING = 6
STARTING_BIOMASS_COD = 44897
STARTING_BIOMASS_HERRING = 737356
STARTING_BIOMASS_SPRAT = 1359874
STARTING_BIOMASS_PLANKTON = 1359874 * 2
# STARTING_BIOMASS_PLANKTON = 2000
# STARTING_BIOMASS_COD = 100
# STARTING_BIOMASS_HERRING = 200
# STARTING_BIOMASS_SPRAT = 200
# STARTING_BIOMASS_PLANKTON = 100
# BASE_FISHING_VALUE_COD = 0.005414
BASE_FISHING_VALUE_COD = 0.002651024
BASE_FISHING_VALUE_HERRING = 0.002651024
BASE_FISHING_VALUE_SPRAT = 0.002651024
MIN_PERCENT_ALIVE = 0
MAX_PERCENT_ALIVE = 8
MAX_ENERGY = 100
BASE_ENERGY_COST = 0.5 * DAYS_PER_STEP
ENERGY_REWARD_FOR_EATING = 250
ENERGY_REWARD_FOR_EATING_COD = 500
GROWTH_MULTIPLIER = 1
print(f"Days per step: {DAYS_PER_STEP}")
MAX_STEPS = 1000
# print years for max steps
print(f"Max steps: {MAX_STEPS} ({MAX_STEPS / 365} years)")

EVAL_AGENT = './agents/test.pt'

# Define species properties in a map
MAX_PLANKTON_IN_CELL = (STARTING_BIOMASS_PLANKTON / (WORLD_SIZE * WORLD_SIZE)) * 10
def update_fishing_scaler(scalar):
    global SCALE_FISHING
    global SPECIES_MAP
    global BASE_FISHING_VALUE_COD
    global BASE_FISHING_VALUE_HERRING
    global BASE_FISHING_VALUE_SPRAT
    SCALE_FISHING = scalar

    fishing_percent = -1
    for species in SPECIES_MAP.keys():
        if species == "plankton":
            continue
        base_fishing_amount = globals()[f"BASE_FISHING_VALUE_{species.upper()}"]
        fishing_percent = base_fishing_amount * DAYS_PER_STEP * SCALE_FISHING
        SPECIES_MAP[species]["fishing_mortality_rate"] = fishing_percent
    
    return fishing_percent

def update_initial_biomass(species, value):
    global SPECIES_MAP

    if species not in SPECIES_MAP:
        raise ValueError(f"Species '{species}' not found in SPECIES_MAP.")

    if value < 0:
        raise ValueError("Initial biomass cannot be negative.")

    SPECIES_MAP[species]["starting_biomass"] = value
    SPECIES_MAP[species]["original_starting_biomass"] = value
    SPECIES_MAP[species]["max_biomass_in_cell"] = (value / (WORLD_SIZE * WORLD_SIZE)) * 20
    # SPECIES_MAP[species]["max_biomass_in_cell"] = SPECIES_MAP[species]["max_biomass_in_cell"] * 100
    SPECIES_MAP[species]["min_biomass_in_cell"] = 0

def update_energy_params(energy_cost, energy_reward, energy_reward_cod, eat_amount_boost):
    global BASE_ENERGY_COST
    global ENERGY_REWARD_FOR_EATING
    global ENERGY_REWARD_FOR_EATING_COD
    global EAT_AMOUNT_BOOST

    BASE_ENERGY_COST = energy_cost * DAYS_PER_STEP
    ENERGY_REWARD_FOR_EATING = energy_reward
    ENERGY_REWARD_FOR_EATING_COD = energy_reward_cod
    EAT_AMOUNT_BOOST = eat_amount_boost

    for species in SPECIES_MAP.keys():
        if species == "plankton":
            continue
        SPECIES_MAP[species]["max_consumption_rate"] = SPECIES_MAP[species]["max_consumption_rate"] * EAT_AMOUNT_BOOST * DAYS_PER_STEP
    
def update_fishing_for_species(species, value):
    global SPECIES_MAP

    SPECIES_MAP[species]["fishing_mortality_rate"] = value * DAYS_PER_STEP * SCALE_FISHING

SPECIES_MAP = {
    "plankton": {
        "index": 0,
        "original_starting_biomass": STARTING_BIOMASS_PLANKTON,
        "starting_biomass": STARTING_BIOMASS_PLANKTON,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": 0,
        "max_biomass_in_cell": MAX_PLANKTON_IN_CELL,
        "hardcoded_logic": True,
        "hardcoded_rules": {
            "growth_rate_constant": 50,
            # "growth_rate_constant": 2.5,
            "respawn_delay": 10,
        },
        "visualization": {
            "color": [100, 220, 100],
            "color_ones": [100 / 255, 220 / 255, 100 / 255]
        },
        "noise_threshold": 0.35,
        "noise_scaling": NOISE_SCALING,
    },
    "herring": {
        "index": 1,
        "original_starting_biomass": STARTING_BIOMASS_HERRING,
        "starting_biomass": STARTING_BIOMASS_HERRING,
        "average_weight": 7,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": 0,
        "max_biomass_in_cell": (STARTING_BIOMASS_HERRING / (WORLD_SIZE * WORLD_SIZE)) * 20,
        "activity_metabolic_rate": 0.022360679760000002 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "standard_metabolic_rate": 0.00447213596 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "max_consumption_rate": 0.0529 * DAYS_PER_STEP * EAT_AMOUNT_BOOST,
        "natural_mortality_rate": 0.001604815 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "fishing_mortality_rate": 0.002651024 * DAYS_PER_STEP * SCALE_FISHING,
        "growth_rate": 0.026 * DAYS_PER_STEP * GROWTH_MULTIPLIER,
        "hardcoded_logic": False,
        "visualization": {
            "color": [220, 60, 60],
            "color_ones": [220 / 255, 60 / 255, 60 / 255]
        },
        "noise_threshold": 0.35,
        "noise_scaling": NOISE_SCALING,
    },
    "sprat": {
        "index": 2,
        "original_starting_biomass": STARTING_BIOMASS_SPRAT,
        "starting_biomass": STARTING_BIOMASS_SPRAT,
        "average_weight": 7,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": 0,
        "max_biomass_in_cell": (STARTING_BIOMASS_SPRAT / (WORLD_SIZE * WORLD_SIZE)) * 20,
        "activity_metabolic_rate": 0.02686424833333333 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "standard_metabolic_rate": 0.005372849666666666 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "max_consumption_rate": 0.0611 * DAYS_PER_STEP * EAT_AMOUNT_BOOST,
        "natural_mortality_rate": 0.001056525 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "fishing_mortality_rate": 0.002651024 * DAYS_PER_STEP * SCALE_FISHING,
        "growth_rate": 0.029 * DAYS_PER_STEP * GROWTH_MULTIPLIER,
        "hardcoded_logic": False,
        "visualization": {
            "color": [240, 140, 60],
            "color_ones": [240 / 255, 140 / 255, 60 / 255]
        },
        "noise_threshold": 0.35,
        "noise_scaling": NOISE_SCALING,
    },
    "cod": {
        "index": 3,
        "original_starting_biomass": STARTING_BIOMASS_COD,
        "starting_biomass": STARTING_BIOMASS_COD,
        "max_in_cell": STARTING_BIOMASS_COD / 2,
        "smell_emission_rate": 0.1,
        "min_biomass_in_cell": 0,
        "max_biomass_in_cell": (STARTING_BIOMASS_COD / (WORLD_SIZE * WORLD_SIZE)) * 50,
        "activity_metabolic_rate": 0.014535768421428572 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "standard_metabolic_rate": 0.0029071536857142857 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "max_consumption_rate": 0.0376 * DAYS_PER_STEP * EAT_AMOUNT_BOOST,
        "natural_mortality_rate": 0.003 * DAYS_PER_STEP * MULTIPLY_DEATH_RATE,
        "fishing_mortality_rate": 0.005414 * DAYS_PER_STEP * SCALE_FISHING,
        "growth_rate": 0.02 * DAYS_PER_STEP * GROWTH_MULTIPLIER,
        "hardcoded_logic": False,
        "visualization": {
            "color": [40, 40, 40],
            "color_ones": [40 / 255, 40 / 255, 40 / 255]
        },
        "noise_threshold": 0.7,
        "noise_scaling": NOISE_SCALING * 5,
    }
}

EATING_MAP = {
    "plankton": {},
    "herring": {
        "plankton": {}
    },
    "sprat": {
        "plankton": {}
    },
    "cod": {
        "herring": {},
        "sprat": {},
    }
}

FISHING_AMOUNT = 0.1
FISHING_OCCURRENCE = 100

SMELL_EMISSION_RATE = 0.05
SMELL_DECAY_RATE = 0.2

NUM_AGENTS = 16
AGENT_EVALUATIONS = 4
ELITISM_SELECTION = 8
TOURNAMENT_SELECTION = 4
GENERATIONS_PER_RUN = 150

INITIAL_MUTATION_RATE = 0.15
MIN_MUTATION_RATE = 0.01
MUTATION_RATE_DECAY = 0.995

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

TOTAL_TENSOR_VALUES = offset

NETWORK_INPUT_SIZE = TOTAL_TENSOR_VALUES * 9
AVAILABLE_ACTIONS = len(Action)
NETWORK_HIDDEN_SIZE = 64

# number of species with hardcoded = False
ACTION_TAKING_SPECIES = sum(1 for species in SPECIES_MAP.values() if not species["hardcoded_logic"])
NETWORK_OUTPUT_SIZE = AVAILABLE_ACTIONS
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
    global EAT_AMOUNT_BOOST
    
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
        SPECIES_MAP[species]["max_biomass_in_cell"] = biomass / 2
        SPECIES_MAP[species]["min_biomass_in_cell"] = biomass / (WORLD_SIZE * WORLD_SIZE) / 20

    MAX_STEPS = options["maxSteps"]

    