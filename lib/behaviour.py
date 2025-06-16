import os
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from lib.world import (
    update_smell,
    apply_movement_delta,
    all_movement_delta,
    matrix_perform_eating
)
import lib.constants as const
from lib.model import Model, SingleSpeciesModel

# Define the Action enum
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

ACTION_ORDER = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN, Action.EAT]

def observe(world):
    terrain = world[..., 0:3]
    biomass = []
    smell = []
    for species in const.SPECIES_MAP.keys():
        max_biomass = world[..., const.SPECIES_MAP[species]["biomass_offset"]].max()
        max_smell = world[..., const.SPECIES_MAP[species]["smell_offset"]].max()
        biomass.append(world[..., const.SPECIES_MAP[species]["biomass_offset"]] / (max_biomass + 1e-8))
        smell.append(world[..., const.SPECIES_MAP[species]["smell_offset"]] / (max_smell + 1e-8))
    biomass = np.stack(biomass, axis=-1)
    smell = np.stack(smell, axis=-1)
    
    energy = world[..., const.OFFSETS_ENERGY:const.OFFSETS_ENERGY+4] / (const.MAX_ENERGY + 1e-8)
    observation = np.concatenate([terrain, biomass, smell, energy], axis=-1)
    
    return observation

def take_step(world, world_data, species, action):
    movement_deltas = all_movement_delta(world, world_data, species, action)
    apply_movement_delta(world, species, movement_deltas)
    matrix_perform_eating(world, species, action)

def get_action(candidate, species, world):
    obs = observe(world)
    obs = obs.reshape(-1, 135)

    # single species
    return candidate.forward(obs)[0]

    # Add species information as an extra feature.
    species_index = const.SPECIES_MAP[species]["index"]
    species_normalized = species_index / len(const.SPECIES_MAP)
    obs = np.concatenate([np.full((obs.shape[0], 1), species_normalized), obs], axis=1)

    # Forward pass of the candidate model returns a probability vector in the specified order.
    return candidate.forward(obs, species)[0]

# ---------------------------------------------------------------------------
#  SCENARIOS  — 10 per species (Herring, Sprat, Cod)
# ---------------------------------------------------------------------------

# ------------------------------ HERRING ------------------------------------

def scenario_1():
    """Herring moves LEFT toward Plankton."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[1, 0, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_2():
    """Herring moves UP toward Plankton."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[0, 1, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_3():
    """Herring moves RIGHT toward Plankton."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[1, 2, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_4():
    """Herring moves DOWN toward Plankton."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[2, 1, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_5():
    """Herring EATS Plankton when co-located."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[1, 1, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_6():
    """Herring chooses the NEAREST of two Plankton, moves LEFT."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[1, 0, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10  # nearer
    world[0, 2, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10  # further
    return world, world_data


def scenario_7():
    """Herring FLEES Cod to the RIGHT or DOWN (any away)."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    # threatening Cod on left
    world[1, 0, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    return world, world_data


def scenario_8():
    """Herring FLEES Cod located ABOVE, expected DOWN."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[0, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    return world, world_data


def scenario_9():
    """Herring FLEES Cod when sharing a cell (any non-EAT movement)."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    # herring & cod in same center cell
    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    return world, world_data


def scenario_10():
    """Herring EATS immediately when Plankton & Cod absent."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    world[1, 1, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 5
    return world, world_data

# ------------------------------ SPRAT --------------------------------------

def scenario_11():
    """Sprat moves LEFT toward Plankton."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[1, 0, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_12():
    """Sprat moves UP toward Plankton."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[0, 1, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_13():
    """Sprat moves RIGHT toward Plankton."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[1, 2, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_14():
    """Sprat moves DOWN toward Plankton."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[2, 1, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_15():
    """Sprat EATS Plankton when co-located."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[1, 1, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_16():
    """Sprat FLEES Cod to LEFT/UP/DOWN (choose any safe)."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[1, 2, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    return world, world_data


def scenario_17():
    """Sprat FLEES Cod ABOVE (expected DOWN)."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[0, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    return world, world_data


def scenario_18():
    """Sprat chooses NEAREST of two Plankton (expected LEFT)."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[1, 0, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    world[2, 2, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 10
    return world, world_data


def scenario_19():
    """Sprat EATS when energy low and food present."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[1, 1, const.SPECIES_MAP["sprat"]["energy_offset"]] = 0
    world[1, 1, const.SPECIES_MAP["plankton"]["biomass_offset"]] = 5
    return world, world_data


def scenario_20():
    """Sprat FLEES TWO Cod (one same cell, one adjacent)."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[0, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    return world, world_data

# ------------------------------ COD ----------------------------------------

def scenario_21():
    """Cod moves LEFT toward Herring."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[1, 0, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    return world, world_data


def scenario_22():
    """Cod moves RIGHT toward Sprat."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[1, 2, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    return world, world_data


def scenario_23():
    """Cod moves UP toward Herring."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[0, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    return world, world_data


def scenario_24():
    """Cod moves DOWN toward Sprat."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[2, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    return world, world_data


def scenario_25():
    """Cod EATS Sprat when co-located."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[1, 1, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    return world, world_data


def scenario_26():
    """Cod chooses NEAREST of multiple prey, LEFT toward Herring."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    # nearer Herring left
    world[1, 0, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    # farther Sprat up-right
    world[0, 2, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    return world, world_data


def scenario_27():
    """Cod moves RIGHT when prey diagonal but right-closest Sprat."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[0, 2, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    return world, world_data


def scenario_28():
    """Cod FOLLOWS tailing Herring DOWN."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[0, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10  # prey ahead
    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10  # cod centre
    return world, world_data


def scenario_29():
    """Cod FLEES bigger Cod? (self) — Not applicable, instead stays chasing SPRAT LEFT."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[1, 0, const.SPECIES_MAP["sprat"]["biomass_offset"]] = 10
    world[0, 2, const.SPECIES_MAP["herring"]["biomass_offset"]] = 10
    return world, world_data


def scenario_30():
    """Cod EATS Herring when energy low."""
    world = np.zeros((3, 3, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((3, 3, 5), dtype=np.float32)
    world[..., :3] = np.array([0, 1, 0], dtype=np.float32)

    world[1, 1, const.SPECIES_MAP["cod"]["biomass_offset"]] = 10
    world[1, 1, const.SPECIES_MAP["cod"]["energy_offset"]] = 0
    world[1, 1, const.SPECIES_MAP["herring"]["biomass_offset"]] = 6
    return world, world_data

# ---------------------------------------------------------------------------
#  SCENARIOS LIST (30 total — 10 per species)
# ---------------------------------------------------------------------------

SCENARIOS = [
    # Herring (1–10)
    {"name": "Herring moves LEFT toward Plankton", "creator": scenario_1, "species": "herring", "expected_actions": [Action.LEFT]},
    {"name": "Herring moves UP toward Plankton", "creator": scenario_2, "species": "herring", "expected_actions": [Action.UP]},
    {"name": "Herring moves RIGHT toward Plankton", "creator": scenario_3, "species": "herring", "expected_actions": [Action.RIGHT]},
    {"name": "Herring moves DOWN toward Plankton", "creator": scenario_4, "species": "herring", "expected_actions": [Action.DOWN]},
    {"name": "Herring eats Plankton present", "creator": scenario_5, "species": "herring", "expected_actions": [Action.EAT]},
    {"name": "Herring chooses nearest Plankton LEFT", "creator": scenario_6, "species": "herring", "expected_actions": [Action.LEFT]},
    {"name": "Herring flees Cod on LEFT", "creator": scenario_7, "species": "herring", "expected_actions": [Action.RIGHT, Action.DOWN]},
    {"name": "Herring flees Cod ABOVE", "creator": scenario_8, "species": "herring", "expected_actions": [Action.DOWN]},
    {"name": "Herring flees Cod same cell", "creator": scenario_9, "species": "herring", "expected_actions": [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]},
    {"name": "Herring eats when safe", "creator": scenario_10, "species": "herring", "expected_actions": [Action.EAT]},

    # Sprat (11–20)
    {"name": "Sprat moves LEFT toward Plankton", "creator": scenario_11, "species": "sprat", "expected_actions": [Action.LEFT]},
    {"name": "Sprat moves UP toward Plankton", "creator": scenario_12, "species": "sprat", "expected_actions": [Action.UP]},
    {"name": "Sprat moves RIGHT toward Plankton", "creator": scenario_13, "species": "sprat", "expected_actions": [Action.RIGHT]},
    {"name": "Sprat moves DOWN toward Plankton", "creator": scenario_14, "species": "sprat", "expected_actions": [Action.DOWN]},
    {"name": "Sprat eats Plankton present", "creator": scenario_15, "species": "sprat", "expected_actions": [Action.EAT]},
    {"name": "Sprat flees Cod RIGHT", "creator": scenario_16, "species": "sprat", "expected_actions": [Action.LEFT, Action.UP, Action.DOWN]},
    {"name": "Sprat flees Cod ABOVE", "creator": scenario_17, "species": "sprat", "expected_actions": [Action.DOWN]},
    {"name": "Sprat chooses nearest Plankton LEFT", "creator": scenario_18, "species": "sprat", "expected_actions": [Action.LEFT]},
    {"name": "Sprat eats when energy low", "creator": scenario_19, "species": "sprat", "expected_actions": [Action.EAT]},
    {"name": "Sprat flees multiple Cod", "creator": scenario_20, "species": "sprat", "expected_actions": [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]},

    # Cod (21–30)
    {"name": "Cod moves LEFT toward Herring", "creator": scenario_21, "species": "cod", "expected_actions": [Action.LEFT]},
    {"name": "Cod moves RIGHT toward Sprat", "creator": scenario_22, "species": "cod", "expected_actions": [Action.RIGHT]},
    {"name": "Cod moves UP toward Herring", "creator": scenario_23, "species": "cod", "expected_actions": [Action.UP]},
    {"name": "Cod moves DOWN toward Sprat", "creator": scenario_24, "species": "cod", "expected_actions": [Action.DOWN]},
    {"name": "Cod eats Sprat present", "creator": scenario_25, "species": "cod", "expected_actions": [Action.EAT]},
    {"name": "Cod chooses nearest prey LEFT", "creator": scenario_26, "species": "cod", "expected_actions": [Action.LEFT]},
    {"name": "Cod moves RIGHT toward diagonal Sprat", "creator": scenario_27, "species": "cod", "expected_actions": [Action.RIGHT]},
    {"name": "Cod follows Herring DOWN", "creator": scenario_28, "species": "cod", "expected_actions": [Action.DOWN]},
    {"name": "Cod chases Sprat LEFT", "creator": scenario_29, "species": "cod", "expected_actions": [Action.LEFT]},
    {"name": "Cod eats Herring when hungry", "creator": scenario_30, "species": "cod", "expected_actions": [Action.EAT]},
]

# ---------------------------------------------------------------------------
#  RUNNER (unchanged)
# ---------------------------------------------------------------------------

def run_all_scenarios(model, species, visualize=False):
    scenarios = SCENARIOS
    if species is not None:
        scenarios = [s for s in SCENARIOS if s["species"] == species]
    if species == "all":
        scenarios = SCENARIOS
    if not scenarios:
        print("No scenarios to run.")
        return 0
    print(f"Running {len(scenarios)} scenarios for species: {species}")

    num_scenarios = len(scenarios)
    if visualize:
        fig, axs = plt.subplots(num_scenarios, 3, figsize=(12, 3 * num_scenarios))
        if num_scenarios == 1:
            axs = [axs]

    total_score = 0

    for idx, scenario in enumerate(scenarios):
        world, world_data = scenario["creator"]()
        update_smell(world)
        initial_world = world.copy()

        # Get the model's predicted probability vector.
        action = get_action(model, scenario["species"], world)

        # --- Score Calculation ---
        expected = scenario.get("expected_actions", [])
        expected_indices = [ACTION_ORDER.index(a) for a in expected]
        score = sum(action[i] for i in expected_indices)
        total_score += score

        # Apply the action to update the world.
        take_step(world, world_data, scenario["species"], action)

        if not visualize:
            continue

        ax_grid_init, ax_action, ax_grid_after = axs[idx]
        grid_size = world.shape[:2]
        species_list = list(const.SPECIES_MAP.keys())


        def draw_world(ax, world, title):
            ax.set_title(title)
            ax.set_xlim(0, grid_size[1])
            ax.set_ylim(0, grid_size[0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
            ax.set_aspect('equal')
            max_biomass = max([world[:, :, const.SPECIES_MAP[s]["biomass_offset"]].max() for s in species_list] + [1e-5])
            for y in range(grid_size[0]):
                for x in range(grid_size[1]):
                    for i, species in enumerate(species_list):
                        props = const.SPECIES_MAP[species]
                        biomass = world[y, x, props["biomass_offset"]]
                        if biomass > 0:
                            radius = 0.3 * (biomass / max_biomass)
                            offset_angle = 2 * np.pi * (i / len(species_list))
                            dx = 0.15 * np.cos(offset_angle)
                            dy = 0.15 * np.sin(offset_angle)
                            circle = plt.Circle((x + 0.5 + dx, y + 0.5 + dy), radius, color=props["visualization"]["color_ones"], alpha=0.6)
                            ax.add_patch(circle)

        draw_world(ax_grid_init, initial_world, f"{scenario['name']} - Before")

        # Action bar plot
        action_labels = [a.name for a in ACTION_ORDER]
        expected_labels = [a.name for a in expected]
        ax_action.set_title(f"Action Probabilities\nExpected: {expected_labels}, Score: {score:.2f}")
        ax_action.bar(action_labels, action * 100)
        ax_action.set_ylim(0, 100)
        ax_action.set_ylabel("%")
        for i, p in enumerate(action):
            ax_action.text(i, p * 100 + 1, f"{p*100:.1f}%", ha='center', fontsize=8)

        draw_world(ax_grid_after, world, f"{scenario['name']} - After")

    total_score /= num_scenarios
    print(f"Total Score: {total_score:.2f}")
    if visualize:
        plt.tight_layout()
        plt.savefig("behaviour.png")
        plt.show()

    return total_score


