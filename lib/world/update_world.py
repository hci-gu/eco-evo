import numpy as np
import warnings
from enum import Enum
from lib.model import MODEL_OFFSETS
from lib.config.const import ACTING_SPECIES
import lib.config.const as const
from lib.config.settings import Settings
from lib.config.species import SpeciesMap, get_feeding_energy_reward
from numba import njit

# Suppress underflow warnings globally - underflow to zero is expected for dying populations
np.seterr(under='ignore', divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

def all_movement_delta(
    species_map: SpeciesMap,
    world,
    world_data,
    species_key,
    actions,
    step_counter: int = 0,
    movement_deltas=None,
):
    pad = 1
    world_data[pad:-pad, pad:-pad, 4] += 1

    props = species_map[species_key]
    biomass_offset = MODEL_OFFSETS[species_key]["biomass"]
    energy_offset = MODEL_OFFSETS[species_key]["energy"]
    activity_metabolic_rate = props.activity_metabolic_rate
    standard_metabolic_rate = props.standard_metabolic_rate
    fishing_mortality_loss = props.fishing_mortality_rate
    
    # bioMARL mortality parameters
    baseline_mortality = props.baseline_mortality
    mortality_k = props.mortality_logistic_k
    mortality_midpoint = props.mortality_energy_midpoint
    low_energy_death_rate = props.low_energy_death_rate

    move_up = actions[:, :, Action.UP.value]
    move_down = actions[:, :, Action.DOWN.value]
    move_left = actions[:, :, Action.LEFT.value]
    move_right = actions[:, :, Action.RIGHT.value]

    # Normalize movement fractions so total moved mass per cell <= 1
    total_move = move_up + move_down + move_left + move_right
    overfull_mask = total_move > 1.0
    # Avoid divide by zero/overflow; clamp totals to a safe range
    safe_total = np.where(total_move <= 0, 1.0, np.clip(total_move, 1e-6, 1e6))
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        scale = np.where(overfull_mask, 1.0 / safe_total, 1.0)
    move_up *= scale
    move_down *= scale
    move_left *= scale
    move_right *= scale

    total_activity = move_up + move_down + move_left + move_right

    # --- Energy updates (bioMARL style: metabolic costs reduce energy, not biomass) ---
    initial_energy = world[pad:-pad, pad:-pad, energy_offset].copy()
    
    # Activity metabolic cost reduces energy
    activity_energy_cost = total_activity * activity_metabolic_rate
    # Standard metabolic cost reduces energy every step
    total_energy_cost = activity_energy_cost + standard_metabolic_rate
    
    # Apply energy costs
    world[pad:-pad, pad:-pad, energy_offset] -= total_energy_cost
    
    # Get updated energy for mortality calculation
    current_energy = world[pad:-pad, pad:-pad, energy_offset]

    # --- Biomass updates ---
    initial_biomass = world[pad:-pad, pad:-pad, biomass_offset].copy()

    # Apply fishing mortality separately (multiplicative)
    world[:, :, biomass_offset] *= 1 - fishing_mortality_loss

    # --- bioMARL-style energy-dependent mortality ---
    # Mortality rate increases as energy decreases using logistic function:
    # mortality = baseline + (1 - baseline) / (1 + exp(-k * (midpoint - energy)))
    # When energy is high: mortality ≈ baseline
    # When energy is low: mortality ≈ 1 (high death rate)
    # Clip the logistic exponent to avoid overflow when energy is high
    # x = -k * (midpoint - energy); large positive x => exp(x) overflows
    exp_arg = np.clip(-mortality_k * (mortality_midpoint - current_energy), -50.0, 50.0)
    mortality_rate = baseline_mortality + (1.0 - baseline_mortality) / (1.0 + np.exp(exp_arg))
    
    # Additional death rate when energy < 1 (starvation)
    # death_prob = 1 - exp(-death_rate) when energy < 1
    # Clip energy deficit to avoid overflow in exp (max deficit of 100)
    starvation_mask = current_energy < 1.0
    energy_deficit = np.clip(1.0 - current_energy, 0.0, 100.0)
    starvation_death_prob = np.where(
        starvation_mask,
        1.0 - np.exp(-low_energy_death_rate * energy_deficit),
        0.0
    )
    
    # Combine mortality: use max of energy-dependent mortality and starvation death
    total_mortality = np.maximum(mortality_rate, starvation_death_prob)
    
    # Apply mortality to biomass
    world[pad:-pad, pad:-pad, biomass_offset] *= (1.0 - total_mortality)
    
    # Zero energy = death (bioMARL: h[energy <= 0] = 0)
    zero_energy_mask = current_energy <= 0
    world[pad:-pad, pad:-pad, biomass_offset] = np.where(
        zero_energy_mask, 0.0, world[pad:-pad, pad:-pad, biomass_offset]
    )
    # Also reset energy to 0 for dead cells
    world[pad:-pad, pad:-pad, energy_offset] = np.maximum(current_energy, 0.0)

    # Reproduction is handled globally for age groups (see spawn_offspring).

    biomass_after_loss = world[pad:-pad, pad:-pad, biomass_offset]
    current_energy_for_movement = world[pad:-pad, pad:-pad, energy_offset]

    # --- Determine valid moves (based on terrain) ---
    water_offset = MODEL_OFFSETS["terrain"]["water"]
    valid_left_mask = world[: -2 * pad, pad:-pad, water_offset] == 1
    valid_right_mask = world[2 * pad :, pad:-pad, water_offset] == 1
    valid_up_mask = world[pad:-pad, : -2 * pad, water_offset] == 1
    valid_down_mask = world[pad:-pad, 2 * pad :, water_offset] == 1

    # Biomass to move in each direction
    biomass_up = biomass_after_loss * move_up * valid_up_mask
    biomass_down = biomass_after_loss * move_down * valid_down_mask
    biomass_left = biomass_after_loss * move_left * valid_left_mask
    biomass_right = biomass_after_loss * move_right * valid_right_mask

    total_biomass_moved = biomass_up + biomass_down + biomass_left + biomass_right

    # Movements (reuse buffer when provided).
    if movement_deltas is None:
        movement_deltas = np.zeros_like(world)
    else:
        movement_deltas.fill(0.0)
    movement_deltas[pad:-pad, : -2 * pad, biomass_offset] += biomass_up
    movement_deltas[pad:-pad, : -2 * pad, energy_offset] += current_energy_for_movement * biomass_up
    movement_deltas[pad:-pad, 2 * pad :, biomass_offset] += biomass_down
    movement_deltas[pad:-pad, 2 * pad :, energy_offset] += current_energy_for_movement * biomass_down
    movement_deltas[: -2 * pad, pad:-pad, biomass_offset] += biomass_left
    movement_deltas[: -2 * pad, pad:-pad, energy_offset] += (
        current_energy_for_movement * biomass_left
    )
    movement_deltas[2 * pad :, pad:-pad, biomass_offset] += biomass_right
    movement_deltas[2 * pad :, pad:-pad, energy_offset] += (
        current_energy_for_movement * biomass_right
    )

    # The stationary.
    movement_deltas[pad:-pad, pad:-pad, energy_offset] += current_energy_for_movement * (
        biomass_after_loss - total_biomass_moved
    )
    movement_deltas[pad:-pad, pad:-pad, biomass_offset] -= total_biomass_moved

    return movement_deltas

def apply_movement_delta(species_map: SpeciesMap, world, species_key, movement_deltas):
    props = species_map[species_key]
    biomass_offset = MODEL_OFFSETS[species_key]["biomass"]
    energy_offset = MODEL_OFFSETS[species_key]["energy"]

    # Apply the movement deltas to biomass and energy.
    world[:, :, biomass_offset] += movement_deltas[:, :, biomass_offset]
    denom = np.where(np.abs(world[:, :, biomass_offset]) == 0, 1, np.abs(world[:, :, biomass_offset]))
    world[:, :, energy_offset] = movement_deltas[:, :, energy_offset] / denom
    # reduce energy by base amount
    world[:, :, energy_offset] -= props.energy_cost

def build_age_transition_table(species_map: SpeciesMap):
    entries = []
    for species_key, props in species_map.items():
        if props.base_species == "plankton":
            continue
        if props.next_species is None:
            continue
        entries.append(
            (
                props.base_species,
                props.age_index,
                MODEL_OFFSETS[species_key]["biomass"],
                MODEL_OFFSETS[species_key]["energy"],
                MODEL_OFFSETS[props.next_species]["biomass"],
                MODEL_OFFSETS[props.next_species]["energy"],
                int(props.age_steps),
            )
        )
    # Sort by base species then descending age_index to avoid cascading.
    entries.sort(key=lambda item: (item[0], -item[1]))
    src_b = np.array([e[2] for e in entries], dtype=np.int32)
    src_e = np.array([e[3] for e in entries], dtype=np.int32)
    dst_b = np.array([e[4] for e in entries], dtype=np.int32)
    dst_e = np.array([e[5] for e in entries], dtype=np.int32)
    age_steps = np.array([e[6] for e in entries], dtype=np.int32)
    return src_b, src_e, dst_b, dst_e, age_steps

def build_spawn_table(species_map: SpeciesMap):
    entries = []
    for species_key, props in species_map.items():
        if not props.is_mature:
            continue
        if props.offspring_species is None:
            continue
        entries.append(
            (
                MODEL_OFFSETS[species_key]["biomass"],
                MODEL_OFFSETS[species_key]["energy"],
                MODEL_OFFSETS[props.offspring_species]["biomass"],
                MODEL_OFFSETS[props.offspring_species]["energy"],
                int(props.reproduction_freq),
                float(props.growth_rate),
            )
        )
    src_b = np.array([e[0] for e in entries], dtype=np.int32)
    src_e = np.array([e[1] for e in entries], dtype=np.int32)
    dst_b = np.array([e[2] for e in entries], dtype=np.int32)
    dst_e = np.array([e[3] for e in entries], dtype=np.int32)
    repro_freq = np.array([e[4] for e in entries], dtype=np.int32)
    growth_rate = np.array([e[5] for e in entries], dtype=np.float32)
    return src_b, src_e, dst_b, dst_e, repro_freq, growth_rate

@njit
def _apply_age_transitions_table(world, step_counter, src_b, src_e, dst_b, dst_e, age_steps):
    if step_counter <= 0:
        return
    pad = 1
    for i in range(src_b.shape[0]):
        if age_steps[i] <= 0:
            continue
        sb = src_b[i]
        se = src_e[i]
        db = dst_b[i]
        de = dst_e[i]
        # Continuous aging: move a fixed fraction each tick.
        # age_steps works as a time constant; e.g. 50 => move 2% per tick.
        transfer_fraction = 1.0 / age_steps[i]
        if transfer_fraction > 1.0:
            transfer_fraction = 1.0

        for x in range(pad, world.shape[0] - pad):
            for y in range(pad, world.shape[1] - pad):
                source_biomass = world[x, y, sb]
                moved_biomass = source_biomass * transfer_fraction
                if moved_biomass <= 0:
                    continue
                dst_biomass = world[x, y, db]
                total_biomass = dst_biomass + moved_biomass
                if total_biomass > 0:
                    combined_energy = (
                        dst_biomass * world[x, y, de] + moved_biomass * world[x, y, se]
                    ) / total_biomass
                else:
                    combined_energy = 0.0
                world[x, y, db] = total_biomass
                world[x, y, de] = combined_energy
                remaining_biomass = source_biomass - moved_biomass
                if remaining_biomass <= 1e-12:
                    world[x, y, sb] = 0.0
                    world[x, y, se] = 0.0
                else:
                    world[x, y, sb] = remaining_biomass

@njit
def _spawn_offspring_table(world, step_counter, src_b, src_e, dst_b, dst_e, repro_freq, growth_rate, max_energy):
    if step_counter <= 0:
        return
    pad = 1
    for i in range(src_b.shape[0]):
        if repro_freq[i] <= 0:
            continue
        if step_counter % repro_freq[i] != 0:
            continue
        sb = src_b[i]
        se = src_e[i]
        db = dst_b[i]
        de = dst_e[i]
        grow = growth_rate[i] - 1.0
        if grow <= 0:
            continue

        for x in range(pad, world.shape[0] - pad):
            for y in range(pad, world.shape[1] - pad):
                parent_biomass = world[x, y, sb]
                if parent_biomass <= 0:
                    continue
                new_biomass = parent_biomass * grow
                offspring_biomass = world[x, y, db]
                total_biomass = offspring_biomass + new_biomass
                if total_biomass > 0:
                    combined_energy = (
                        offspring_biomass * world[x, y, de] + new_biomass * world[x, y, se]
                    ) / total_biomass
                else:
                    combined_energy = 0.0
                if combined_energy > max_energy:
                    combined_energy = max_energy
                elif combined_energy < 0:
                    combined_energy = 0.0
                world[x, y, db] = total_biomass
                world[x, y, de] = combined_energy

def apply_age_transitions_table(world, step_counter, table):
    if table is None:
        return
    src_b, src_e, dst_b, dst_e, age_steps = table
    if src_b.size == 0:
        return
    _apply_age_transitions_table(world, step_counter, src_b, src_e, dst_b, dst_e, age_steps)

def spawn_offspring_table(world, step_counter, table, max_energy):
    if table is None:
        return
    src_b, src_e, dst_b, dst_e, repro_freq, growth_rate = table
    if src_b.size == 0:
        return
    _spawn_offspring_table(
        world, step_counter, src_b, src_e, dst_b, dst_e, repro_freq, growth_rate, max_energy
    )

def apply_age_transitions(species_map: SpeciesMap, world, step_counter: int) -> None:
    if step_counter <= 0:
        return

    pad = 1
    # Group species by base species to avoid cascading multi-step aging.
    by_base = {}
    for species_key, props in species_map.items():
        if props.base_species not in by_base:
            by_base[props.base_species] = []
        by_base[props.base_species].append((species_key, props))

    for base_name, groups in by_base.items():
        # Plankton is single-age.
        if base_name == "plankton":
            continue

        # Sort by age_index descending so new entrants don't age twice.
        groups_sorted = sorted(groups, key=lambda item: item[1].age_index, reverse=True)
        for species_key, props in groups_sorted:
            if props.next_species is None:
                continue
            if props.age_steps <= 0:
                continue
            if step_counter % props.age_steps != 0:
                continue

            src_b = MODEL_OFFSETS[species_key]["biomass"]
            src_e = MODEL_OFFSETS[species_key]["energy"]
            dst_b = MODEL_OFFSETS[props.next_species]["biomass"]
            dst_e = MODEL_OFFSETS[props.next_species]["energy"]

            src_biomass = world[pad:-pad, pad:-pad, src_b]
            src_energy = world[pad:-pad, pad:-pad, src_e]
            dst_biomass = world[pad:-pad, pad:-pad, dst_b]
            dst_energy = world[pad:-pad, pad:-pad, dst_e]

            moved_biomass = src_biomass.copy()
            if np.all(moved_biomass == 0):
                continue

            total_biomass = dst_biomass + moved_biomass
            combined_energy = np.divide(
                (dst_biomass * dst_energy + moved_biomass * src_energy),
                total_biomass,
                out=np.zeros_like(total_biomass),
                where=total_biomass > 0,
            )

            world[pad:-pad, pad:-pad, dst_b] = total_biomass
            world[pad:-pad, pad:-pad, dst_e] = combined_energy
            world[pad:-pad, pad:-pad, src_b] = 0.0
            world[pad:-pad, pad:-pad, src_e] = 0.0

def spawn_offspring(settings: Settings, species_map: SpeciesMap, world, step_counter: int) -> None:
    if step_counter <= 0:
        return

    pad = 1
    for species_key, props in species_map.items():
        if not props.is_mature:
            continue
        if props.reproduction_freq <= 0:
            continue
        if step_counter % props.reproduction_freq != 0:
            continue
        if props.offspring_species is None:
            continue

        src_b = MODEL_OFFSETS[species_key]["biomass"]
        src_e = MODEL_OFFSETS[species_key]["energy"]
        dst_b = MODEL_OFFSETS[props.offspring_species]["biomass"]
        dst_e = MODEL_OFFSETS[props.offspring_species]["energy"]

        parent_biomass = world[pad:-pad, pad:-pad, src_b]
        parent_energy = world[pad:-pad, pad:-pad, src_e]
        if np.all(parent_biomass == 0):
            continue

        growth_factor = max(props.growth_rate - 1.0, 0.0)
        new_biomass = parent_biomass * growth_factor
        if np.all(new_biomass == 0):
            continue

        offspring_biomass = world[pad:-pad, pad:-pad, dst_b]
        offspring_energy = world[pad:-pad, pad:-pad, dst_e]
        total_biomass = offspring_biomass + new_biomass
        combined_energy = np.divide(
            (offspring_biomass * offspring_energy + new_biomass * parent_energy),
            total_biomass,
            out=np.zeros_like(total_biomass),
            where=total_biomass > 0,
        )

        world[pad:-pad, pad:-pad, dst_b] = total_biomass
        world[pad:-pad, pad:-pad, dst_e] = np.clip(combined_energy, 0.0, settings.max_energy)

def matrix_perform_eating(settings: Settings, species_map: SpeciesMap, world, species_key, actions):
    pad = 1
    props = species_map[species_key]
    biomass_offset = MODEL_OFFSETS[species_key]["biomass"]
    energy_offset = MODEL_OFFSETS[species_key]["energy"]

    # Get eating actions and metabolic cost.
    eat = actions[:, :, Action.EAT.value]
    eating_energy_cost = eat * props.activity_metabolic_rate

    # Cache biomass and energy at batch positions.
    init_biomass = world[pad:-pad, pad:-pad, biomass_offset].copy()
    init_energy = world[pad:-pad, pad:-pad, energy_offset].copy()

    # --- bioMARL style: eating costs energy, not biomass ---
    # Apply eating metabolic cost to energy
    world[pad:-pad, pad:-pad, energy_offset] -= eating_energy_cost

    # Compute total eat potential.
    initial_total_eat = init_biomass * eat
    still_left_to_eat = initial_total_eat.copy()

    # Randomize prey order.
    prey_list = props.prey.copy()
    np.random.shuffle(prey_list)

    # Track total energy reward from eating (individual-based)
    total_energy_reward = np.zeros_like(init_biomass)

    # Process each prey species.
    for prey_species in prey_list:
        prey_props = species_map[prey_species]
        prey_biomass_offset = MODEL_OFFSETS[prey_species]["biomass"]
        prey_biomass = world[pad:-pad, pad:-pad, prey_biomass_offset]
        # Consume as much as possible, up to total_eat_amount.
        eat_amount = np.minimum(prey_biomass, still_left_to_eat)
        # Reduce prey biomass.
        world[pad:-pad, pad:-pad, prey_biomass_offset] -= eat_amount
        still_left_to_eat -= eat_amount

        # Calculate energy reward based on individuals eaten
        # Number of prey individuals eaten = biomass eaten / prey individual weight
        prey_individuals_eaten = eat_amount / prey_props.individual_weight
        # Energy reward per prey individual from the feeding matrix
        energy_per_prey = get_feeding_energy_reward(species_key, prey_species, species_map)
        # Total energy from eating this prey species
        total_energy_reward += prey_individuals_eaten * energy_per_prey

    # Calculate number of predator individuals to distribute energy across
    # Using the eating biomass (individuals that attempted to eat)
    eating_biomass = initial_total_eat
    predator_individuals = np.divide(
        eating_biomass,
        props.individual_weight,
        out=np.zeros_like(eating_biomass),
        where=eating_biomass > 0,
    )

    # Energy per predator individual, then scale by what fraction of biomass was eating
    # Average energy reward per individual that was eating
    energy_per_predator = np.divide(
        total_energy_reward,
        predator_individuals,
        out=np.zeros_like(total_energy_reward),
        where=predator_individuals > 0,
    )

    # Scale the energy reward by the eating percentage (how much of total biomass was eating)
    eating_percentage = np.divide(
        eating_biomass,
        init_biomass,
        out=np.zeros_like(eating_biomass),
        where=init_biomass > 0,
    )

    world[pad:-pad, pad:-pad, energy_offset] += energy_per_predator * eating_percentage

    # --- Handle low biomass: if biomass falls below minimum, set it (and energy) to zero ---
    low_biomass = world[..., biomass_offset] < props.min_biomass_in_cell
    world[low_biomass, biomass_offset] = 0
    world[low_biomass, energy_offset] = 0

    np.clip(
        world[pad:-pad, pad:-pad, energy_offset],
        0,
        settings.max_energy,
        out=world[pad:-pad, pad:-pad, energy_offset],
    )

    return world

def total_biomass(world):
    total = 0
    for species in ACTING_SPECIES:
        biomass_offset = MODEL_OFFSETS[species]["biomass"]
        total += world[:, :, biomass_offset].sum()
    return np.log(total + 1e-8)

def world_is_alive(settings: Settings, species_map: SpeciesMap, world):
    """
    Checks if the world is still "alive" by verifying the biomass of the species in the world tensor.
    The world tensor has shape (WORLD_SIZE, WORLD_SIZE, TOTAL_CHANNELS) where:
      - The first 3 channels represent the terrain (one-hot encoded).
      - The subsequent channels represent the biomass for each species.
    """
    reason = ""

    totals_by_base = {}
    starting_by_base = {}
    for species, props in species_map.items():
        if props.base_species not in const.ACTING_BASE_SPECIES:
            continue
        biomass_offset = MODEL_OFFSETS[species]["biomass"]
        biomass = world[:, :, biomass_offset].sum()
        totals_by_base[props.base_species] = totals_by_base.get(props.base_species, 0.0) + biomass
        starting_by_base[props.base_species] = (
            starting_by_base.get(props.base_species, 0.0) + props.starting_biomass
        )

    for base_species, biomass in totals_by_base.items():
        base_start = starting_by_base.get(base_species, 0.0)
        if base_start <= 0:
            continue
        biomass_too_low = biomass < (base_start * settings.min_percent_alive)
        biomass_too_high = biomass > (base_start * settings.max_percent_alive)
        if biomass_too_low or biomass_too_high:
            reason = f"{base_species} "
            if biomass_too_low:
                reason += "low"
            if biomass_too_high:
                reason += "high"
            return False, reason
    return True, ""
