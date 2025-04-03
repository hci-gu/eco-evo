import numpy as np
import lib.constants as const
import matplotlib.pyplot as plt
import itertools
import copy

# --- Simulation Config ---
STEPS = 500
EATING_INTERVAL = 25
ENERGY_REWARD = 75
ENERGY_DECAY = 2
ENERGY_CAP = 100
GROWTH_THRESHOLD = 50
PREDATION_INTERVAL = 10
PREDATION_RATE = 0.5

def update_nested_param(d, key_path, factor):
    keys = key_path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] *= factor

def get_nested_value(d, key_path):
    keys = key_path.split(".")
    cur = d
    for k in keys:
        cur = cur[k]
    return cur

def simulate_energy_based_growth(loss_rate, B0, growth_rate, steps, eating_interval,
                                 energy_reward=30, energy_decay=2, growth_threshold=50,
                                 predation_interval=None, predation_rate=0.0):
    """
    Simulate fish growth with fixed energy reward, energy decay, and optional predation.
    Growth rate is passed from species parameters.
    """
    biomass = [B0]
    energy = [50]

    for t in range(steps):
        B = biomass[-1]
        E = energy[-1]

        # Apply biomass decay
        B *= (1 - loss_rate)

        # Predation event
        if predation_interval and t % predation_interval == 0 and t > 0:
            B *= (1 - predation_rate)

        # Eating step
        if t % eating_interval == 0:
            E += energy_reward

        # Apply energy decay
        E -= ENERGY_DECAY
        E = max(0, min(E, ENERGY_CAP))  # Clamp between 0–100

        # Growth
        if E > growth_threshold:
            B += growth_rate * B

        biomass.append(B)
        energy.append(E)

    return biomass, energy

def plankton_growth(current, growth_rate=100, max_biomass=5000):
    current = np.asarray(current)
    growth = growth_rate * (1 - current / max_biomass)
    growth = np.maximum(growth, 0)  # Prevent negative growth
    updated = current + growth
    return np.minimum(updated, max_biomass)

def simulate_plankton_growth(B0, steps, P_max, k):
    biomass = [B0]
    for t in range(steps):
        B = biomass[-1]
        B_next = plankton_growth(B, k, P_max)
        biomass.append(B_next)
    return biomass

# --- Optional param tweaks ---
PARAM_VARIATIONS = {
}

results = {}

# Main loop over species
for species_name, species_params_default in const.SPECIES_MAP.items():
    default_params = copy.deepcopy(species_params_default)

    if "starting_biomass" not in default_params:
        continue

    B0_species = default_params["starting_biomass"] / (const.WORLD_SIZE * const.WORLD_SIZE)

    if not default_params.get("hardcoded_logic", False):
        fish_max_species = B0_species * 10
        K_default = 0.8 * fish_max_species
    else:
        K_default = default_params["hardcoded_rules"]["growth_threshold"]

    species_variations = PARAM_VARIATIONS.get(species_name, {})
    variation_keys = list(species_variations.keys())
    combos = list(itertools.product(*[species_variations[k] for k in variation_keys])) if variation_keys else [()]

    results[species_name] = {}

    for combo in combos:
        species_params = copy.deepcopy(default_params)
        label_parts = []

        for i, key in enumerate(variation_keys):
            factor = combo[i]
            update_nested_param(species_params, key, factor)
            updated_val = get_nested_value(species_params, key)
            label_parts.append(f"{key}={updated_val:.5f}")

        if species_params.get("hardcoded_logic", False):
            P_max = const.SPECIES_MAP["plankton"]["max_in_cell"]
            P_threshold = const.SPECIES_MAP["plankton"]["hardcoded_rules"]["growth_threshold"]
            k = const.SPECIES_MAP["plankton"]["hardcoded_rules"]["growth_rate_constant"]
            biomass_curve = simulate_plankton_growth(B0_species, STEPS, P_max, k)
            results[species_name]["default"] = {"biomass": biomass_curve, "energy": None}
        else:
            loss_rate = (
                species_params["activity_metabolic_rate"]
                + species_params["standard_metabolic_rate"]
                + species_params["natural_mortality_rate"]
            )

            growth_rate = species_params["growth_rate"]

            # Predation setup
            if species_name in ("herring", "sprat"):
                predation_interval = PREDATION_INTERVAL
                predation_rate = PREDATION_RATE
            else:
                predation_interval = None
                predation_rate = 0.0

            biomass_curve, energy_curve = simulate_energy_based_growth(
                loss_rate=loss_rate,
                B0=B0_species,
                growth_rate=growth_rate,
                steps=STEPS,
                eating_interval=EATING_INTERVAL,
                energy_reward=ENERGY_REWARD,
                energy_decay=ENERGY_DECAY,
                growth_threshold=GROWTH_THRESHOLD,
                predation_interval=predation_interval,
                predation_rate=predation_rate
            )

            r_eff = ENERGY_REWARD - ENERGY_DECAY - loss_rate
            label = ", ".join(label_parts) + f", net_energy_growth={r_eff:.5f}" if label_parts else f"net_energy_growth={r_eff:.5f}"
            results[species_name][label] = {"biomass": biomass_curve, "energy": energy_curve}

# --- Plotting ---
num_species = len(results)
fig, axes = plt.subplots(num_species, 1, figsize=(10, 4 * num_species), sharex=True)
if num_species == 1:
    axes = [axes]

for i, (species_name, sim_runs) in enumerate(results.items()):
    ax = axes[i]
    ax2 = ax.twinx()

    for label, data in sim_runs.items():
        time = np.arange(len(data["biomass"]))
        ax.plot(time, data["biomass"], marker='o', markersize=3, linestyle='-', label=f"{label} (biomass)")
        if data["energy"] is not None:
            ax2.plot(time, data["energy"], linestyle='--', label=f"{label} (energy)")

    ax.set_title(f"{species_name.capitalize()} Growth Dynamics")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Biomass per cell")
    ax2.set_ylabel("Energy")

    ax.grid(True)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize="small")

plt.tight_layout()
plt.show()
