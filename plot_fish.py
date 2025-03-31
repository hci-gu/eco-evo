import numpy as np
import lib.constants as const  # Updated import to use lib.const directly
import matplotlib.pyplot as plt
import itertools
import copy

STEPS = 200
EATING_INTERVAL = 5

def update_nested_param(d, key_path, factor):
    """
    Multiply the value at a nested key (given as a dot-separated string) by factor.
    """
    keys = key_path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] *= factor

def get_nested_value(d, key_path):
    """
    Retrieve the value at a nested key (given as a dot-separated string).
    """
    keys = key_path.split(".")
    cur = d
    for k in keys:
        cur = cur[k]
    return cur

# --- Simulation Functions ---
def simulate_fish_growth(consumption_gain, loss_rate, K, B0, steps, eating_interval):
    """
    Simulate biomass for a fish species using a discrete logistic update.
    
    Every step the biomass suffers continuous losses:
      B_after_loss = B * (1 - loss_rate)
    
    And on steps where t % eating_interval == 0, an additional growth term is applied:
      growth = consumption_gain * B * (1 - B/K)
    """
    biomass = [B0]
    for t in range(steps):
        B = biomass[-1]
        B_after_loss = B * (1 - loss_rate)
        if t % eating_interval == 0:
            growth = consumption_gain * B * (1 - B / K)
            B_next = B_after_loss + growth
        else:
            B_next = B_after_loss
        biomass.append(B_next)
    return biomass

def simulate_plankton_growth(r, K, B0, steps, eating_interval):
    """
    Simulate biomass for a plankton species (hardcoded_logic=True) using a logistic update.
    
    Here r is the growth_rate_constant from hardcoded_rules and K is the growth_threshold.
    The update is applied only every eating_interval steps.
    """
    biomass = [B0]
    for t in range(steps):
        B = biomass[-1]
        # For simplicity, we assume no continuous losses here
        B_next = B + r * B * (1 - B / K)
        biomass.append(B_next)
    return biomass

# --- Parameter Variation Setup ---
PARAM_VARIATIONS = {
    "cod": {
        "activity_metabolic_rate": [0.25, 1.0],
    }
}

# --- Main Simulation & Plotting ---
results = {}  # dictionary to hold simulation results per species

# Iterate over all species in the ecosystem.
for species_name, species_params_default in const.SPECIES_MAP.items():
    # Make a deep copy of default parameters for this species.
    default_params = copy.deepcopy(species_params_default)
    
    # Compute starting biomass per cell.
    if "starting_biomass" in default_params:
        B0_species = default_params["starting_biomass"] / (const.WORLD_SIZE * const.WORLD_SIZE)
    else:
        continue  # Skip species without starting_biomass defined.
    
    # For fish-like species, define carrying capacity based on B0.
    # Here we assume a maximum biomass (e.g. 10×B0) and then K = 0.8× that.
    if not default_params.get("hardcoded_logic", False):
        fish_max_species = B0_species * 10
        K_default = 0.8 * fish_max_species
    # For plankton, we use the growth_threshold from hardcoded_rules.
    else:
        K_default = default_params["hardcoded_rules"]["growth_threshold"]
    
    # Get parameter variation settings for this species (if any)
    species_variations = PARAM_VARIATIONS.get(species_name, {})
    variation_keys = list(species_variations.keys())
    if variation_keys:
        variation_lists = [species_variations[k] for k in variation_keys]
        combos = list(itertools.product(*variation_lists))
    else:
        combos = [()]  # No variation: one combination
    
    results[species_name] = {}
    # Iterate over each combination of multipliers.
    for combo in combos:
        # Start with the default copy.
        species_params = copy.deepcopy(default_params)
        label_parts = []
        for i, key in enumerate(variation_keys):
            factor = combo[i]
            update_nested_param(species_params, key, factor)
            # For display, get the updated value.
            updated_val = get_nested_value(species_params, key)
            label_parts.append(f"{key}={updated_val:.5f}")
        
        # Run the simulation for this variant.
        if species_params.get("hardcoded_logic", False):
            # For plankton, use its logistic growth update.
            r_val = species_params["hardcoded_rules"]["growth_rate_constant"]
            K_val = species_params["hardcoded_rules"]["growth_threshold"]
            biomass_curve = simulate_plankton_growth(r_val, K_val, B0_species, STEPS, EATING_INTERVAL)
            net_label = f"growth={r_val:.3f}"
        else:
            # For fish species, compute consumption gain and loss rate.
            consumption_gain = 0.5 * species_params["max_consumption_rate"]
            loss_rate = (species_params["activity_metabolic_rate"] +
                         species_params["standard_metabolic_rate"] +
                         species_params["natural_mortality_rate"])
            r_eff = consumption_gain - loss_rate
            # Use the carrying capacity computed earlier.
            biomass_curve = simulate_fish_growth(consumption_gain, loss_rate, K_default, B0_species, STEPS, EATING_INTERVAL)
            net_label = f"growth={r_eff:.3f}"
        
        # Construct a label for this simulation run.
        if label_parts:
            label = ", ".join(label_parts) + f", {net_label}"
        else:
            label = net_label
        results[species_name][label] = biomass_curve

# --- Plotting ---
num_species = len(results)
fig, axes = plt.subplots(num_species, 1, figsize=(10, 4 * num_species), sharex=True)
if num_species == 1:
    axes = [axes]
for i, species_name in enumerate(results):
    ax = axes[i]
    for label, curve in results[species_name].items():
        ax.plot(np.arange(STEPS + 1), curve, marker="o", markersize=3, linestyle="-", label=label)
    ax.set_title(f"{species_name.capitalize()} Growth Dynamics")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Biomass per cell")
    ax.grid(True)
    ax.legend(fontsize="small")
plt.tight_layout()
plt.show()
