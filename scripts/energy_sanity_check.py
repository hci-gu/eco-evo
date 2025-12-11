#!/usr/bin/env python3
"""
Energy Sanity Check Script

This script simulates the bioMARL-style energy mechanics in isolation to verify
that survival is theoretically possible under ideal conditions.

It answers the question: "With the current parameter values, can a species survive
indefinitely if it has access to consistent food?"

Usage:
    python scripts/energy_sanity_check.py [options]

Options:
    --species SPECIES       Species to test (herring, sprat, cod, all) [default: all]
    --steps N               Number of simulation steps [default: 1000]
    --eating-rate RATE      Fraction of steps where eating is successful (0.0-1.0) [default: 0.2]
    --prey-available N      Number of prey individuals available per eating attempt [default: 2.0]
    --include-mortality     Include stochastic mortality in simulation [default: True]
    --find-threshold        Find minimum eating rate for survival [default: False]
    --speed-multiplier N    Speed multiplier (affects steps_per_day) [default: 1.0]
    --max-energy N          Initial energy reserves [default: 200]
    --verbose               Print detailed step-by-step output [default: False]
"""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config.settings import Settings, MAP_METER_SIZE, FISH_SWIM_SPEED, SECONDS_IN_DAY
from lib.config.species import FEEDING_ENERGY_REWARD


@dataclass
class SpeciesConfig:
    """Simplified species config for sanity check"""
    name: str
    index: int
    activity_metabolic_rate_base: float  # Before spd scaling
    standard_metabolic_rate_base: float  # Before spd scaling
    energy_cost_base: float  # Before spd scaling
    individual_weight: float
    baseline_mortality: float
    mortality_logistic_k: float
    mortality_energy_midpoint: float
    low_energy_death_rate: float
    prey_indices: list[int]  # Indices of prey in FEEDING_ENERGY_REWARD


# Species configurations (base values before spd scaling)
SPECIES_CONFIGS = {
    "herring": SpeciesConfig(
        name="herring",
        index=1,
        activity_metabolic_rate_base=0.007317884210714286,
        standard_metabolic_rate_base=0.0014635768428571428,
        # Base cost tuned to ~0.50 energy/step at spd≈2.78
        energy_cost_base=0.18,
        individual_weight=0.1,
        baseline_mortality=0.01,
        mortality_logistic_k=0.5,
        mortality_energy_midpoint=5.0,
        low_energy_death_rate=0.1,
        prey_indices=[0],  # plankton
    ),
    "sprat": SpeciesConfig(
        name="sprat",
        index=2,
        activity_metabolic_rate_base=0.007317884210714286,
        standard_metabolic_rate_base=0.0014635768428571428,
        # Base cost tuned to ~0.50 energy/step at spd≈2.78
        energy_cost_base=0.18,
        individual_weight=0.05,
        baseline_mortality=0.01,
        mortality_logistic_k=0.5,
        mortality_energy_midpoint=5.0,
        low_energy_death_rate=0.1,
        prey_indices=[0],  # plankton
    ),
    "cod": SpeciesConfig(
        name="cod",
        index=3,
        activity_metabolic_rate_base=0.014535768421428572,
        standard_metabolic_rate_base=0.0029071536857142857,
        # Base cost tuned to ~0.80 energy/step at spd≈2.78
        energy_cost_base=0.288,
        individual_weight=2.0,
        baseline_mortality=0.008,
        mortality_logistic_k=0.5,
        mortality_energy_midpoint=6.0,
        low_energy_death_rate=0.08,
        prey_indices=[1, 2],  # herring, sprat
    ),
}

# Prey individual weights for energy calculation
PREY_WEIGHTS = {
    0: 0.001,  # plankton
    1: 0.1,    # herring
    2: 0.05,   # sprat
    3: 2.0,    # cod
}


def calculate_steps_per_day(speed_multiplier: float = 1.0) -> float:
    """Calculate steps per day based on speed multiplier"""
    days_to_cross_map = MAP_METER_SIZE / (FISH_SWIM_SPEED * SECONDS_IN_DAY)
    return (days_to_cross_map / 50) * speed_multiplier


def calculate_mortality(energy: float, config: SpeciesConfig) -> float:
    """
    Calculate mortality rate based on current energy level.
    
    Uses bioMARL logistic function:
    mortality = baseline + (1 - baseline) / (1 + exp(-k * (midpoint - energy)))
    """
    # Energy-dependent mortality (logistic)
    mortality_rate = config.baseline_mortality + (1.0 - config.baseline_mortality) / (
        1.0 + np.exp(-config.mortality_logistic_k * (config.mortality_energy_midpoint - energy))
    )
    
    # Additional starvation death when energy < 1
    if energy < 1.0:
        energy_deficit = np.clip(1.0 - energy, 0.0, 100.0)
        starvation_death_prob = 1.0 - np.exp(-config.low_energy_death_rate * energy_deficit)
        mortality_rate = max(mortality_rate, starvation_death_prob)
    
    return mortality_rate


def calculate_energy_costs(config: SpeciesConfig, spd: float, is_moving: bool = True, is_eating: bool = True) -> float:
    """
    Calculate total energy cost per step.
    
    Costs:
    - Base energy cost (always applied)
    - Standard metabolic rate (always applied)
    - Activity metabolic rate (only when moving or eating)
    """
    # Scale rates by steps_per_day
    activity_rate = config.activity_metabolic_rate_base * spd
    standard_rate = config.standard_metabolic_rate_base * spd
    base_cost = config.energy_cost_base * spd
    
    total_cost = base_cost + standard_rate
    
    if is_moving:
        total_cost += activity_rate  # Movement action cost
    if is_eating:
        total_cost += activity_rate  # Eating action cost
    
    return total_cost


def calculate_energy_gain(config: SpeciesConfig, prey_individuals: float) -> float:
    """
    Calculate energy gained from eating prey individuals.
    
    Uses FEEDING_ENERGY_REWARD matrix.
    """
    total_energy = 0.0
    
    # Distribute prey evenly across available prey types
    prey_per_type = prey_individuals / len(config.prey_indices) if config.prey_indices else 0
    
    for prey_idx in config.prey_indices:
        energy_per_prey = FEEDING_ENERGY_REWARD[config.index][prey_idx]
        total_energy += prey_per_type * energy_per_prey
    
    return total_energy


def simulate_species(
    config: SpeciesConfig,
    steps: int,
    eating_rate: float,
    prey_available: float,
    include_mortality: bool,
    speed_multiplier: float,
    verbose: bool,
    max_energy: float = 100.0,
) -> dict:
    """
    Simulate energy dynamics for a single species.
    
    Returns dict with simulation results.
    """
    spd = calculate_steps_per_day(speed_multiplier)
    
    # Initial state
    energy = max_energy
    biomass = 1.0  # Normalized starting biomass
    
    # Tracking
    energy_history = [energy]
    biomass_history = [biomass]
    deaths_from_mortality = 0
    deaths_from_starvation = 0
    total_energy_gained = 0.0
    total_energy_lost = 0.0
    steps_survived = 0
    
    rng = np.random.default_rng(42)
    
    for step in range(steps):
        if biomass <= 0:
            break
            
        steps_survived = step + 1
        
        # Determine if eating this step
        is_eating = rng.random() < eating_rate
        is_moving = True  # Assume always moving
        
        # Calculate energy cost
        energy_cost = calculate_energy_costs(config, spd, is_moving, is_eating)
        total_energy_lost += energy_cost
        
        # Apply energy cost
        energy -= energy_cost
        
        # Calculate energy gain if eating
        if is_eating and prey_available > 0:
            energy_gain = calculate_energy_gain(config, prey_available)
            total_energy_gained += energy_gain
            energy += energy_gain
        
        # Cap energy at max
        energy = min(energy, max_energy)
        
        # Check for starvation death (energy <= 0)
        if energy <= 0:
            deaths_from_starvation += 1
            biomass = 0
            if verbose:
                print(f"  Step {step}: DEATH from starvation (energy={energy:.2f})")
            break
        
        # Apply mortality if enabled
        if include_mortality:
            mortality = calculate_mortality(energy, config)
            if rng.random() < mortality:
                deaths_from_mortality += 1
                biomass = 0
                if verbose:
                    print(f"  Step {step}: DEATH from mortality (energy={energy:.2f}, mortality={mortality:.2%})")
                break
            # Apply mortality to biomass (fractional death)
            biomass *= (1.0 - mortality)
        
        energy_history.append(energy)
        biomass_history.append(biomass)
        
        if verbose and step % 100 == 0:
            print(f"  Step {step}: energy={energy:.2f}, biomass={biomass:.4f}")
    
    return {
        "species": config.name,
        "steps_survived": steps_survived,
        "final_energy": energy,
        "final_biomass": biomass,
        "survived": biomass > 0,
        "min_energy": min(energy_history),
        "max_energy": max(energy_history),
        "avg_energy": np.mean(energy_history),
        "total_energy_gained": total_energy_gained,
        "total_energy_lost": total_energy_lost,
        "net_energy_per_step": (total_energy_gained - total_energy_lost) / max(steps_survived, 1),
        "deaths_from_mortality": deaths_from_mortality,
        "deaths_from_starvation": deaths_from_starvation,
        "energy_history": energy_history,
        "biomass_history": biomass_history,
        "deaths": deaths_from_mortality + deaths_from_starvation,
    }


def find_minimum_eating_rate(
    config: SpeciesConfig,
    steps: int,
    prey_available: float,
    include_mortality: bool,
    speed_multiplier: float,
    max_energy: float = 100.0,
    trials_per_rate: int = 10,
) -> float:
    """
    Binary search to find minimum eating rate for survival.
    """
    low, high = 0.0, 1.0
    
    while high - low > 0.01:
        mid = (low + high) / 2
        
        # Run multiple trials to account for stochasticity
        survivals = 0
        for _ in range(trials_per_rate):
            result = simulate_species(
                config, steps, mid, prey_available,
                include_mortality, speed_multiplier, verbose=False, max_energy=max_energy
            )
            if result["survived"]:
                survivals += 1
        
        survival_rate = survivals / trials_per_rate
        
        if survival_rate >= 0.5:
            high = mid
        else:
            low = mid
    
    return high


def calculate_theoretical_breakeven(config: SpeciesConfig, spd: float, prey_available: float) -> dict:
    """
    Calculate theoretical break-even eating rate (ignoring mortality).
    
    Returns dict with energy budget analysis.
    """
    # Energy cost per step (worst case: moving + eating)
    max_cost_per_step = calculate_energy_costs(config, spd, is_moving=True, is_eating=True)
    min_cost_per_step = calculate_energy_costs(config, spd, is_moving=True, is_eating=False)
    
    # Energy gain per eating event
    energy_per_eat = calculate_energy_gain(config, prey_available)
    
    # Break-even eating rate: cost_per_step = eating_rate * energy_per_eat
    # eating_rate = cost_per_step / energy_per_eat
    if energy_per_eat > 0:
        # When eating, we have both the cost and the gain
        # net_per_eat = energy_per_eat - activity_cost_for_eating
        activity_cost = config.activity_metabolic_rate_base * spd
        net_per_eat = energy_per_eat - activity_cost
        
        if net_per_eat > 0:
            # min_cost_per_step is cost when NOT eating
            # We need: eating_rate * net_per_eat >= min_cost_per_step
            breakeven_rate = min_cost_per_step / net_per_eat
        else:
            breakeven_rate = float('inf')  # Can't break even
    else:
        breakeven_rate = float('inf')  # No energy gain
    
    return {
        "max_cost_per_step": max_cost_per_step,
        "min_cost_per_step": min_cost_per_step,
        "energy_per_eat": energy_per_eat,
        "breakeven_eating_rate": min(breakeven_rate, 1.0) if breakeven_rate != float('inf') else float('inf'),
        "activity_cost": config.activity_metabolic_rate_base * spd,
        "standard_cost": config.standard_metabolic_rate_base * spd,
        "base_cost": config.energy_cost_base * spd,
    }


def print_energy_budget(config: SpeciesConfig, spd: float, prey_available: float):
    """Print detailed energy budget analysis"""
    analysis = calculate_theoretical_breakeven(config, spd, prey_available)
    
    print(f"\n{'='*60}")
    print(f"ENERGY BUDGET ANALYSIS: {config.name.upper()}")
    print(f"{'='*60}")
    print(f"\nSteps per day (spd): {spd:.4f}")
    print(f"\n--- COSTS PER STEP ---")
    print(f"  Base energy cost:      {analysis['base_cost']:.4f}")
    print(f"  Standard metabolic:    {analysis['standard_cost']:.4f}")
    print(f"  Activity (per action): {analysis['activity_cost']:.4f}")
    print(f"  ---")
    print(f"  Min cost (no eating):  {analysis['min_cost_per_step']:.4f}")
    print(f"  Max cost (move+eat):   {analysis['max_cost_per_step']:.4f}")
    print(f"\n--- GAINS PER EAT ---")
    print(f"  Prey available:        {prey_available:.1f} individuals")
    print(f"  Energy per eat:        {analysis['energy_per_eat']:.4f}")
    print(f"  Net gain per eat:      {analysis['energy_per_eat'] - analysis['activity_cost']:.4f}")
    print(f"\n--- BREAK-EVEN ANALYSIS ---")
    if analysis['breakeven_eating_rate'] == float('inf'):
        print(f"  ⚠️  IMPOSSIBLE TO BREAK EVEN - eating costs more than it provides!")
    elif analysis['breakeven_eating_rate'] > 1.0:
        print(f"  ⚠️  IMPOSSIBLE TO BREAK EVEN - would need to eat more than 100% of steps")
        print(f"  Theoretical rate: {analysis['breakeven_eating_rate']:.2%}")
    else:
        print(f"  ✓  Theoretical break-even rate: {analysis['breakeven_eating_rate']:.2%}")
    
    # How long can survive without eating?
    max_energy = 100.0
    steps_without_food = max_energy / analysis['min_cost_per_step']
    print(f"\n--- SURVIVAL WITHOUT FOOD ---")
    print(f"  Steps to deplete energy: {steps_without_food:.1f}")
    print(f"  (Starting from max energy of {max_energy})")


def print_mortality_analysis(config: SpeciesConfig):
    """Print mortality rate at various energy levels"""
    print(f"\n--- MORTALITY ANALYSIS ---")
    print(f"  Baseline mortality:    {config.baseline_mortality:.2%}")
    print(f"  Logistic k:            {config.mortality_logistic_k}")
    print(f"  Energy midpoint:       {config.mortality_energy_midpoint}")
    print(f"  Low energy death rate: {config.low_energy_death_rate}")
    print(f"\n  Energy Level -> Mortality Rate:")
    for energy in [100.0, 75.0, 50.0, 35.0, 25.0, 15.0, 10.0, 5.0, 1.0, 0.5, 0.1]:
        mortality = calculate_mortality(energy, config)
        status = "🟢" if mortality < 0.05 else "🟡" if mortality < 0.2 else "🔴"
        print(f"    {energy:6.1f} -> {mortality:6.2%} {status}")


DEFAULTS = {
    "steps": 1000,
    "eating_rate": 0.2,  # 20% (more realistic for RL)
    "prey_available": 2.0,  # Lowered from 10
    "include_mortality": True,
    "speed_multiplier": 1.0,
    "max_energy": 100.0,  # Keep aligned with Settings.max_energy
}

ENERGY_LEVELS = [100.0, 75.0, 50.0, 35.0, 25.0, 15.0, 10.0, 5.0, 1.0, 0.5, 0.1]


def main():
    parser = argparse.ArgumentParser(description="Energy sanity check simulation")
    parser.add_argument("--species", type=str, default="all", 
                        help="Species to test (herring, sprat, cod, all)")
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--eating-rate", "--eating_rate", dest="eating_rate",
                        type=float, default=DEFAULTS["eating_rate"])
    parser.add_argument("--prey-available", "--prey_available", dest="prey_available",
                        type=float, default=DEFAULTS["prey_available"])
    parser.add_argument("--include-mortality", "--include_mortality", dest="include_mortality",
                        action="store_true", default=DEFAULTS["include_mortality"])
    parser.add_argument("--speed-multiplier", "--speed_multiplier", dest="speed_multiplier",
                        type=float, default=DEFAULTS["speed_multiplier"])
    parser.add_argument("--max-energy", "--max_energy", dest="max_energy",
                        type=float, default=DEFAULTS["max_energy"])
    parser.add_argument("--find-threshold", "--find_threshold",
                        action="store_true",
                        dest="find_threshold",
                        help="Find minimum eating rate for survival")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed step-by-step output")
    args = parser.parse_args()
    
    spd = calculate_steps_per_day(args.speed_multiplier)
    
    print("=" * 70)
    print("ENERGY SANITY CHECK - bioMARL-style mechanics")
    print("=" * 70)
    print(f"\nSimulation Parameters:")
    print(f"  Steps:            {args.steps}")
    print(f"  Eating rate:      {args.eating_rate:.0%}")
    print(f"  Prey available:   {args.prey_available}")
    print(f"  Include mortality: {args.include_mortality}")
    print(f"  Speed multiplier: {args.speed_multiplier}")
    print(f"  Steps per day:    {spd:.4f}")
    
    # Determine which species to test
    if args.species == "all":
        species_list = list(SPECIES_CONFIGS.keys())
    else:
        if args.species not in SPECIES_CONFIGS:
            print(f"Error: Unknown species '{args.species}'")
            print(f"Available: {list(SPECIES_CONFIGS.keys())}")
            sys.exit(1)
        species_list = [args.species]
    
    # Run analysis for each species
    for species_name in species_list:
        config = SPECIES_CONFIGS[species_name]
        
        # Print energy budget
        print_energy_budget(config, spd, args.prey_available)
        
        # Print mortality analysis
        print_mortality_analysis(config)
        
        # Find threshold if requested
        if args.find_threshold:
            print(f"\n--- FINDING MINIMUM EATING RATE ---")
            min_rate = find_minimum_eating_rate(
                config, args.steps, args.prey_available,
                args.include_mortality, args.speed_multiplier, args.max_energy
            )
            print(f"  Minimum eating rate for survival: {min_rate:.1%}")
        
        # Run simulation
        print(f"\n--- SIMULATION RESULTS ---")
        result = simulate_species(
            config, args.steps, args.eating_rate, args.prey_available,
            args.include_mortality, args.speed_multiplier, args.verbose,
            args.max_energy
        )
        
        print(f"  Survived:          {'✓ YES' if result['survived'] else '✗ NO'}")
        print(f"  Steps survived:    {result['steps_survived']} / {args.steps}")
        print(f"  Final energy:      {result['final_energy']:.2f}")
        print(f"  Final biomass:     {result['final_biomass']:.4f}")
        print(f"  Min energy:        {result['min_energy']:.2f}")
        print(f"  Avg energy:        {result['avg_energy']:.2f}")
        print(f"  Net energy/step:   {result['net_energy_per_step']:.4f}")
        print(f"  Deaths:            {result['deaths']}")
        if result['deaths_from_starvation']:
            print(f"  ⚠️  Died from starvation")
        if result['deaths_from_mortality']:
            print(f"  ⚠️  Died from mortality")
        print()
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Summary table
    print(f"\n{'Species':<10} {'Break-even':<12} {'Survival':<10} {'Deaths'}")
    print("--------------------------------------------------")
    for species_name in species_list:
        config = SPECIES_CONFIGS[species_name]
        analysis = calculate_theoretical_breakeven(config, spd, args.prey_available)
        result = simulate_species(
            config, args.steps, args.eating_rate, args.prey_available,
            args.include_mortality, args.speed_multiplier, verbose=False,
            max_energy=args.max_energy
        )
        
        be_rate = analysis['breakeven_eating_rate']
        be_str = f"{be_rate:.1%}" if be_rate <= 1.0 else "∞"
        survival_str = "✓" if result['survived'] else "✗"
        
        print(f"{species_name:<10} {be_str:<12} {survival_str:<10} {result['deaths']}")


if __name__ == "__main__":
    main()
