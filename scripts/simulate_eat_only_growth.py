#!/usr/bin/env python3
"""Simulate one species in one cell when its only action is eating.

This is a local diagnostic, not a replacement for the full ecosystem model.
It mirrors the environment's predation -> feeding metabolism -> growth update
for a single predator/prey pair in a single cell.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml


def _setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mareld")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_yaml(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _project_initial(project, species_id, field):
    for section in ("decision_makers", "non_decision_makers"):
        for row in project.get(section, []) or []:
            if row.get("group_id") == species_id and field in row:
                return float(row[field])
    return None


def _interaction(library, predator_id, prey_id):
    key = f"{predator_id}_preys_on_{prey_id}"
    return (library.get("interaction_definitions", {}) or {}).get(key, {}) or {}


def _energy_level(energy_reserve, biomass, max_energy_reserve):
    if biomass <= 1e-12 or max_energy_reserve <= 0.0:
        return 0.0
    return energy_reserve / biomass / max_energy_reserve


def simulate(args):
    library = _load_yaml(args.library)
    project = _load_yaml(args.project) if args.project else {}
    species_defs = library.get("species_definitions", {}) or {}
    if args.species not in species_defs:
        raise SystemExit(f"Unknown species: {args.species}")
    if args.prey not in species_defs:
        raise SystemExit(f"Unknown prey species: {args.prey}")

    pred_cfg = species_defs[args.species]
    prey_cfg = species_defs[args.prey]
    inter = _interaction(library, args.species, args.prey)
    if not inter.get("preys_on", True):
        raise SystemExit(f"{args.species} is configured not to prey on {args.prey}")

    B = (
        float(args.biomass)
        if args.biomass is not None
        else _project_initial(project, args.species, "inference_initial_biomass")
    )
    if B is None:
        B = float(pred_cfg.get("initial_biomass", 100.0) or 100.0)

    prey_B = (
        float(args.prey_biomass)
        if args.prey_biomass is not None
        else _project_initial(project, args.prey, "inference_initial_biomass")
    )
    if prey_B is None:
        prey_B = float(prey_cfg.get("initial_biomass", 100.0) or 100.0)

    max_energy_reserve = float(pred_cfg.get("max_energy_reserve", 0.0) or 0.0)
    energy_reserve = B * max_energy_reserve * float(args.initial_energy_ratio)
    resting_metabolism = float(pred_cfg.get("resting_metabolism", 0.0) or 0.0)
    feeding_cost = float(pred_cfg.get("feeding_cost", 1.0) or 1.0)
    maintenance_level = float(pred_cfg.get("maintenance_level", 0.0) or 0.0)
    growth_rate = float(pred_cfg.get("growth_rate", 0.0) or 0.0)
    carrying_capacity = (
        float(args.carrying_capacity)
        if args.carrying_capacity is not None
        else pred_cfg.get("max_carrying_capacity")
    )
    carrying_capacity = float(carrying_capacity) if carrying_capacity is not None else None
    starvation_rate = (
        float(args.starvation_rate)
        if args.starvation_rate is not None
        else float(pred_cfg.get("starvation_rate", growth_rate) or growth_rate)
    )
    natural_mortality = float(pred_cfg.get("natural_mortality", 0.0) or 0.0)
    max_intake_rate = float(pred_cfg.get("max_intake_rate", 0.0) or 0.0)
    handling_time = float(inter.get("handling_time", 0.0) or 0.0)
    assimilation = float(inter.get("assimilation_factor", 1.0) or 1.0)
    energy_gain_per_tonne = float(inter.get("energy_gain", 0.0) or 0.0) * assimilation

    prey_growth_rate = float(prey_cfg.get("growth_rate", 0.0) or 0.0)
    prey_carrying_capacity = float(prey_cfg.get("max_carrying_capacity", prey_B) or prey_B)

    rows = []
    for tick in range(int(args.steps) + 1):
        s = _energy_level(energy_reserve, B, max_energy_reserve)
        rows.append({
            "tick": tick,
            "biomass": B,
            "prey_biomass": prey_B,
            "energy_reserve": energy_reserve,
            "energy_level": s,
            "hunger": max(0.0, 1.0 - s / 0.8),
            "intake": 0.0,
            "energy_gain": 0.0,
            "growth": 0.0,
        })
        if tick == int(args.steps):
            break
        if B <= 0.0:
            continue

        if args.mortality:
            keep = max(0.0, 1.0 - natural_mortality)
            B *= keep
            energy_reserve *= keep

        s = _energy_level(energy_reserve, B, max_energy_reserve)
        hunger = max(0.0, 1.0 - s / 0.8)
        if handling_time > 0.0:
            a_eff = max_intake_rate / (1.0 + max_intake_rate * handling_time * prey_B)
        else:
            a_eff = max_intake_rate
        demand = B * a_eff * hunger
        intake = min(demand, prey_B)
        energy_gain = intake * energy_gain_per_tonne

        if args.prey_mode in ("deplete", "logistic"):
            prey_B = max(0.0, prey_B - intake)
        if args.prey_mode == "logistic":
            prey_growth = prey_growth_rate * prey_B * (
                1.0 - prey_B / (prey_carrying_capacity + 1e-9)
            )
            prey_B = max(0.0, min(prey_carrying_capacity, prey_B + prey_growth))

        energy_reserve = max(
            0.0,
            energy_reserve - B * resting_metabolism * feeding_cost,
        ) + energy_gain
        energy_reserve = min(energy_reserve, B * max_energy_reserve)

        s = _energy_level(energy_reserve, B, max_energy_reserve)
        if s >= maintenance_level:
            growth = B * growth_rate * (s - maintenance_level)
            if carrying_capacity is not None:
                cap_factor = max(0.0, 1.0 - B / (carrying_capacity + 1e-9))
                growth = max(0.0, growth * cap_factor)
        else:
            starvation_scale = (
                (maintenance_level - s) / maintenance_level
                if maintenance_level > 0.0 else 0.0
            )
            growth = -B * starvation_rate * starvation_scale
        
        if growth < 0.0:
            loss = min(B, -growth)
            reduction = (B - loss) / (B + 1e-9)
            energy_reserve *= max(0.0, min(1.0, reduction))
        B = max(0.0, B + growth)

        rows[-1]["intake"] = intake
        rows[-1]["energy_gain"] = energy_gain
        rows[-1]["growth"] = growth

    return rows, {
        "species": args.species,
        "prey": args.prey,
        "prey_mode": args.prey_mode,
        "max_intake_rate": max_intake_rate,
        "handling_time": handling_time,
        "energy_gain_per_tonne": energy_gain_per_tonne,
        "growth_rate": growth_rate,
        "carrying_capacity": carrying_capacity if carrying_capacity is not None else 0.0,
        "starvation_rate": starvation_rate,
        "maintenance_level": maintenance_level,
    }


def write_csv(path, rows):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(path, rows, meta):
    plt = _setup_matplotlib()
    t = np.asarray([r["tick"] for r in rows], dtype=np.float64)
    biomass = np.asarray([r["biomass"] for r in rows], dtype=np.float64)
    prey = np.asarray([r["prey_biomass"] for r in rows], dtype=np.float64)
    energy_level = np.asarray([r["energy_level"] for r in rows], dtype=np.float64)
    hunger = np.asarray([r["hunger"] for r in rows], dtype=np.float64)
    intake = np.asarray([r["intake"] for r in rows], dtype=np.float64)
    growth = np.asarray([r["growth"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t, biomass, label=meta["species"], color="#1f4e79", linewidth=2)
    axes[0].plot(t, prey, label=meta["prey"], color="#6b8e23", linewidth=1.6)
    axes[0].set_ylabel("Biomass")
    axes[0].legend(frameon=False)

    axes[1].plot(t, energy_level * 100.0, color="#7b3294", linewidth=2)
    axes[1].axhline(meta["maintenance_level"] * 100.0, color="#9aa6b2", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Energy level (%)")

    axes[2].plot(t, intake, label="intake", color="#c46a1b", linewidth=1.8)
    axes[2].plot(t, growth, label="growth", color="#238b45", linewidth=1.8)
    axes[2].axhline(0.0, color="#9aa6b2", linewidth=1)
    axes[2].set_ylabel("Tonnes / tick")
    axes[2].legend(frameon=False)

    axes[3].plot(t, hunger * 100.0, color="#b2182b", linewidth=1.8)
    axes[3].set_ylabel("Hunger (%)")
    axes[3].set_xlabel("Tick")

    title = (
        f"{meta['species']} eat-only one-cell simulation, prey={meta['prey']} "
        f"({meta['prey_mode']})"
    )
    fig.suptitle(title, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate one species sitting in one cell and always eating."
    )
    parser.add_argument("--project", default="mareld2.yaml", help="Project YAML for default initial biomasses.")
    parser.add_argument("--library", default="fgconfig/fg_library.yaml", help="Functional-group library YAML.")
    parser.add_argument("--species", default="pelagic_fish", help="Focal predator species.")
    parser.add_argument("--prey", default="zooplankton", help="Prey species eaten by the focal species.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of ticks to simulate.")
    parser.add_argument("--biomass", type=float, default=None, help="Initial focal biomass. Defaults to project inference_initial_biomass.")
    parser.add_argument("--prey-biomass", type=float, default=None, help="Initial prey biomass. Defaults to project inference_initial_biomass.")
    parser.add_argument("--initial-energy-ratio", type=float, default=0.7, help="Initial energy fill ratio, 0..1.")
    parser.add_argument("--starvation-rate", type=float, default=None,
                        help="Override max biomass loss fraction per tick at zero energy.")
    parser.add_argument("--carrying-capacity", type=float, default=None,
                        help="Override focal species max carrying capacity for positive growth. Defaults to library max_carrying_capacity when present.")
    parser.add_argument("--prey-mode", choices=["fixed", "deplete", "logistic"], default="fixed",
                        help="fixed: prey availability resets each tick; deplete: consumed prey is removed; logistic: consumed prey regrows with library NDM params.")
    parser.add_argument("--mortality", action="store_true", help="Apply the species natural_mortality term each tick before growth.")
    parser.add_argument("--output", default=None, help="PNG output path. Defaults to results/eat_only_<species>_on_<prey>.png.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path for the simulated time series.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    rows, meta = simulate(args)
    output = (
        Path(args.output)
        if args.output
        else Path("results") / f"eat_only_{args.species}_on_{args.prey}.png"
    )
    plot(output, rows, meta)
    if args.csv:
        write_csv(Path(args.csv), rows)
    print(f"Wrote {output}")
    if args.csv:
        print(f"Wrote {args.csv}")
    print(
        f"start={rows[0]['biomass']:.6g} end={rows[-1]['biomass']:.6g} "
        f"end_pct={rows[-1]['biomass'] / rows[0]['biomass'] * 100.0 if rows[0]['biomass'] else 0.0:.3f}%"
    )


if __name__ == "__main__":
    main()
