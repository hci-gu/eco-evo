"""Starvation calibration test for decision-maker FGs.

Reproduces Section 46 formula exactly: a single isolated cell with no prey,
forced rest action (pi_rest=1, pi_eat=pi_move=0), no impacts. Measures
T_total per FG until biomass falls below 5% of initial value, and compares
against the literature baseline encoded in mareld_resume.txt §46.

Mathematically mirrors lib/environments/ecosystem.py::_apply_movement and
::_apply_growth (DM branch) with pi_rest=1. No policy, no checkpoints,
no observability needed.

Usage:
    python tools/starve_calibration.py [--project mareld2.yaml] [--ticks 1500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from lib.world.tick_time import LIBRARY_TICK_HOURS
from lib.world.tick_time import DEFAULT_TICK_HOURS, ticks_per_day

# Tick length comes from project_metadata.tick_hours; the literature
# targets below are in DAYS, so this is what makes them comparable.
# Resolved per run in main() once the project path is known.
# Section 97.
TICKS_PER_DAY = ticks_per_day(DEFAULT_TICK_HOURS)

# Section 46 / 47 literature targets (days to ~95% biomass loss after start
# at half reserve, i.e. s_X = u_X). Used purely for the verdict column.
LITERATURE = {
    "zooplankton":   (3,   30),   # active copepods 3-14 d; lipid-rich up to 30+
    "pelagic_fish":  (30,  90),
    "gadoids":       (90,  180),
    "porpoises":     (3,   5),
    "seals":         (30,  60),
    "seabirds":      (5,   14),
}


def load_fg_params(project_path: Path, library_path: Path) -> dict:
    """Resolve effective per-FG parameter dicts: library defaults overridden
    by project-level fields. Returns {fg_id: params_dict} for DMs only."""
    with open(library_path, "r") as f:
        lib = yaml.safe_load(f) or {}
    with open(project_path, "r") as f:
        proj = yaml.safe_load(f) or {}

    species = lib.get("species_definitions", {}) or {}
    out = {}
    for entry in proj.get("decision_makers", []) or []:
        gid = entry.get("group_id")
        if not gid:
            continue
        base = dict(species.get(gid, {}) or {})
        # Project-level overrides (FG editor writes per-project values too).
        for k, v in entry.items():
            if k in ("group_id", "initial_biomass_min", "initial_biomass_max",
                     "inference_initial_biomass", "muted"):
                continue
            base[k] = v
        if entry.get("muted"):
            continue
        out[gid] = base
    return out


def simulate_starvation(params: dict, n_ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a single-cell FG in rest-only mode with no food.

    Returns (B_history, R_history) of length n_ticks+1. Initial state matches
    Section 46 baseline: s_X(0) = u_X (half-reserve break-even start).
    """
    ME_X    = float(params.get("max_energy_reserve", 1000.0))
    rm      = float(params.get("resting_metabolism", 0.0))
    cost_rest = float(params.get("resting_cost", 1.0) or 1.0)
    r_pos   = float(params.get("growth_rate", 0.0))
    r_starve_raw = float(params.get("starve_rate", 0.0) or 0.0)
    r_starve = r_starve_raw if r_starve_raw > 0.0 else r_pos
    u_X     = float(params.get("maintenance_level", 0.0))
    nm      = float(params.get("natural_mortality", 0.0) or 0.0)

    # Section 46 baseline assumption: start at s_X = u_X (break-even reserve).
    B = 1.0  # 1 ton; absolute scale is irrelevant for T_total (formula is scale-free).
    R = B * ME_X * u_X

    Bh = np.zeros(n_ticks + 1)
    Rh = np.zeros(n_ticks + 1)
    Bh[0], Rh[0] = B, R

    for t in range(1, n_ticks + 1):
        # _apply_movement, pi_rest=1, pi_eat=pi_move=0, no impacts.
        # r_rest = max(0, R*1 - B*1*rm*cost_rest*1)
        # b_rest = B*1
        m_rest = B * rm * cost_rest
        R = max(0.0, R - m_rest)
        # B unchanged in movement step (pi_rest only redistributes to itself).
        # Clamp R to <= B * ME_X (already satisfied by construction since R can only shrink here).

        # _apply_growth (DM branch).
        if nm > 0.0:
            # Same as ecosystem: scale both R and B by (1 - nm).
            keep = max(0.0, 1.0 - nm)
            R *= keep
            B *= keep

        s_x = (R / (B * ME_X)) if (B > 1e-12 and ME_X > 0) else 0.0
        q_x = s_x - u_X
        rate = r_pos if q_x >= 0.0 else r_starve
        growth = B * rate * q_x  # negative when q_x < 0

        # Negative-growth: drain energy reserve proportionally (mirrors ecosystem code).
        if growth < 0.0:
            total_loss = -growth
            if B > 1e-12:
                reduction = max(0.0, (B - total_loss) / (B + 1e-9))
                reduction = min(1.0, reduction)
                R *= reduction
        B = max(0.0, B + growth)

        Bh[t], Rh[t] = B, R

        if B <= 1e-9:
            # Fill remaining ticks with zeros for plotting consistency.
            Bh[t + 1:] = 0.0
            Rh[t + 1:] = 0.0
            break

    return Bh, Rh


def measure_t_total(Bh: np.ndarray, B0_frac: float = 0.05) -> int | None:
    """Return number of ticks until biomass falls below B0_frac of initial."""
    B0 = Bh[0]
    if B0 <= 0:
        return None
    thresh = B0 * B0_frac
    idx = np.argmax(Bh <= thresh)
    if Bh[idx] > thresh:
        return None  # never reached threshold within n_ticks
    return int(idx)


def measure_t_reserve_empty(Rh: np.ndarray, eps: float = 1e-6) -> int | None:
    """Tick at which energy reserve first reaches ~0 (T_1 measurement)."""
    R0 = Rh[0]
    if R0 <= 0:
        return 0
    idx = np.argmax(Rh <= R0 * eps)
    if Rh[idx] > R0 * eps:
        return None
    return int(idx)


def verdict(t_total_days: float | None, fg_id: str) -> str:
    if t_total_days is None:
        return "no collapse within window"
    lit_lo, lit_hi = LITERATURE.get(fg_id, (None, None))
    if lit_lo is None:
        return "n/a"
    if lit_lo <= t_total_days <= lit_hi:
        return f"OK (lit {lit_lo}-{lit_hi} d)"
    if t_total_days < lit_lo:
        ratio = lit_lo / max(t_total_days, 1e-9)
        return f"TOO FAST ({ratio:.1f}x; lit {lit_lo}-{lit_hi} d)"
    ratio = t_total_days / lit_hi
    return f"TOO SLOW ({ratio:.1f}x; lit {lit_lo}-{lit_hi} d)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="mareld2.yaml",
                    help="Project YAML (default: mareld2.yaml)")
    ap.add_argument("--library", default="fgconfig/fg_library.yaml",
                    help="FG library YAML (default: fgconfig/fg_library.yaml)")
    ap.add_argument("--ticks", type=int, default=1500,
                    help="Simulation horizon in ticks (default 1500 = ~1 year)")
    ap.add_argument("--collapse-frac", type=float, default=0.05,
                    help="Fraction of initial biomass that defines T_total (default 0.05)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    project_path = (root / args.project) if not Path(args.project).is_absolute() else Path(args.project)
    library_path = (root / args.library) if not Path(args.library).is_absolute() else Path(args.library)

    global TICKS_PER_DAY
    # Reads the library's own numbers, which mean LIBRARY_TICK_HOURS.
    tick_hours = LIBRARY_TICK_HOURS
    TICKS_PER_DAY = ticks_per_day(tick_hours)

    fg_params = load_fg_params(project_path, library_path)
    if not fg_params:
        print(f"No active DM FGs found in {project_path}", file=sys.stderr)
        return 1

    print("=" * 110)
    print(f"Starvation calibration test ({args.ticks} ticks of {tick_hours} h "
          f"= {args.ticks/TICKS_PER_DAY:.0f} days)")
    print(f"Project: {project_path.name} | Library: {library_path.name}")
    print(f"Start state: s_X(0) = u_X (half reserve); pi_rest=1; no prey; no impacts")
    print(f"T_total = ticks until B <= {args.collapse_frac*100:.0f}% of B(0)")
    print("=" * 110)

    header = f"{'FG':<16} {'ME_X':>7} {'rm':>6} {'starve':>7} {'T1(d)':>7} {'T_total(d)':>11}  {'verdict':<30}"
    print(header)
    print("-" * 110)

    for fg_id, p in fg_params.items():
        Bh, Rh = simulate_starvation(p, args.ticks)
        t1 = measure_t_reserve_empty(Rh)
        t_total = measure_t_total(Bh, args.collapse_frac)
        t1_days = (t1 / TICKS_PER_DAY) if t1 is not None else None
        t_total_days = (t_total / TICKS_PER_DAY) if t_total is not None else None
        v = verdict(t_total_days, fg_id)
        print(
            f"{fg_id:<16} "
            f"{float(p.get('max_energy_reserve',0)):>7.0f} "
            f"{float(p.get('resting_metabolism',0)):>6.1f} "
            f"{float(p.get('starve_rate',0) or 0):>7.3f} "
            f"{(f'{t1_days:.1f}' if t1_days is not None else '-'):>7} "
            f"{(f'{t_total_days:.1f}' if t_total_days is not None else '>window'):>11}  "
            f"{v:<30}"
        )

    print("-" * 110)
    print("Note: T1 = days to drain energy reserve from u_X to ~0 (resting metabolism only).")
    print("      T_total = days to 95% biomass loss (includes T1 + catabolism phase via starve_rate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
