"""Neighbour weight w per (observer, target) under reading B.

w = P(the observer's 6 h track enters a given neighbour cell)
    x E[1 - exp(-n) | entered]

n = expected number of encounters with the target while the observer was in that
neighbour cell:
- observer searching the target (prey, or no link):
      n = (B / m_unit) * 2 R_obs * path_inside
- target preys on the observer: the meeting is the predator's (it approaches what it
  detects and attacks; the prey learns of it then), so the predator's swath counts:
      n = (B / m_unit) * 2 R_pred * L_pred * path_inside / L_obs
  (predator swath per 6 h times the share of the 6 h the observer spent there).
B is the target's box-mean density (t/km2), m_unit one detectable unit (an animal or
a school, t), R detection radii (km), L paths per 6 h (km). A diffuse target
(plankton, krill layer, bed fauna) has n -> infinity: w = P(enter).

The track is a correlated random walk at constant speed L / 6 h, 1-min steps,
wrapped-normal turning with its sd fitted so the median 6 h net displacement is D.
Start uniform in the centre cell [0, 1]^2 km. Inputs and their grades: design.md.

Run: python3 -I neighbour_weight.py [central|low|high]
"""
import sys
from functools import lru_cache

import numpy as np

N_TRACKS = 4000
STEPS = 360  # 1-min steps over 6 h

CASE = sys.argv[1] if len(sys.argv) > 1 else "central"


def _rng(*key):
    return np.random.default_rng(abs(hash(key)) % (2 ** 32))


def _net(step, sd, n, rng):
    head = rng.random((n, 1)) * 2 * np.pi
    ang = head + np.cumsum(rng.normal(0.0, sd, (n, STEPS)), axis=1)
    return np.hypot((np.cos(ang) * step).sum(1), (np.sin(ang) * step).sum(1))


@lru_cache(maxsize=None)
def visits(L, D):
    """(P(enter a given neighbour), path km inside it, pooled over the four)."""
    rng = _rng(L, D)
    step = L / STEPS
    sd = 0.0
    if D < 0.97 * L:
        lo, hi = 0.0, 3.0
        for _ in range(30):
            sd = 0.5 * (lo + hi)
            if np.median(_net(step, sd, 600, rng)) > D:
                lo = sd
            else:
                hi = sd
    start = rng.random((N_TRACKS, 1, 2))
    ang = rng.random((N_TRACKS, 1)) * 2 * np.pi + np.cumsum(
        rng.normal(0.0, sd, (N_TRACKS, STEPS)), axis=1)
    d = np.stack([np.cos(ang), np.sin(ang)], axis=2) * step
    pos = np.concatenate([start, start + np.cumsum(d, axis=1)], axis=1)
    cell = np.floor(pos[:, 1:, :]).astype(int)
    p_in, paths = [], []
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        path = ((cell[:, :, 0] == dx) & (cell[:, :, 1] == dy)).sum(1) * step
        p_in.append((path > 0).mean())
        paths.append(path)
    return float(np.mean(p_in)), np.concatenate(paths)


# ---- inputs (central, low, high); low = less information, high = more ------------
def pick(central, low, high):
    return {"central": central, "low": low, "high": high}[CASE]


# Observer members: (share, path L km / 6 h, median net D km / 6 h).
OBSERVERS = {
    "planktivores": [(0.55, 17.0, pick(4.0, 2.0, 11.0)),   # herring
                     (0.45, 3.0, pick(0.5, 0.2, 1.0))],    # pout, sprat, blue whiting
    "mackerel": [(1.0, 29.0, pick(8.0, 5.0, 15.0))],
    "water column": [(0.80, 20.0, pick(3.0, 0.5, 8.0)),    # saithe
                     (0.10, 12.0, pick(6.0, 3.0, 10.0)),   # spurdog
                     (0.10, 1.0, 0.05)],                   # hake, ling, anglerfish
    "whiting": [(1.0, pick(3.0, 2.2, 6.5), pick(0.7, 0.3, 2.0))],
    "benthivores": [(0.31, 2.0, pick(0.3, 0.1, 0.5)),      # cod
                    (0.39, 3.0, pick(0.7, 0.3, 2.0)),      # haddock (as whiting)
                    (0.30, 0.5, 0.1)],                     # flatfish and the rest
    "porpoises": [(1.0, 14.0, pick(5.5, 4.0, 7.0))],
    "seals": [(0.50, 20.0, 5.9), (0.20, 20.0, 17.0), (0.30, 5.0, 0.1)],
}
PATH = {o: sum(s * L for s, L, _ in m) for o, m in OBSERVERS.items()}

DENS = {"planktivores": 4.72, "benthivores": 1.53, "whiting": 0.446,
        "water column": 0.617, "mackerel": 0.85, "porpoises": 84 / 3600,
        "seals": 15 / 3600}
SCHOOL = pick(5.0, 50.0, 0.5)          # t per pelagic school (planktivores, mackerel)
UNIT = {"planktivores": SCHOOL, "mackerel": SCHOOL, "benthivores": 0.0005,
        "whiting": 0.0003, "water column": 0.002, "porpoises": 0.045, "seals": 0.08}
NIGHT_UNIT = 0.0001                    # planktivores dispersed at night, one fish
FISH_R = pick(0.010, 0.005, 0.030)     # a fish's detection radius (vision, lateral line)
SONAR = {"planktivores": pick(0.20, 0.14, 0.26), "benthivores": 0.054,
         "whiting": 0.054, "water column": 0.08, "mackerel": pick(0.15, 0.10, 0.20),
         "seals": 0.14, "porpoises": 0.0}
SEAL_R = pick(0.025, 0.015, 0.035)

# Who eats whom (the model's checked pairs).
PREYS_ON = {
    "planktivores": set(),
    "mackerel": {"planktivores"},
    "water column": {"planktivores", "benthivores", "whiting", "mackerel"},
    "whiting": {"planktivores", "benthivores"},
    "benthivores": {"planktivores", "whiting"},
    "porpoises": {"planktivores", "benthivores", "whiting", "water column"},
    "seals": {"planktivores", "benthivores", "whiting", "water column"},
}


def radius(searcher, target):
    if searcher == "porpoises":
        return SONAR[target]
    if searcher == "seals":
        return SEAL_R
    return FISH_R


def w(observer, target):
    if target == "diffuse":
        return sum(s * visits(L, D)[0] for s, L, D in OBSERVERS[observer])
    units = [(1.0, UNIT[target])]
    if target == "planktivores":
        units = [(0.5, UNIT[target]), (0.5, NIGHT_UNIT)]   # day schools, night single
    total = 0.0
    for share, L, D in OBSERVERS[observer]:
        _, paths = visits(L, D)
        for part, m in units:
            rho = DENS[target] / m
            # The observer's own search, and where the target preys on it,
            # the predator's swath; the larger of the two (mutual pairs).
            n = rho * 2 * radius(observer, target) * paths
            if observer in PREYS_ON.get(target, set()):
                n = np.maximum(n, rho * 2 * radius(target, observer)
                               * PATH[target] * paths / L)
            total += share * part * float((1.0 - np.exp(-n)).mean())
    return total


ROWS = [
    ("planktivores", ["diffuse", "benthivores", "whiting", "water column", "mackerel",
                      "porpoises", "seals"]),
    ("mackerel", ["diffuse", "planktivores", "water column"]),
    ("water column", ["diffuse", "planktivores", "benthivores", "whiting", "mackerel",
                      "porpoises", "seals"]),
    ("whiting", ["diffuse", "planktivores", "benthivores", "water column",
                 "porpoises", "seals"]),
    ("benthivores", ["diffuse", "planktivores", "whiting", "water column",
                     "porpoises", "seals"]),
    ("porpoises", ["planktivores", "benthivores", "whiting", "water column"]),
    ("seals", ["planktivores", "benthivores", "whiting", "water column"]),
]

# Library ids -> the script's names; diffuse targets (layers, bed fauna).
LIB = {"planktivorous_fish": "planktivores", "mackerel": "mackerel",
       "predatory_fish": "water column", "whiting": "whiting",
       "benthivorous_fish": "benthivores", "porpoises": "porpoises",
       "seals": "seals"}
DIFFUSE = {"zooplankton", "krill", "benthic_infauna", "mobile_epifauna",
           "norway_lobster", "phytoplankton", "microzooplankton",
           "subsurface_phytoplankton"}


def library_weight(observer_id, target_id):
    """w for a library pair, or None where it is not derived here."""
    if observer_id == "zooplankton":
        return 0.0            # movement_speed 0: it never ranges (232.4)
    obs = LIB.get(observer_id)
    if obs is None:
        return None
    if target_id == observer_id or target_id in DIFFUSE:
        return w(obs, "diffuse")
    tgt = LIB.get(target_id)
    return None if tgt is None else w(obs, tgt)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "library":
    import yaml
    lib = yaml.safe_load(open(sys.argv[3]))["interaction_definitions"]
    for key, entry in sorted(lib.items()):
        if "_observes_" not in key or not (entry or {}).get("observes"):
            continue
        o, t = key.split("_observes_", 1)
        value = library_weight(o, t)
        print(f"{key} {'-' if value is None else f'{value:.3f}'}")
    sys.exit(0)

if __name__ == "__main__":
    print(f"case {CASE}")
    for obs, targets in ROWS:
        for t in targets:
            print(f"{obs:13s} -> {t:13s} w {w(obs, t):.3f}")
