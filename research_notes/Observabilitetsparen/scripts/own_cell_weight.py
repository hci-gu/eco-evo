"""Own-cell analogue of reading B (design.md section 5): P(the observer met the target
in its own cell during one 6 h tick), with the same correlated random walk, detection
radii, unit masses and predator swath as neighbour_weight.py.

    P = E[1 - exp(-n)],   n = expected encounters in the own cell

- observer searching the target: n = (c / m_unit) 2 R_obs x path in the own cell;
- target preys on the observer: the predator's swath counts too,
      n = max(own search, (c / m_unit) 2 R_pred L_pred x path in the own cell / L_obs).

Two frames for the path in the own cell:
- "track": the real 6 h track leaves the cell (CRW from a uniform start, path inside
  the start cell);
- "confined": the observer searches its own cell all tick (path = L_obs), as the engine
  keeps a cell's biomass in place and computes a whole tick's encounters there.

Two densities c: the target's box mean, and one detectable unit per km2 (a lone
porpoise, seal, school or fish in the cell).

Run: python3 -I own_cell_weight.py [central|low|high]
"""
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neighbour_weight as nw  # noqa: E402  (reads the case from sys.argv[1])


@lru_cache(maxsize=None)
def own_paths(L, D):
    """Path km inside the start cell for N_TRACKS tracks of path L, median net D."""
    rng = nw._rng("own", L, D)
    step = L / nw.STEPS
    sd = 0.0
    if D < 0.97 * L:
        lo, hi = 0.0, 3.0
        for _ in range(30):
            sd = 0.5 * (lo + hi)
            if np.median(nw._net(step, sd, 600, rng)) > D:
                lo = sd
            else:
                hi = sd
    start = rng.random((nw.N_TRACKS, 1, 2))
    ang = rng.random((nw.N_TRACKS, 1)) * 2 * np.pi + np.cumsum(
        rng.normal(0.0, sd, (nw.N_TRACKS, nw.STEPS)), axis=1)
    d = np.stack([np.cos(ang), np.sin(ang)], axis=2) * step
    cell = np.floor(start + np.cumsum(d, axis=1)).astype(int)
    return ((cell[:, :, 0] == 0) & (cell[:, :, 1] == 0)).sum(1) * step


def _units(target):
    if target == "planktivores":
        return [(0.5, nw.UNIT[target]), (0.5, nw.NIGHT_UNIT)]   # day schools, night single
    return [(1.0, nw.UNIT[target])]


def p_met(observer, target, dens, frame):
    total = 0.0
    for share, L, D in nw.OBSERVERS[observer]:
        paths = own_paths(L, D) if frame == "track" else np.array([L])
        for part, m in _units(target):
            rho = dens / m
            n = rho * 2 * nw.radius(observer, target) * paths
            if observer in nw.PREYS_ON.get(target, set()):
                n = np.maximum(n, rho * 2 * nw.radius(target, observer)
                               * nw.PATH[target] * paths / L)
            total += share * part * float((1.0 - np.exp(-n)).mean())
    return total


if __name__ == "__main__":
    print(f"case {nw.CASE}")
    for obs, targets in nw.ROWS:
        own = sum(s * own_paths(L, D).mean() for s, L, D in nw.OBSERVERS[obs])
        print(f"{obs:13s} path in the own cell (track) {own:5.2f} km of {nw.PATH[obs]:5.1f}")
        for t in targets:
            if t == "diffuse":
                continue
            row = [p_met(obs, t, dens, frame)
                   for frame in ("track", "confined")
                   for dens in (nw.DENS[t], nw.UNIT[t])]
            line = (f"{obs:13s} -> {t:13s} track: mean {row[0]:.3f} unit {row[1]:.3f}"
                    f"   confined: mean {row[2]:.3f} unit {row[3]:.3f}")
            if t == "porpoises":   # the trained runs' porpoise cells, 4-6 animals per km2
                line += f"   track at 6/km2 {p_met(obs, t, 6 * nw.UNIT[t], 'track'):.3f}"
            print(line)
