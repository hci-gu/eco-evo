"""Encounter components per library observation pair (design.md section 5.9;
resume section 275), from the inputs of neighbour_weight.py.

Each pair gets two component lists [share, kappa], kappa in encounters per
6 h tick per t/km2 of the target's visible density (1 km cells):

- encounter_own, the confined frame (5.5): one component per observer member
  (share s) and target unit (part q):
      share = s q,  kappa = max(2 R_obs L / m, 2 R_pred L_pred / m)
  (the second term where the target preys on the observer: its swath);
- encounter_neighbour, the CRW of 1.2: share = s q P(enter), kappa = the same
  rate per km of the observer's path x the mean path inside an entered
  neighbour (L_obs replaced by the path inside, the predator term x path /
  L_obs).

A diffuse target (plankton and krill layers, bed fauna) and the own group are
met wherever present: own [[1, .inf]], neighbour [[P(enter), .inf]].
Zooplankton has movement_speed 0: own [[1, .inf]], neighbour [[0, .inf]].

Run:
  python3 -I encounter_components.py [central|low|high]            check table
  python3 -I encounter_components.py central library <fg_library.yaml>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neighbour_weight as nw  # noqa: E402  (reads the case from sys.argv[1])
from own_cell_weight import p_met  # noqa: E402

INF = float("inf")
BINS = int(os.environ.get("ENCOUNTER_BINS", "2"))


def _units(target):
    if target == "planktivores":
        return [(0.5, nw.UNIT[target]), (0.5, nw.NIGHT_UNIT)]
    return [(1.0, nw.UNIT[target])]


def _rate(observer, target, L, m):
    """Encounters per km of the observer's path per t/km2."""
    rate = 2 * nw.radius(observer, target) / m
    if observer in nw.PREYS_ON.get(target, set()):
        rate = max(rate, 2 * nw.radius(target, observer) * nw.PATH[target] / (m * L))
    return rate


def p_enter(observer):
    return sum(s * nw.visits(L, D)[0] for s, L, D in nw.OBSERVERS[observer])


def components(observer, target):
    """(own, neighbour) lists for script names; target 'diffuse' for layers."""
    if target == "diffuse" or target == observer:
        return [[1.0, INF]], [[p_enter(observer), INF]]
    own, nbr = [], []
    for s, L, D in nw.OBSERVERS[observer]:
        p_in, paths = nw.visits(L, D)
        entered = paths[paths > 0]
        # The entered path split into BINS equal-count bins: the mean path
        # alone over-states P(met) (Jensen), by up to 14 % (5.9).
        bins = np.array_split(np.sort(entered), BINS)
        for q, m in _units(target):
            r = _rate(observer, target, L, m)
            own.append([s * q, r * L])
            for b in bins:
                nbr.append([s * q * p_in * len(b) / len(entered), r * float(b.mean())])
    return own, nbr


def p_of(comps, dens):
    return sum(s * (1.0 if np.isinf(k) else 1.0 - np.exp(-k * dens)) for s, k in comps)


def library_components(observer_id, target_id):
    if observer_id == "zooplankton":
        return [[1.0, INF]], [[0.0, INF]]
    obs = nw.LIB.get(observer_id)
    if obs is None:
        return None
    if target_id == observer_id or target_id in nw.DIFFUSE:
        return components(obs, "diffuse")
    tgt = nw.LIB.get(target_id)
    return None if tgt is None else components(obs, tgt)


def _fmt(comps):
    return "[" + ", ".join(f"[{s:.4g}, {'.inf' if np.isinf(k) else f'{k:.4g}'}]"
                           for s, k in comps) + "]"


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "library":
    import yaml
    lib = yaml.safe_load(open(sys.argv[3]))["interaction_definitions"]
    for key, entry in sorted(lib.items()):
        if "_observes_" not in key or not (entry or {}).get("observes"):
            continue
        o, t = key.split("_observes_", 1)
        c = library_components(o, t)
        print(f"{key} -" if c is None else
              f"{key} encounter_own: {_fmt(c[0])} encounter_neighbour: {_fmt(c[1])}")
    sys.exit(0)

if __name__ == "__main__":
    print(f"case {nw.CASE}")
    print("observer      -> target        P(nbr) comp / exact 1.3 w   P(own) comp / exact 5.5")
    worst = 0.0
    for obs, targets in nw.ROWS:
        for t in targets:
            own, nbr = components(obs, t)
            dens = 1.0 if t == "diffuse" else nw.DENS[t]
            pn, wn = p_of(nbr, dens), nw.w(obs, t)
            po = p_of(own, dens)
            eo = 1.0 if t == "diffuse" else p_met(obs, t, dens, "confined")
            worst = max(worst, abs(pn - wn) / max(wn, 1e-3))
            print(f"{obs:13s} -> {t:13s} {pn:.3f} / {wn:.3f}            {po:.3f} / {eo:.3f}")
    print(f"largest relative neighbour error {worst:.3f}")
