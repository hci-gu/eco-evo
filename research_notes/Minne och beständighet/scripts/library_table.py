"""Reading C weights for every observation pair in the library (neighbour and own cell).

w_C = memory_weight.weight(observer, target, M, rho): the most recent visit
within the memory span M that met the target, discounted by the target's
persistence rho. Inputs (central):

- Movement: the reading-B observers' L and D (free correlated random walk).
- Memory span M (ticks), by information type:
    food / no link (observer searching the target): FOOD_M
    predator (the target eats the observer): PRED_M
  memory_mammals.md and memory_fish.md (final tables).
- Persistence rho(t) = a exp(-t/T1) + (1 - a) exp(-t/T2), t in hours, fitted
  by eye to the "raw rho in a 1 km cell" rows of persistence.md (final
  table, within season). RHO below; the fits are printed with the table.

Reading B (273) is recovered with M = 1 and rho = 1 inside the current tick.

Run: python3 -I library_table.py [LIBRARY_YAML] > ../tables/library_reading_c.txt
"""
import sys

import numpy as np
import yaml

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_weight as mw

nw = mw.nw

FOOD_M = {'porpoises': 8, 'seals': 28, 'planktivores': 8, 'mackerel': 8,
          'water column': 28, 'whiting': 28, 'benthivores': 28}
PRED_M = {'porpoises': 1, 'seals': 1, 'planktivores': 1, 'mackerel': 1,
          'water column': 1, 'whiting': 1, 'benthivores': 4}

# (a, T1 h, T2 h): rho = a exp(-t/T1) + (1-a) exp(-t/T2)
RHO = {
    'planktivores': (0.65, 2.0, 575.0),     # 0.3-0.4 at 6 h-1 d, 0.25 3 d, 0.2 1 wk, 0.1 1 mo
    'mackerel': (0.88, 4.0, 48.0),          # 0.1-0.3 at 6 h, ~0.1 at 1 d, then base
    'water column': (0.70, 4.0, 600.0),     # 0.4, 0.3, 0.2, 0.2, 0.1
    'whiting': (0.35, 6.0, 350.0),          # 0.7, 0.6, 0.4, 0.3, 0.2
    'benthivores': (0.25, 12.0, 600.0),     # 0.8, 0.7, 0.6, 0.5, 0.3
    'porpoises': (0.80, 3.0, 720.0),        # 0.1-0.35 at 6 h, floor 0.1-0.3
    'seals': (0.75, 3.0, 1000.0),           # low at 6 h, 0.1-0.3 1 d, 0.2-0.4 weeks
    'zooplankton': (0.80, 12.0, 2500.0),    # 0.4-0.8 at 6 h, 0.1-0.4 1 d, floor 0.1-0.3
    'krill': (0.25, 12.0, 900.0),           # 0.8, 0.7, 0.6, 0.5, 0.3-0.5
    'bed': (0.0, 1e9, 1e9),                 # infauna, epifauna, Nephrops: ~1
}
LAGS = (6, 24, 72, 168, 720)

LIB = {'planktivorous_fish': 'planktivores', 'mackerel': 'mackerel',
       'predatory_fish': 'water column', 'whiting': 'whiting',
       'benthivorous_fish': 'benthivores', 'porpoises': 'porpoises',
       'seals': 'seals'}
DIFFUSE_RHO = {'zooplankton': 'zooplankton', 'microzooplankton': 'zooplankton',
               'phytoplankton': 'zooplankton', 'subsurface_phytoplankton': 'zooplankton',
               'krill': 'krill', 'benthic_infauna': 'bed', 'mobile_epifauna': 'bed',
               'norway_lobster': 'bed'}


def rho_of(key):
    return mw.rho_fn(*RHO[key])


def pair(observer_id, target_id):
    """(w_B, w_C neighbour, w_C own cell, M, rho key) or None."""
    if observer_id == 'zooplankton':
        return None
    obs = LIB.get(observer_id)
    if obs is None:
        return None
    if target_id in DIFFUSE_RHO:
        tgt, rkey, enc = None, DIFFUSE_RHO[target_id], 'diffuse'
    elif target_id in LIB:
        tgt = LIB[target_id]
        rkey = tgt
        enc = 'diffuse' if tgt == obs else tgt
    else:
        return None
    pred = tgt is not None and obs in nw.PREYS_ON.get(tgt, set())
    m = PRED_M[obs] if pred else FOOD_M[obs]
    rho = rho_of(rkey)
    w_b = nw.w(obs, enc)
    w_n = mw.weight(obs, enc, m, rho)
    w_o = mw.weight(obs, enc, m, rho, own=True)
    return w_b, w_n, w_o, m, rkey


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/home/simon/eco-evo/fgconfig/fg_library.yaml'
    lib = yaml.safe_load(open(path))['interaction_definitions']
    print('persistence fits rho at 6 h / 1 d / 3 d / 1 wk / 1 mo:')
    for k, v in RHO.items():
        r = mw.rho_fn(*v)
        print(f'  {k:13s} a {v[0]:.2f} T1 {v[1]:g} h T2 {v[2]:g} h: '
              + ' '.join(f'{float(r(t)):.2f}' for t in LAGS))
    print('\npair                                          lib     B   C_nbr  C_own  M  rho')
    for key, entry in sorted(lib.items()):
        if '_observes_' not in key or not (entry or {}).get('observes'):
            continue
        o, t = key.split('_observes_', 1)
        res = pair(o, t)
        lib_w = entry.get('neighbour_weight', 1.0)
        if res is None:
            print(f'{key:45s} {lib_w:6.3f}     -      -      -')
            continue
        w_b, w_n, w_o, m, rkey = res
        print(f'{key:45s} {lib_w:6.3f} {w_b:6.3f} {w_n:6.3f} {w_o:6.3f} {m:2d}  {rkey}')


if __name__ == '__main__':
    main()
