"""Key pairs under reading C: separate the two changes and the two persistence rows.

Columns:
  B          reading B (273): M = 1, rho = 1 inside the tick
  C_tp       memory M with rho, but rho = 1 inside the current tick (memory alone)
  C          memory M with rho everywhere (memory + within-tick decay)
for two persistence rows of the target where they differ:
  raw   the raw 1 km density correlation (school / single-animal shot noise)
  hot   P_hot: is a rich cell still rich (planktivores 0.5/0.5/0.4/0.3/0.15;
        porpoises the upper end of the cell-presence row, 0.35 at 6 h, floor 0.3)
and the own cell (open point 1.6 of 273), at box-mean density and at x6
(about three porpoises in the cell).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_weight as mw
import library_table as lt

nw = mw.nw
HOT = {'planktivores': (0.50, 2.0, 450.0),     # ~0.5 at 6 h-1 d, 0.4 3 d, 0.3 1 wk, 0.15 1 mo
       'porpoises': (0.70, 3.0, 720.0)}        # upper end: 0.35 at 6 h, 0.3 at 1 d, floor 0.3
RAW = lt.RHO
LAGS = (6, 24, 72, 168, 720)

for k, v in HOT.items():
    r = mw.rho_fn(*v)
    print(f'hot fit {k:12s}: ' + ' '.join(f'{float(r(t)):.2f}' for t in LAGS))

PAIRS = [('porpoises', 'planktivores', 8), ('porpoises', 'benthivores', 8),
         ('planktivores', 'porpoises', 1), ('whiting', 'planktivores', 28),
         ('benthivores', 'planktivores', 28), ('seals', 'planktivores', 28)]
print('\npair                          rho   B      C_tp   C      own_B  own_C  own_C x6')
for o, t, m in PAIRS:
    rows = [('raw', RAW[t])] + ([('hot', HOT[t])] if t in HOT else [])
    for name, par in rows:
        rho = mw.rho_fn(*par)
        b = nw.w(o, t)
        ctp = mw.weight(o, t, m, rho, tick_perfect=True)
        c = mw.weight(o, t, m, rho)
        ob = mw.weight(o, t, 1, mw.rho_fn(1, 1e9, 1e9), own=True)
        oc = mw.weight(o, t, m, rho, own=True)
        oc6 = mw.weight(o, t, m, rho, own=True, density_factor=6.0)
        print(f'{o:12s} -> {t:12s} M{m:<3d}{name:4s} {b:.3f}  {ctp:.3f}  {c:.3f}  '
              f'{ob:.3f}  {oc:.3f}  {oc6:.3f}')
