"""Tables from spatial_probe.py output.

    python3 spatial_analysis.py OUT_DIR [YEAR ...]

Per year: totals; starvation by benthivore density class, infauna
depletion class (mean infauna / K and / start), depth band and bottom-
trawl effort quartile; the starving cells against the rest (infauna,
density, eat share, gains and costs per tonne, fill, the visible
infauna against the Holling half-saturation); monthly starvation and
fill. Prints to stdout.
"""
import json
import os
import sys

import numpy as np

TPY = 1460
A, H, F, W = 0.25, 95.9, 0.2, 0.45   # infauna pair a, h, accessible share, interference


def load(out_dir, y):
    d = dict(np.load(os.path.join(out_dir, f'year_{y:02d}.npz')))
    return d


def classes(name, x, edges, d, starve_rate_ref):
    B = d['B'] / TPY
    st = d['starve']
    print(f'  by {name}:')
    print(f'    {"class":>16} {"cells":>5} {"B t":>7} {"starve t":>8} {"st/B /yr":>8} {"fill":>5} '
          f'{"inf t/km2":>9} {"inf/K":>5} {"gain/B":>7} {"cost/B":>7}')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if not m.any():
            continue
        Bs = B[m].sum()
        stv = st[m].sum()
        fill = d['Bs'][m].sum() / max(d['B'][m].sum(), 1e-9)
        inf = (d['inf_B'][m] / TPY).mean()
        infk = (d['inf_B'][m] / TPY / d['inf_K'][m]).mean()
        gb = d['gains'][m].sum() / max(d['B'][m].sum(), 1e-9) * TPY
        cb = d['cost'][m].sum() / max(d['B'][m].sum(), 1e-9) * TPY
        print(f'    {f"[{lo:g},{hi:g})":>16} {m.sum():5d} {Bs:7.0f} {stv:8.1f} '
              f'{stv / max(Bs, 1e-9):8.3f} {fill:5.2f} {inf:9.1f} {infk:5.2f} {gb:7.0f} {cb:7.0f}')


def main(out_dir, years):
    st = dict(np.load(os.path.join(out_dir, 'static.npz')))
    mon = json.load(open(os.path.join(out_dir, 'monthly.json')))
    avail = sorted(int(f[5:7]) for f in os.listdir(out_dir) if f.startswith('year_'))
    years = [y for y in (years or [1, 4, 6, 8]) if y in avail]
    depth, trawl, K = st['depth'], st['trawl'], st['inf_K']
    print(f'== {out_dir}; years available {avail}')
    print(f'depth {np.nanmin(depth):.0f}-{np.nanmax(depth):.0f} m; trawl effort mean {trawl.mean():.1f} '
          f'(max {trawl.max():.0f}); infauna K {K.min():.1f}-{K.max():.1f}')
    tq = np.quantile(trawl, [0, 0.25, 0.5, 0.75, 1.0])
    tq[-1] = trawl.max() + 1
    for y in years:
        d = load(out_dir, y)
        d['inf_K'] = K
        B = d['B'] / TPY
        Bt = B.sum()
        stv = d['starve'].sum()
        print(f'\n=== year {y}: mean benthivores {Bt:.0f} t, starvation {stv:.1f} t ({stv / Bt:.3f} /yr), '
              f'sweep {d["sweep"].sum():.2f} t; infauna mean {d["inf_B"].sum() / TPY / 3600:.1f} t/km2; '
              f'Q/B infauna {d["inf_eat"].sum() / d["B"].sum() * TPY:.2f}')
        occ = (d['occ'] > 0).sum()
        print(f'  occupied cells (B>0 at some tick) {occ}; cells with B mean > 1 t {int((B > 1).sum())}; '
              f'top-10 % cells hold {np.sort(B.ravel())[::-1][:360].sum() / Bt:.2f} of the biomass')
        rs = d['starve'] / np.maximum(B, 1e-9)
        sc = rs > 0.05
        print(f'  starving cells (starvation > 0.05 /yr of their mean B): {sc.sum()}, holding '
              f'{B[sc].sum() / Bt:.2f} of B and {d["starve"][sc].sum() / max(stv, 1e-9):.2f} of starvation')
        classes('benthivore mean density (t/cell)', B, [0, 0.5, 1, 2, 4, 8, 16, 1e9], d, None)
        infK = d['inf_B'] / TPY / K
        classes('infauna mean / K', infK, [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 2], d, None)
        classes('depth (m)', depth, [0, 100, 150, 200, 250, 300, 400, 1000], d, None)
        classes('bottom-trawl effort quartile', trawl, list(tq), d, None)
        # starving vs rest
        for lab, m in (('starving', sc), ('rest (B>0)', (~sc) & (B > 0))):
            if not m.any():
                continue
            Bm = d['B'][m].sum()
            vis = F * d['inf_B'][m] / TPY
            half = (1 + W * B[m]) / (A * H)
            print(f'  {lab:11}: cells {m.sum()}, B {B[m].sum():.0f} t, density {B[m].mean():.2f} t/cell, '
                  f'infauna {(d["inf_B"][m] / TPY).mean():.1f} t/km2 (start {st["inf_B0"][m].mean():.1f}, '
                  f'K {K[m].mean():.1f}), end-of-year infauna {d["inf_end"][m].mean():.1f}, '
                  f'visible/half-sat {np.median(vis / half):.1f} (median), '
                  f'eat {d["eatB"][m].sum() / Bm:.3f}, move {d["moveB"][m].sum() / Bm:.3f}, '
                  f'gain/B {d["gains"][m].sum() / Bm * TPY:.0f}, cost/B {d["cost"][m].sum() / Bm * TPY:.0f}, '
                  f'fill {d["Bs"][m].sum() / Bm:.2f}, low-fill share {d["Blow"][m].sum() / Bm:.2f}, '
                  f'depth {depth[m].mean():.0f}, trawl {trawl[m].mean():.1f}')
        # infauna removal vs supply per cell
        rem = d['inf_eat']
        print(f'  infauna removal by benthivores {rem.sum() / 3600:.2f} t/km2/yr; cells where removal > '
              f'r K/4 = K/4 (the logistic maximum surplus at r_max 1.0): {(rem > K / 4).sum()}')
        mm = mon['years'][y - 1]
        print('  month   B t   starve t  fill  lowfill  gain/B  cost/B (MJ/t/month)  tmult')
        for k in range(12):
            if mm['ticks'][k] == 0:
                continue
            Bm = mm['B'][k]
            print(f'   {k + 1:2}  {Bm / mm["ticks"][k]:6.0f} {mm["starve"][k]:8.1f}  {mm["Bs"][k] / Bm:.2f}  '
                  f'{mm["Blow"][k] / Bm:.3f}  {mm["gains"][k] / Bm * mm["ticks"][k]:6.0f}  '
                  f'{mm["cost"][k] / Bm * mm["ticks"][k]:6.0f}  {mm["tmult"][k] / mm["ticks"][k]:.3f}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(2)
    main(a[0], [int(x) for x in a[1:]])
