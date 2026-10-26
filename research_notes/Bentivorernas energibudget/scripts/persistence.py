"""Where the starving cells come from and whether they persist.

    python3 persistence.py OUT_DIR

Starving cell: starvation > 0.05 /yr of the cell's mean benthivore
biomass. Prints the overlap of the starving masks across years 4, 6, 8;
their overlap with the spawn colonies (ben_B0) and with the year-1
densest cells; how the infauna in them evolved year by year; and the
infauna's recovery time from the depleted level if the fish left
(logistic at the loaded gross r and the trawl loss of the cell).
"""
import os
import sys

import numpy as np

TPY = 1460
R_GROSS = 1.0 + 0.029      # infauna r_max + checked M2 (/yr), the derived g x 1460
F_TRAWL_MEAN = 0.136       # /yr at the map's mean effort (library table)


def main(out_dir):
    st = dict(np.load(os.path.join(out_dir, 'static.npz')))
    avail = sorted(int(f[5:7]) for f in os.listdir(out_dir) if f.startswith('year_'))
    Y = {y: dict(np.load(os.path.join(out_dir, f'year_{y:02d}.npz'))) for y in avail}
    K, trawl = st['inf_K'], st['trawl']
    masks = {}
    for y, d in Y.items():
        B = d['B'] / TPY
        masks[y] = (d['starve'] / np.maximum(B, 1e-9)) > 0.05
    print('starving cells per year:', {y: int(m.sum()) for y, m in masks.items()})
    picks = [y for y in (4, 6, 8) if y in masks]
    for a in picks:
        for b in picks:
            if b > a:
                inter = (masks[a] & masks[b]).sum()
                print(f'  year {a} starving cells also starving in year {b}: {inter} of {masks[a].sum()}')
    B0 = st['ben_B0']
    col = B0 > np.quantile(B0[B0 > 0], 0.75) if (B0 > 0).any() else B0 > 0
    top1 = Y[1]['B'] / TPY
    top1m = top1 >= np.sort(top1.ravel())[::-1][359]
    last = max(picks) if picks else max(avail)
    for y in picks:
        m = masks[y]
        if m.sum() == 0:
            continue
        print(f'  year {y}: share of starving cells that were spawn-colony cells (top quartile of the '
              f'start biomass) {(m & col).sum() / m.sum():.2f} (grid share {col.mean():.2f}); in the year-1 '
              f'top-10 % density cells {(m & top1m).sum() / m.sum():.2f} (grid share 0.10); '
              f'start benthivore density there {B0[m].mean():.2f} vs grid {B0.mean():.2f} t/cell')
    m = masks[last]
    print(f'\ninfauna (t/km2, yearly mean) and benthivore density (t/cell) in the year-{last} starving cells:')
    print('  year  infauna  inf/K   ben   rest-of-grid infauna  ben')
    for y, d in Y.items():
        inf = d['inf_B'] / TPY
        B = d['B'] / TPY
        print(f'  {y:4}  {inf[m].mean():7.1f}  {(inf[m] / K[m]).mean():5.2f}  {B[m].mean():5.2f}   '
              f'{inf[~m].mean():7.1f}               {B[~m].mean():5.2f}')
    # recovery from the depleted level if the fish left
    d = Y[last]
    inf_end = d['inf_end'][m]
    Fm = F_TRAWL_MEAN * trawl[m] / trawl.mean()
    r = R_GROSS - Fm
    Keff = K[m] * (1 - Fm / R_GROSS)
    target = 0.5 * Keff
    x0 = np.maximum(inf_end, 1e-3)
    # logistic dx/dt = r x (1 - x/Keff): t = ln[(target (Keff - x0)) / (x0 (Keff - target))] / r
    t = np.log(target * (Keff - x0) / (x0 * (Keff - target))) / r
    t = np.where(x0 >= target, 0.0, t)
    print(f'\nrecovery to half the trawled equilibrium if the fish left: median {np.median(t):.1f} yr '
          f'(10-90 % {np.quantile(t, 0.1):.1f}-{np.quantile(t, 0.9):.1f}); end-of-year infauna median '
          f'{np.median(inf_end):.2f} t/km2')
    # demand vs supply in those cells
    dem = d['inf_eat'][m]
    print(f'infauna removal by benthivores there {dem.mean():.1f} t/km2/yr vs the logistic maximum surplus '
          f'r K/4 = {np.mean(r * Keff / 4):.1f}; benthivore density needed to exceed it at Q/B 7.7: '
          f'{np.mean(r * Keff / 4) / 7.7:.2f} t/cell')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
