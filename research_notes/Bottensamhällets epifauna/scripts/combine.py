"""Per-cell epifauna stock on the Mareld grid and the combined benthic_community numbers.
Inputs: maps/depth.npz, data/uwtv_survey_share_per_cell.npy. All values g WW/m2 = t/km2."""
import sys
sys.path.insert(0, '/home/simon/eco-evo/tools')
import numpy as np
from make_fishing_maps import wgs84_to_sweref99tm, E0, N0, CELL, GRID
base = sys.argv[1]
dep = np.load('/home/simon/eco-evo/maps/depth.npz')['depth']
sv = np.load(base + '/data/uwtv_survey_share_per_cell.npy')
sv = np.nan_to_num(sv)
# cell-centre lon/lat by forward projection of a fine mesh
LO, LA = np.meshgrid(np.linspace(10.0, 11.25, 1000), np.linspace(57.75, 58.36, 1000))
E, N = wgs84_to_sweref99tm(LO.ravel(), LA.ravel())
c = np.floor((E - E0) / CELL).astype(int); r = GRID - 1 - np.floor((N - N0) / CELL).astype(int)
ok = (c >= 0) & (c < 60) & (r >= 0) & (r < 60)
n = np.zeros((60, 60)); clon = np.zeros((60, 60)); clat = np.zeros((60, 60))
np.add.at(n, (r[ok], c[ok]), 1); np.add.at(clon, (r[ok], c[ok]), LO.ravel()[ok]); np.add.at(clat, (r[ok], c[ok]), LA.ravel()[ok])
clon /= n; clat /= n
S = np.where(clon < 10.71, 1, np.where(clat > 57.83 + (clon - 10.71) * (0.17 / 0.49), 8, 9))
def lin(a, b): return np.interp(dep, [100, 300], [a, b])
cases = {  # name: (nephrops burrow dens by stratum, g/animal, pandalus q, other@100, other@300)
    'low':     ({1: 0.19, 8: 0.06, 9: 0.32}, 23, 0.55, 0.5, 0.3),
    'central': ({1: 0.26, 8: 0.09, 9: 0.55}, 30, 0.40, 1.4, 1.0),
    'high':    ({1: 0.33, 8: 0.10, 9: 0.55}, 45, 0.17, 4.7, 3.3),
}
# high Nephrops density = 2025 (2024 x 1.73)
cases['high'][0].update({1: 0.19 * 1.73, 8: 0.06 * 1.73, 9: 0.32 * 1.73})
pand_raw = np.select([dep < 200, dep < 300, dep < 500], [0.12, 0.57, 0.35], 0.0)  # survey t/km2, strata 14/15/16 2017-21
r_i = {'inf': 1.0, 'nep': 0.34, 'pan': 0.85, 'oth': 1.2}
f_i = {'inf': 0.2, 'nep': 0.10, 'pan': 0.7, 'oth': 0.6}
inf = lin(110, 55)
F, M2 = 0.226, 0.0446
for name, (dens, w, q, o1, o3) in cases.items():
    nep = sum(sv * (S == s) * d for s, d in dens.items()) * w
    pan = pand_raw / q
    oth = lin(o1, o3)
    epi = nep + pan + oth
    print(f'== {name}: box means  nephrops {nep.mean():.2f}  pandalus {pan.mean():.2f}  other {oth.mean():.2f}  epifauna {epi.mean():.2f}  infauna {inf.mean():.1f}')
    for lo, hi in [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 600)]:
        m = (dep >= lo) & (dep < hi)
        print(f'   {lo}-{hi} m: nep {nep[m].mean():.2f} pan {pan[m].mean():.2f} oth {oth[m].mean():.2f} epi {epi[m].mean():.2f}')
    # least-squares linear table (clamped outside 100-300) for the epifauna
    x = np.clip(dep, 100, 300).ravel(); t = (x - 100) / 200
    A = np.vstack([1 - t, t]).T
    a, b = np.linalg.lstsq(A, epi.ravel(), rcond=None)[0]
    print(f'   LSQ linear epifauna anchors: 100 m {a:.2f}, 300 m {b:.2f}; box mean of table {lin(a, b).mean():.2f}')
    comps = {'inf': inf, 'nep': nep, 'pan': pan, 'oth': oth}
    tot = sum(comps.values())
    rw = sum(r_i[k] * v for k, v in comps.items()) / tot
    fw = sum(f_i[k] * v for k, v in comps.items()) / tot
    share = 1 - (F + M2) / rw
    print(f'   combined box mean {tot.mean():.1f}; r_w box {(rw*tot).sum()/tot.sum():.3f}; f_w box {(fw*tot).sum()/tot.sum():.3f}')
