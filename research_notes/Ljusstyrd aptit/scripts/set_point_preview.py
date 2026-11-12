"""Sketch of a light-entrained reserve set point s_T and a preview of its effect, from tables only.

s_T for a member species (herring):
- the clock's progress g runs from 0 at the onset (day length L_on on FALLING L) to 1 at the low
  (L_low on RISING L);
- it is linear in the photoperiod change travelled since the onset: |dL| summed down to the
  winter solstice and up again;
- recovery runs g 1 -> 0 linearly in L from L_low to L_top (rising); g = 0 from L_top on;
- s_T = 1 - depth * g.

Group value (biomass-weighted): s_T,group = 1 - share * depth * g.

Preview: the planktivores' winter months in the fix98 ledger, recomputed for the herring share if its
reserve tracks s_T. Two cases:
- (a) the policy keeps its eat share, so the cap alone acts;
- (b) the fish rest whenever the room is closed and eat just enough to stay on s_T. This is what a
  policy could learn if it can perceive its room.

Material growth is scaled by (s_T - u)/(s - u). Tables only; no rollout.
    python3 -I set_point_preview.py <repo> <ledger.json> <limits.txt>
"""
import json
import os
import sys

import numpy as np

REPO, LEDGER, LIMITS = sys.argv[1:4]
sys.path.insert(0, REPO)
import yaml  # noqa: E402
from lib.world import daylight  # noqa: E402

M = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
man = yaml.safe_load(open(os.path.join(REPO, 'mareld2.yaml')))
lat = float(man['simulation_settings']['daylight']['latitude_deg'])
L = np.asarray(daylight.day_length_hours(lat), dtype=np.float64)       # (365,) by day of year
dL = np.gradient(L)
L_ws = float(L.min())


def progress(l_on, l_low, l_top):
    """g by day of year from (L, sign dL/dt) only."""
    down = l_on - L_ws                     # photoperiod travelled from onset to the solstice
    up = l_low - L_ws                      # and from the solstice to the low
    g = np.zeros_like(L)
    for d in range(365):
        l, falling = L[d], dL[d] < 0
        if falling:
            g[d] = 0.0 if l >= l_on else (l_on - l) / (down + up)
        elif l <= l_low:
            g[d] = (down + (l - L_ws)) / (down + up)
        elif l < l_top:
            g[d] = 1.0 - (l - l_low) / (l_top - l_low)
    return g


CASES = {
    'central (Bull: onset 1 Sep, low mid-Apr)': (13.64, 14.11, 17.44),
    'Berg phase (low at the winter solstice)': (13.64, L_ws + 1e-6, 14.11),
    'late onset (1 Oct), low mid-Apr': (11.05, 14.11, 17.44),
}
SHARE, U = 0.554, 0.079
DEPTHS = (0.28, 0.5, 0.66)

d = json.load(open(LEDGER))
acc = {m: {} for m in M}
for r in d['rows']:
    if r['year'] < 1:
        continue
    g = r['groups']['planktivorous_fish']
    a = acc[r['month']]
    for k, v in g.items():
        a[k] = a.get(k, 0.0) + v
pre = {}
for line in open(LIMITS):
    p = line.split()
    if len(p) == 6 and p[0] == 'planktivorous_fish':
        pre[p[1]] = float(p[5])               # MJ/t/tick before the room cap
ME = d['meta']['ME']['planktivorous_fish']

start = np.cumsum([0] + MDAYS[:-1])
for name, (l_on, l_low, l_top) in CASES.items():
    g = progress(l_on, l_low, l_top)
    gm = np.array([g[s:s + n].mean() for s, n in zip(start, MDAYS)])
    print(f"\n== {name}: L_on {l_on:.2f} h (falling), L_low {l_low:.2f} h, L_top {l_top:.2f} h (rising)")
    print("   g by month: " + " ".join(f"{m} {v:.2f}" for m, v in zip(M, gm)))
    for depth in DEPTHS:
        sT = 1.0 - depth * gm
        dsdt = np.gradient(np.concatenate([sT[-1:], sT, sT[:1]]))[1:-1] * 12   # per year
        print(f"   depth {depth}: herring s_T " + " ".join(f"{v:.2f}" for v in sT)
              + f" | group s_T min {1 - SHARE * depth * gm.max():.2f}")
        rows = []
        for k, m in enumerate(M):
            if m not in ('Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'):
                continue
            a = acc[m]
            per = 1460.0 / a['B']
            eat = a['sh_eat'] / a['B']
            rm = a['c_eat'] * per / (1.65 * eat)                   # RM x Q10, MJ/t/yr
            cost_now = (a['c_rest'] + a['c_eat'] + a['c_move']) * per
            gain_now = a['gain'] * per
            fill = a['R'] / a['B']
            mj_per_t = gain_now / (sum(v for kk, v in a.items() if kk.startswith('t:')) * per)
            growth = gain_now - cost_now                      # realised: s ~ flat, so gain - cost ~ growth
            growth_T = max(growth, 0.0) * (sT[k] - U) / max(fill - U, 1e-9)
            draw = -ME * dsdt[k]                              # MJ/t/yr from the reserve (>0 falling)
            gain_a = max(cost_now + growth_T - draw, 0.0)
            e_pre = pre[m] * 1460.0                           # MJ/t/yr at the full eat share
            x = (rm + growth_T - draw) / (e_pre - 0.65 * rm)
            x = min(max(x, 0.0), 1.0)
            gain_b = e_pre * x
            conv = 100.0 / 365.0 / mj_per_t
            rows.append(f"{m} {gain_now * conv:.2f}->{gain_a * conv:.2f}/{gain_b * conv:.2f}")
        print("      %BW/d now -> (a) cap only / (b) rest at target: " + "  ".join(rows))
