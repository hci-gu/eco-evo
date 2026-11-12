"""Day length at the box and the effect size of a light-cued appetite, from existing tables.

Reads the fix97 ledger (research_notes/"Ransonen i energi"/tables/ledger_fix97.json) and
prints: day length by month at the manifest latitude; the planktivores' and whiting's monthly
krill flow; and the krill flow if (i) the whole planktivore group, (ii) only its herring share
(0.554, pout keeps eating as realised), ate the field winter krill ration in Nov-Mar.
Field krill rations (%BW/d) from research_notes/"Krillparen mot vintermagar": group 0.24
(0.10-0.46); herring 0.10 (0.03-0.53, Onsrud 2004 herring >25 cm; Baltic 0); whiting 0.30 (0.14-0.50).
python3 -I effect_size.py <repo>
"""
import json, os, sys
import numpy as np
REPO = sys.argv[1]
sys.path.insert(0, REPO)
import yaml
from lib.world import daylight

man = yaml.safe_load(open(os.path.join(REPO, 'mareld2.yaml')))
lat = float(man['simulation_settings']['daylight']['latitude_deg'])
L = daylight.day_length_hours(lat)
mid = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
M = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(f"latitude {lat}: day length (h) at mid-month")
print("  " + "  ".join(f"{m} {float(L[d]):.1f}" for m, d in zip(M, mid)))
dL = [float(L[(d + 1) % 365] - L[d - 1]) / 2 * 60 for d in mid]
print("  change (min/day): " + "  ".join(f"{m} {v:+.1f}" for m, v in zip(M, dL)))

d = json.load(open(os.path.join(REPO, 'research_notes/Ransonen i energi/tables/ledger_fix97.json')))
win = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
tot = 0.0
krill = {f: {m: 0.0 for m in M} for f in ('planktivorous_fish', 'whiting')}
Bm = {f: {m: 0.0 for m in M} for f in krill}
ticks = {m: 0 for m in M}
for r in d['rows']:
    if r['year'] < 1:
        continue
    for f, g in r['groups'].items():
        tot += g.get('t:krill', 0.0) / 2
        if f in krill:
            krill[f][r['month']] += g.get('t:krill', 0.0) / 2
            Bm[f][r['month']] += g['B'] / 2
    ticks[r['month']] += r['groups']['whiting']['ticks'] / 2
print(f"\nfive-group krill flow (years 2-3 mean): {tot:.0f} t/yr")
for f, field in (('planktivorous_fish', (0.24, 0.10, 0.46)), ('whiting', (0.30, 0.14, 0.50))):
    w = sum(krill[f][m] for m in win)
    rates = {m: krill[f][m] / Bm[f][m] * ticks[m] / ticks[m] * 4 * 100 for m in win}  # %BW/d (B summed per tick)
    meanB = {m: Bm[f][m] / ticks[m] for m in win}
    days = {m: ticks[m] / 4 for m in win}
    cf = {k: sum(v / 100 * meanB[m] * days[m] for m in win) for k, v in zip(('c', 'lo', 'hi'), field)}
    print(f"{f}: Nov-Mar krill {w:.0f} t ({w / tot:.3f} of the total); model Nov-Mar krill ration "
          + ", ".join(f"{m} {rates[m]:.2f}" for m in win) + " %BW/d")
    print(f"   whole group at the field krill ration: {cf['c']:.0f} t ({cf['lo']:.0f}-{cf['hi']:.0f}); "
          f"total flow {(tot - w + cf['c']) / tot - 1:+.1%} ({(tot - w + cf['hi']) / tot - 1:+.1%} to {(tot - w + cf['lo']) / tot - 1:+.1%})")
    if f == 'planktivorous_fish':
        share = 0.554
        for lab, hr in (('central 0.10', 0.10), ('low 0.03', 0.03), ('high 0.53', 0.53)):
            new = (1 - share) * w + share * sum(hr / 100 * meanB[m] * days[m] for m in win)
            print(f"   herring share only (0.554) at {lab} %BW/d, pout as realised: {new:.0f} t; "
                  f"total flow {(tot - w + new) / tot - 1:+.1%}")
