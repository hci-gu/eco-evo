"""Prey-side arithmetic for WC -> benthivores: the booked and gate-implied M2 split into the juvenile (J) and adult
(A) compartments by the pair's window, against SMS 2023 key-run saithe partial M2 on haddock (2001-22) scaled to
the box's saithe density.

Usage: python3 -I prey_side.py <repo_root> <gate_hs.json>
Inputs: research_notes/"Sejens diet pa unga torskfiskar"/data/sms_keyrun2023/saithe_partial_m2_on_haddock.txt and
saithe_haddock_share_by_year.txt (extracted there from the ICES SMS 2023 key run); library booked M2 0.0833,
benthivores 1.53 t/km2, box J/B 0.15 (Arsungarnas andel), WC 0.617 t/km2 with saithe 0.80 of it (library member
shares). North Sea area 575 000 km2 [J, as in the diet note]. Booked flow split J/A in the same proportion as the
gate's encounter flow [J: the booking has no stage split].
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
G = json.load(open(sys.argv[2]))
D = os.path.join(ROOT, "research_notes", "Sejens diet på unga torskfiskar", "data", "sms_keyrun2023")
m2 = {}
for line in open(os.path.join(D, "saithe_partial_m2_on_haddock.txt")):
    p = line.split()
    if p and p[0].isdigit():
        m2[int(p[0])] = [float(x) for x in p[1:4]]
bio = {}
for line in open(os.path.join(D, "saithe_haddock_share_by_year.txt")):
    p = line.split()
    if p and p[0].isdigit():
        bio[int(p[0])] = (float(p[1]), float(p[4]))
yrs = [y for y in range(2001, 2023) if y in m2 and y in bio]
M = np.array([m2[y] for y in yrs])
Bs = np.array([bio[y][0] for y in yrs]) / 575000.0
sh = np.array([bio[y][1] for y in yrs])
print(f"SMS 2001-{yrs[-1]} ({len(yrs)} yr): saithe partial M2 on haddock age 0 {M[:, 0].mean():.3f}, age 1 {M[:, 1].mean():.3f},"
      f" age 2 {M[:, 2].mean():.3f} /yr; saithe Q1 biomass density {Bs.mean():.3f} t/km2 ({Bs.min():.2f}-{Bs.max():.2f});"
      f" haddock share of saithe food {sh.mean():.1f} %")
B_box = 0.617 * 0.80
per = M.mean(axis=0) / Bs.mean()
print(f"per t/km2 of saithe: age 0 {per[0]:.3f}, age 1 {per[1]:.3f}, age 2 {per[2]:.3f}; at the box's saithe {B_box:.3f} t/km2:"
      f" age 0 {per[0] * B_box:.3f}, age 1 {per[1] * B_box:.3f}, age 2 {per[2] * B_box:.3f} /yr")
print("SMS modelled vs observed stomach haddock share 1980-91: 9.1 vs 4.1 % (diet note, evidence_saithe entry 4): x0.45 on the"
      " SMS level if the stomachs are right")
JB, M2B, BB = 0.15, 0.0833, 1.53
print("\nPer run: implied/booked; window; J share of the encounter flow; M2 on J and on A (booked-equivalent -> implied)")
for run in ("library", "combined_central", "exp_Hs10", "exp_Hs20", "exp_Hs40", "gamma2_s6", "uniform_35", "juv_H0.5",
            "juv_H0.5_had", "gamma2_s6_juvH0.5_had"):
    r = G["gate"][run]["prey_pair"]["predatory_fish->benthivorous_fish"][0]
    fj, fa = G["edits"][run]["predatory_fish_preys_on_benthivorous_fish"]["stage_access"]
    sj = JB * fj / (JB * fj + (1 - JB) * fa)
    mj_b = M2B * BB * sj / (JB * BB)
    ma_b = M2B * BB * (1 - sj) / ((1 - JB) * BB)
    print(f"  {run:24s} ratio {r:.2f}  window [{fj}, {fa}]  J share {sj:.2f}  M2_J {mj_b:.3f} -> {r * mj_b:.3f}  "
          f"M2_A {ma_b:.4f} -> {r * ma_b:.4f}")
print("\nReading: J holds 0-group (from settlement) and 1-group haddock, cod and flatfish juveniles; saithe take haddock"
      " (cod a trace). SMS age 0 / age 1 haddock M2 from saithe at the box density is the like-for-like field-model value"
      " for M2_J on the haddock part; x0.45 for the stomach level.")
