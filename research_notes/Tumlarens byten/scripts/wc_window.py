"""Porpoise prey-size window on the water-column rest (predatory_fish since section 228).

Usage: python3 -I wc_window.py <pred_length_biomass.csv>

Input: "../../Fiskgrupper i rutan/predators/tables/pred_length_biomass.csv" (box, long-run 2001-25
stratified GOV kg/h per 5-cm class, pooled Q1+Q3; lower class edge in cm).
The rest's member weights are the library's (section 228, params_split_mack.md): saithe 0.797,
spurdog 0.097, hake 0.048, anglerfish 0.027, ling 0.014, velvet belly 0.009, pollack 0.008.
Porpoises swallow fish whole; prey mostly < 30 cm, practical limit ~40 cm, largest recorded 49-51 cm
(Aarefjord et al. 1995; MacLeod et al. 2006; Santos & Pierce 2003; "../../Arsungarnas andel"
evidence_prey_size.md 3.1). Sharks (spurdog, velvet belly) and anglerfish never appear in any
porpoise stomach set read (Borjesson 2003, Aarefjord 1995, Andreasen 2017, Santos 2004, Ransijn
2021), so they get 0 whatever their length. A 5-cm class counts as inside the window when its
upper edge is <= the limit (so "< 40" = classes 0-35).
"""
import csv
import sys
from collections import defaultdict

W = {"saithe": 0.797, "spurdog": 0.097, "hake": 0.048, "anglerfish": 0.027, "ling": 0.014,
     "velvet belly": 0.009, "pollack": 0.008}
EATEN = {"saithe", "hake", "ling", "pollack"}
tot = defaultdict(float)
inside = defaultdict(lambda: defaultdict(float))
for r in csv.DictReader(open(sys.argv[1])):
    sp, lo, b = r["common"], float(r["L5"]), float(r["pooled"])
    if sp not in W:
        continue
    tot[sp] += b
    for lim in (25, 30, 35, 40, 45, 50):
        if lo + 5 <= lim:
            inside[sp][lim] += b
print("Biomass share inside the window, per member (box GOV, long-run):")
print(f"  {'member':14s} {'weight':>6s} " + " ".join(f"<{lim:>3d}" for lim in (25, 30, 35, 40, 45, 50)))
for sp in W:
    print(f"  {sp:14s} {W[sp]:6.3f} " + " ".join(f"{inside[sp][lim] / tot[sp]:4.3f}" for lim in (25, 30, 35, 40, 45, 50)))
print("\nGroup accessible fraction f = sum_m w_m * share_m(<limit) (eaten taxa only; sharks, anglerfish 0):")
for lim in (25, 30, 35, 40, 45, 50):
    f = sum(W[s] * inside[s][lim] / tot[s] for s in W if s in EATEN)
    f_all = sum(W[s] * inside[s][lim] / tot[s] for s in W)
    parts = ", ".join(f"{s} {W[s] * inside[s][lim] / tot[s] / f:.2f}" for s in W if s in EATEN and f > 0)
    print(f"  < {lim} cm: f {f:.4f} (all members incl. sharks {f_all:.4f}); composition of the accessible part: {parts}")
B_WC = 0.617
for lim in (30, 40, 45):
    f = sum(W[s] * inside[s][lim] / tot[s] for s in W if s in EATEN)
    print(f"  accessible biomass at < {lim} cm: {B_WC * f:.4f} t/km2 of {B_WC} ({B_WC * f * 3600:.0f} t on the grid)")
