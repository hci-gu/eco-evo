"""Biomass-weighted pooled Q/B for benthivorous_fish from inputs/qb_species.csv.
Usage: python3 -I qb_pooled.py <qb_species.csv> <out_csv>
Also gives temperature-scaled variants (judgement): Q10 = 2 and the Palomares & Pauly (1998) temperature
term (-1.360 per unit of 1000/K in log10 Q/B), from a reference T to 6, 7.5 and 9 C.
"""
import csv, sys, math
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["share_pct"]) for r in rows)
res = {}
for c in ("qb_low", "qb_central", "qb_high"):
    res[c] = sum(float(r["share_pct"]) * float(r[c]) for r in rows) / tot
def pp(t_from, t_to):
    return 10 ** (-1.360 * (1000 / (t_to + 273.15) - 1000 / (t_from + 273.15)))
def q10(t_from, t_to, q=2.0):
    return q ** ((t_to - t_from) / 10)
out = [["variant", "pooled_QB_per_yr"]]
for c, v in res.items():
    out.append([c, round(v, 2)])
for tref in (8.0, 10.0):
    for t in (6.0, 7.5, 9.0):
        out.append([f"central x PalomaresPauly {tref}->{t} C", round(res["qb_central"] * pp(tref, t), 2)])
        out.append([f"central x Q10=2 {tref}->{t} C", round(res["qb_central"] * q10(tref, t), 2)])
gad = [r for r in rows if r["species"] in ("haddock", "cod")]
gt = sum(float(r["share_pct"]) for r in gad)
out.append(["gadoids only (haddock+cod) central", round(sum(float(r["share_pct"]) * float(r["qb_central"]) for r in gad) / gt, 2)])
with open(sys.argv[2], "w", newline="") as f:
    csv.writer(f).writerows(out)
for o in out:
    print(*o, sep=" | ")
