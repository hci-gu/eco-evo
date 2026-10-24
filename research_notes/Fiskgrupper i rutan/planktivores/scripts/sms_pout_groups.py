"""Map SMS 2023 predators of Norway pout to model groups (eaten / mean pout biomass, per year).
Usage: python3 -I sms_pout_groups.py <sms2023 dir> y0 y1"""
import csv, os, sys
from collections import defaultdict
d, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
GRP = {"Cod": "benthivorous_fish", "Haddock": "benthivorous_fish", "G.gurnards": "benthivorous_fish", "A.radiata": "benthivorous_fish",
       "Whiting": "predatory_fish", "Saithe": "predatory_fish", "Hake": "predatory_fish", "Mackerel": "predatory_fish",
       "W.horse.mac": "predatory_fish", "N.horse.mac": "predatory_fish", "Grey.seal": "seals", "H.porpoise": "porpoises"}
hdr = None; bq = defaultdict(float)
for line in open(os.path.join(d, "summary.out")):
    p = line.split()
    if hdr is None: hdr = p; continue
    r = dict(zip(hdr, p))
    if r["Species.n"] == "24": bq[int(r["Year"])] += float(r["N.bar"]) * float(r["west"]) / 4
out = defaultdict(float); ys = range(y0, y1 + 1)
for r in csv.DictReader(open(os.path.join(d, "who_eats_whom_level2.csv"))):
    y = int(r["Year"])
    if r["Prey"].strip() == "Nor. pout" or r["Prey"].strip().startswith("Nor"):
        if y0 <= y <= y1:
            out[GRP.get(r["Predator"].strip(), "seabirds")] += float(r["eatenW"]) / bq[y] / len(ys)
for k, v in sorted(out.items(), key=lambda x: -x[1]): print(f"{k:20s} {v:.3f}")
print("total", round(sum(out.values()), 3))
