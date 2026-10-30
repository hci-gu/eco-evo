"""Quarterly biomass of mackerel and horse mackerel inside the SMS North Sea model area (ICES IV)
from the WGSAM SMS key-run output files (github.com/ices-eg/wg_WGSAM). Reads text tables only.

- other_predators.out (2014 and 2023 key runs): columns Species.n Year Quarter Age N west ...;
  biomass (t) = N (thousands) * west (kg). These N are the stock numbers PRESENT in the North Sea
  (the input already carries the residence fraction).
- summary.out (2023 key run): Mackerel is a modelled species (Species.n 20) with whole-stock N and
  the column prop.in (proportion in the model area); in-area biomass = BIO * prop.in.

Usage: python3 -I lit_sms_mackerel_quarters.py <wgsam_sms dir> [years ...]
"""
import os, sys
from collections import defaultdict

d = sys.argv[1]
years = [int(y) for y in sys.argv[2:]] or [1991, 2000, 2010, 2015, 2020, 2022]
AREA = 570000.0  # km2, the EwE North Sea area, used only to express t/km2 for comparison

def other_pred(path, names):
    bio = defaultdict(float)
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.split()
            if len(p) < 6:
                continue
            sp, yr, q = int(p[0]), int(p[1]), int(p[2])
            n, w = float(p[4]), float(p[5])
            if sp in names and n > 0 and w > 0:
                bio[(names[sp], yr, q)] += n * w
    return bio

def summary(path, sp_target):
    tot, inside = defaultdict(float), defaultdict(float)
    with open(path) as f:
        hdr = f.readline().split()
        ix = {h: i for i, h in enumerate(hdr)}
        for line in f:
            p = line.split()
            if len(p) < len(hdr) or int(p[ix["Species.n"]]) != sp_target:
                continue
            yr, q = int(p[ix["Year"]]), int(p[ix["Quarter"]])
            b, pin = float(p[ix["BIO"]]), float(p[ix["prop.in"]])
            tot[(yr, q)] += b
            inside[(yr, q)] += b * pin
    return tot, inside

def show(title, bio, labels):
    print(f"== {title}  (tonnes in ICES IV; annual mean of quarters; t/km2 at {AREA:.0f} km2)")
    for lab in labels:
        for yr in years:
            qs = [bio.get((lab, yr, q), 0.0) for q in (1, 2, 3, 4)]
            if sum(qs) == 0:
                continue
            m = sum(qs) / 4
            print(f"  {lab:12s} {yr}: Q1 {qs[0]:9.0f} Q2 {qs[1]:9.0f} Q3 {qs[2]:9.0f} Q4 {qs[3]:9.0f} | mean {m:9.0f} = {m/AREA:.3f} t/km2 | max {max(qs)/AREA:.3f}")

b14 = other_pred(os.path.join(d, "NorthSeaKeyRun_2014_other_predators.out"),
                 {11: "W_mackerel", 12: "N_mackerel", 13: "W_horse_mac", 14: "N_horse_mac"})
show("SMS key run 2014 (WGSAM 2014/2015), other predators", b14,
     ["W_mackerel", "N_mackerel", "W_horse_mac", "N_horse_mac"])
b23 = other_pred(os.path.join(d, "NorthSeaKeyRun_2023_other_predators.out"),
                 {11: "W_horse_mac", 12: "N_horse_mac"})
show("SMS key run 2023, other predators", b23, ["W_horse_mac", "N_horse_mac"])
tot, inside = summary(os.path.join(d, "NorthSeaKeyRun_2023_summary.out"), 20)
print("== SMS key run 2023, Mackerel (species 20): whole-stock BIO and BIO*prop.in (in ICES IV)")
for yr in years:
    qs = [inside.get((yr, q), 0.0) for q in (1, 2, 3, 4)]
    ts = [tot.get((yr, q), 0.0) for q in (1, 2, 3, 4)]
    if sum(ts) == 0:
        continue
    m = sum(qs) / 4
    print(f"  {yr}: in-area Q1 {qs[0]:9.0f} Q2 {qs[1]:9.0f} Q3 {qs[2]:9.0f} Q4 {qs[3]:9.0f} | mean {m:9.0f} = {m/AREA:.3f} t/km2"
          f" | share of stock by Q: " + " ".join(f"{(a/b if b else 0):.3f}" for a, b in zip(qs, ts)))
