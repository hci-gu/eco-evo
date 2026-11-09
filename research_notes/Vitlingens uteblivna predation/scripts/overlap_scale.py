"""Lloyd's interspecific overlap of whiting with cod and saithe at the haul scale and at coarser aggregations,
NS-IBTS GOV, Q1 and Q3 2000-2025 (cached extract; nothing downloaded).

Usage: python3 -I overlap_scale.py <gov dir: hauls.csv, catch.csv.gz> > tables/overlap_scale_stdout.txt
Input: ../../Sejens djupläge/data/gov/ (written by that note's extract_gov.py from cached DATRAS XML).

Question: is the positive haul-scale co-occurrence (box L ~1.6, cod >= 20 cm with whiting 10-28 cm) a large-scale
habitat match hiding a negative small-scale association? The finest scale the survey resolves is one tow
(~30 min at ~4 kn: ~3.7 km x ~20 m wing spread, a line about two model cells long). Aggregation cannot reveal
anything below the tow; it can only show whether L changes between the tow and 10-50 km.

Biomass per haul = sum over length classes of n/h x a L^b (g, cm) as ../../Vitlingens predatorer/scripts/gov_overlap.py.
Units: (i) haul; (ii) the mean of the hauls of one survey (year, quarter) inside a lat-lon cell of 0.1, 0.25, 0.5 deg
(lon cell = 2 x lat cell, ~ square at 58 N); (iii) the cell climatology (all years of a quarter pooled per cell).
Lloyd L = mean(x y) / (mean x mean y) over the units of a quarter (1 = independent, the engine's equal-cell case).
Interval: bootstrap over years (resample whole surveys), 5-95 %.
Also given: L within cells (hauls relative to their cell-survey mean, pooled), i.e. the part of haul-scale L that
is not explained by which cell the hauls are in: L_within = mean over units of sum_h x_h y_h / (n x_c y_c) weighted
by x_c y_c (equals 1 if hauls inside a cell-survey are independent).
"""
import csv, gzip, math, random, sys, collections
from pathlib import Path
d = Path(sys.argv[1])
LW = {126438: ("whiting", 0.0058, 3.10), 126436: ("cod", 0.0077, 3.07), 126441: ("saithe", 0.0089, 3.05)}
REG = {"box": ((57.7, 58.4), (9.9, 11.3)), "skag": ((57.3, 59.2), (7.5, 12.0)), "nns": ((57.0, 61.5), (-2.0, 8.0))}
hauls = {}
with open(d / "hauls.csv") as f:
    for r in csv.DictReader(f):
        if r["Quarter"] not in ("1", "3"):
            continue
        key = (r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip())
        hauls[key] = dict(y=int(r["Year"]), q=int(r["Quarter"]), lat=float(r["ShootLat"]), lon=float(r["ShootLong"]),
                          depth=float(r["Depth"]), kg=collections.Counter())
with gzip.open(d / "catch.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        sp = int(r["aphia"])
        if sp not in LW:
            continue
        h = hauls.get((r["Year"], r["Quarter"], r["Ship"].strip(), r["HaulNo"].strip(), r["ShootLat"].strip()))
        if h is None:
            continue
        name, a, b = LW[sp]
        L, n = float(r["length_cm"]), float(r["n_per_h"])
        if L <= 0 or n <= 0:
            continue
        kg = n * a * L ** b / 1000.0
        if name == "whiting":
            if 10 <= L < 28: h["kg"]["wh10_28"] += kg
            h["kg"]["whiting"] += kg
        elif name == "cod":
            if L >= 20: h["kg"]["cod20"] += kg
        elif name == "saithe":
            if L >= 35: h["kg"]["saithe35"] += kg
H = list(hauls.values())
print(f"GOV hauls Q1/Q3 2000-2025 north of 56 N: {len(H)}")

def lloyd(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    return (sum(a * b for a, b in zip(x, y)) / n) / (mx * my) if mx > 0 and my > 0 else float("nan")

def units(sub, scale, pred, prey, clim=False):
    if scale is None:
        return [h["kg"][pred] for h in sub], [h["kg"][prey] for h in sub]
    g = collections.defaultdict(list)
    for h in sub:
        k = (math.floor(h["lat"] / scale), math.floor(h["lon"] / (2 * scale)))
        g[k if clim else (h["y"],) + k].append(h)
    xs = [sum(h["kg"][pred] for h in v) / len(v) for v in g.values()]
    ys = [sum(h["kg"][prey] for h in v) / len(v) for v in g.values()]
    return xs, ys

def boot(sub, scale, pred, prey, clim, nb=500):
    yrs = sorted({h["y"] for h in sub}); rnd = random.Random(5); v = []
    by = collections.defaultdict(list)
    for h in sub: by[h["y"]].append(h)
    for _ in range(nb):
        s = []
        for i, y in enumerate(rnd.choices(yrs, k=len(yrs))):
            # relabel the year so that repeated years stay separate surveys in the cell-survey units
            s += [dict(h, y=(h["y"], i)) for h in by[y]]
        x, yv = units(s, scale, pred, prey, clim)
        l = lloyd(x, yv)
        if not math.isnan(l): v.append(l)
    v.sort()
    return v[int(.05 * len(v))], v[int(.95 * len(v)) - 1]

def within(sub, scale, pred, prey):
    g = collections.defaultdict(list)
    for h in sub:
        g[(h["y"], math.floor(h["lat"] / scale), math.floor(h["lon"] / (2 * scale)))].append(h)
    num = den = 0.0; multi = 0
    for v in g.values():
        if len(v) < 2: continue
        multi += len(v)
        xc = sum(h["kg"][pred] for h in v) / len(v); yc = sum(h["kg"][prey] for h in v) / len(v)
        num += sum(h["kg"][pred] * h["kg"][prey] for h in v) / len(v); den += xc * yc
    return (num / den if den > 0 else float("nan")), multi

PAIRS = (("cod20", "wh10_28"), ("cod20", "whiting"), ("saithe35", "wh10_28"))
for reg, (LA, LO) in REG.items():
    for q in (1, 3):
        sub = [h for h in H if h["q"] == q and LA[0] <= h["lat"] <= LA[1] and LO[0] <= h["lon"] <= LO[1] and h["y"] >= 2000]
        print(f"\n=== region {reg} {LA} N {LO} E, Q{q}: {len(sub)} hauls, {len({h['y'] for h in sub})} years")
        for pred, prey in PAIRS:
            out = []
            for scale, clim in ((None, False), (0.1, False), (0.25, False), (0.5, False), (0.25, True), (0.5, True)):
                if reg == "box" and scale == 0.5: continue
                x, y = units(sub, scale, pred, prey, clim)
                lo, hi = boot(sub, scale, pred, prey, clim)
                lab = "haul" if scale is None else f"{scale}d{'-clim' if clim else ''}"
                out.append(f"{lab} {lloyd(x, y):4.2f} [{lo:4.2f},{hi:4.2f}] n{len(x)}")
            print(f"  {pred:8s} x {prey:8s}: " + " | ".join(out))
            for scale in (0.1, 0.25, 0.5):
                w, m = within(sub, scale, pred, prey)
                print(f"      within {scale} deg cell-surveys with >= 2 hauls: L_within {w:4.2f} ({m} hauls)")
