"""Fish-prey lengths in the ICES Stomach Content database (public download), per predator.

Usage: python3 -I ices_prey_size.py <ices_stomach_dir> <worms_dir> [<worms_dir> ...]

Reads the extracted x<year>/ folders (HaulInformation, PredatorInformation, PreyInformation; never
executes anything from them). Fish prey = WoRMS lineage contains a fish class (cached cls_<aphia>.json
files, as ../../Fiskgrupper i rutan/benthivores/scripts/benth_diet_pools.py). Rows with GravMethod
DRYWT dropped. Predator Length > 200 read as mm. Prey length: UnitLngt mm or cm. Each row is one prey
record (Count fish of that length); weights: numbers (Count, default 1) and mass (Weight, g).
Reports prey/predator length percentiles (p50/p90/p95/p99/max), by numbers and by mass, for all areas
and for Skagerrak + northern North Sea (areas as benth_common.area_of), and per predator-length band.
Prey groups: P (herring, sprat, Norway pout, blue whiting, other clupeids), Bf (haddock, cod,
flatfish), W (whiting), all fish.
"""
import collections
import csv
import json
import os
import sys

import numpy as np

root, wdirs = sys.argv[1], sys.argv[2:]
PRED = {"126438": "whiting", "126484": "hake", "105923": "spurdog", "126555": "anglerfish",
        "126461": "ling", "126822": "horse mackerel", "126441": "saithe", "127023": "mackerel",
        "126436": "cod", "126437": "haddock", "105913": "velvet belly"}
FISH = {"Actinopterygii", "Actinopteri", "Elasmobranchii", "Chondrichthyes", "Myxini", "Pisces", "Teleostei"}


def fnum(x):
    try:
        return float(str(x).strip('"'))
    except ValueError:
        return float("nan")


def area_of(lat, lon):
    if lat != lat or lon != lon:
        return "na"
    if 55.5 <= lat < 57.75 and 10.3 <= lon <= 13:
        return "Kattegat"
    if lat >= 57.0 and lon <= 13:
        lon_line = 8.6 + (lat - 57.12) * (7.05 - 8.6) / (58.0 - 57.12)
        if lat <= 59.2 and lon >= lon_line:
            return "Skagerrak"
    if lat >= 57.5 and -4 <= lon <= 13:
        return "NNS"
    if 51 <= lat < 57.5 and -4 <= lon <= 9:
        return "CSNS"
    return "other"


cache = {}


def lineage(aid):
    if aid in cache:
        return cache[aid]
    out = None
    for w in wdirs:
        p = os.path.join(w, f"cls_{aid}.json")
        if os.path.exists(p):
            try:
                d = json.load(open(p))
            except Exception:
                break
            out = []
            while d:
                out.append(d.get("scientificname", ""))
                d = d.get("child")
            break
    cache[aid] = out
    return out


def group(L):
    s = set(L)
    if s & {"Clupea", "Sprattus", "Micromesistius", "Clupeidae"} or any(n.startswith("Trisopterus esmark") for n in s):
        return "P"
    if s & {"Melanogrammus", "Gadus", "Pleuronectiformes", "Pleuronectidae", "Hippoglossoides"}:
        return "Bf"
    if "Merlangius" in s:
        return "W"
    return "other"


recs = collections.defaultdict(list)   # predator -> (area, Lpred, Lprey, n, g, group, year, leaf)
nolin = collections.Counter()
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p):
        continue
    hauls = {r["tblHaulID"]: r for r in csv.DictReader(open(os.path.join(p, "HaulInformation.csv"),
                                                            newline="", encoding="utf-8", errors="replace"))}
    preds = {}
    for r in csv.DictReader(open(os.path.join(p, "PredatorInformation.csv"), newline="", encoding="utf-8",
                                 errors="replace")):
        sp = PRED.get(r["AphiaIDPredator"].strip('"'))
        if sp:
            preds[r["tblPredatorInformationID"]] = (sp, r)
    for r in csv.DictReader(open(os.path.join(p, "PreyInformation.csv"), newline="", encoding="utf-8",
                                 errors="replace")):
        pid = r["tblPredatorInformationID"]
        if pid not in preds or r["GravMethod"].strip('"').upper() == "DRYWT":
            continue
        sp, pr = preds[pid]
        aid = r["AphiaIDPrey"].strip('"')
        if not aid:
            continue
        L = lineage(aid)
        if L is None:
            nolin[aid] += 1
            continue
        if not (set(L) & FISH):
            continue
        lq = fnum(r["Length"])
        u = r["UnitLngt"].strip('"').lower()
        if lq != lq or u not in ("mm", "cm"):
            continue
        lq = lq / 10 if u == "mm" else lq
        lp = fnum(pr["Length"])
        if lp != lp:
            continue
        lp = lp / 10 if lp > 200 else lp
        if lq <= 0 or lp <= 0:
            continue
        h = hauls.get(pr["tblHaulID"], {})
        ar = area_of(fnum(h.get("ShootLat")), fnum(h.get("ShootLong")))
        n = fnum(r["Count"])
        g = fnum(r["Weight"])
        if r["UnitWgt"].strip('"').lower() == "mg":
            g /= 1000
        recs[sp].append((ar, lp, lq, n if n == n and n > 0 else 1.0, g if g == g and g > 0 else np.nan,
                         group(L), int(pr["Year"]), L[-1]))
print(f"prey AphiaIDs without a cached lineage (skipped): {len(nolin)} ids, {sum(nolin.values())} rows")


def wq(v, w, ps=(0.5, 0.9, 0.95, 0.99)):
    v, w = np.asarray(v, float), np.asarray(w, float)
    ok = np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if len(v) == 0:
        return None
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(v[o][min(np.searchsorted(cw, p), len(v) - 1)]) for p in ps]


def line(lab, rr):
    if not rr:
        return f"    {lab:34s} n 0"
    lp = np.array([r[1] for r in rr]); lq = np.array([r[2] for r in rr])
    n = np.array([r[3] for r in rr]); g = np.array([r[4] for r in rr])
    r = lq / lp
    a, b = wq(r, n), wq(r, g)
    s = (f"    {lab:34s} rec {len(rr):4d} fish {int(n.sum()):5d} | by number p50 {a[0]:.2f} p90 {a[1]:.2f} "
         f"p95 {a[2]:.2f} p99 {a[3]:.2f} max {r.max():.2f}")
    if b:
        s += f" | by mass p50 {b[0]:.2f} p90 {b[1]:.2f} p95 {b[2]:.2f} p99 {b[3]:.2f}"
    qa = wq(lq, n)
    s += f" | prey cm p50 {qa[0]:.1f} p95 {qa[2]:.1f} max {lq.max():.1f} | pred cm p50 {np.median(lp):.0f}"
    return s


BANDS = {"whiting": [(10, 20), (20, 30), (30, 60)], "hake": [(15, 40), (40, 60), (60, 130)],
         "cod": [(15, 30), (30, 45), (45, 65), (65, 130)], "spurdog": [(30, 70), (70, 130)],
         "anglerfish": [(20, 50), (50, 150)], "horse mackerel": [(10, 30), (30, 50)],
         "mackerel": [(10, 30), (30, 50)], "saithe": [(20, 60), (60, 130)]}
for sp in PRED.values():
    rr = recs.get(sp, [])
    if not rr:
        print(f"\n{sp}: no fish prey with lengths")
        continue
    print(f"\n{sp}: years {sorted(collections.Counter(r[6] for r in rr).items())}")
    print(f"  areas {dict(collections.Counter(r[0] for r in rr))}")
    for arlab, ars in (("all areas", None), ("Skagerrak+NNS", {"Skagerrak", "NNS"})):
        sub = [r for r in rr if ars is None or r[0] in ars]
        print(f"  {arlab}")
        for gname in ("all", "P", "Bf", "W"):
            print(line(f"prey {gname}", [r for r in sub if gname == "all" or r[5] == gname]))
        for lo, hi in BANDS.get(sp, []):
            for gname in ("all", "P", "Bf"):
                print(line(f"prey {gname}, pred {lo}-{hi} cm",
                           [r for r in sub if lo <= r[1] < hi and (gname == "all" or r[5] == gname)]))
    if len(rr) <= 80:
        print("  items (area, pred cm, prey cm, n, taxon):")
        for r in sorted(rr, key=lambda r: r[1]):
            print(f"    {r[0]:9s} {r[6]} {r[1]:5.1f} {r[2]:5.1f} {r[3]:3.0f} {r[7]}")
