"""Cod (16) and haddock (18) diet by mass from the SMS 2023 key-run stomach file (North Sea, 1981, 1985-87, 1991; the ICES
Year-of-the-Stomach data as processed by WGSAM), by predator length class and quarter; and the
prey/predator length ratios and prey lengths of the planktivore-pool prey.

Usage: python3 -I sms_cod_haddock_diet.py <summary_stom.out> <predator no: 16 cod | 18 haddock>
Adapted from research_notes/"Vitlingens fiskbyten"/scripts/sms_whiting_diet.py.

stomcon = share of the stomach content mass of one prey (species x length class) in one sample (year,
quarter, area, predator length class); Prey.no 0 = "other food" (every prey SMS does not model:
crustaceans, gobies, dragonets, cephalopods, polychaetes, ...). Sample weight = N.haul (as the earlier
SMS scripts in research_notes/"Arsungarnas andel" (A-ring)). Exact duplicate rows are dropped as there.
Model groups: plk = herring 21, Norway pout 24, sprat 25; ben = cod 16, haddock 18, plaice 26, sole 27;
wht = whiting 17 (the benthivores -> whiting pair); sand = sandeels 22, 23 (no model pool); other = Prey.no 0.
"""
import collections
import sys

import numpy as np

rows = [l.split() for l in open(sys.argv[1])]
PREDNO = int(sys.argv[2])
PNAME = {16: "cod", 18: "haddock"}[PREDNO]
hdr = rows[0]
ix = {h: i for i, h in enumerate(hdr)}
seen, R = set(), []
for x in rows[1:]:
    if int(x[ix["Predator.no"]]) != PREDNO:
        continue
    k = (x[0], x[1], x[2], x[3], x[7], x[15], x[16], x[24], x[26])
    if k in seen:
        continue
    seen.add(k)
    R.append(x)
GRP = {21: "plk", 24: "plk", 25: "plk", 16: "ben", 18: "ben", 26: "ben", 27: "ben", 17: "wht", 22: "sand",
       23: "sand", 0: "other"}
SPN = {21: "herring", 24: "Norway pout", 25: "sprat", 16: "cod", 18: "haddock", 26: "plaice", 27: "sole",
       17: "whiting", 22: "N.sandeel", 23: "S.sandeel", 0: "other food"}
# samples
samp = collections.defaultdict(lambda: collections.Counter())
sp = collections.defaultdict(lambda: collections.Counter())
nh = {}
plen = {}
lens = []  # (pred class, pred L cm, quarter, prey species, prey L cm, ratio, weight)
for x in R:
    key = (x[ix["Year"]], x[ix["Quarter.no"]], x[ix["Predator.length.class"]])
    prey = int(x[ix["Prey.no"]])
    sc = float(x[ix["stomcon"]])
    nh[key] = float(x[ix["N.haul"]])
    plen[key] = float(x[ix["Predator.length.mean"]]) / 10.0
    samp[key][GRP.get(prey, "other")] += sc
    sp[key][SPN.get(prey, str(prey))] += sc
    if prey > 0:
        pl = float(x[ix["Prey.length.mean"]]) / 10.0
        lens.append((key[2], plen[key], int(key[1]), SPN[prey], pl, pl / plen[key], sc * nh[key]))


BANDS = ("<30", "30-39", "40-49", "50-69", ">=70")


def band(L):
    return "<30" if L < 30 else ("30-39" if L < 40 else ("40-49" if L < 50 else ("50-69" if L < 70 else ">=70")))


print(f"SMS {PNAME} stomach samples: {len(samp)} (year x quarter x length class), rows {len(R)}")
G = ["plk", "ben", "wht", "sand", "other"]
for qsel in ("all", "1", "2", "3", "4"):
    acc = collections.defaultdict(lambda: collections.Counter())
    accs = collections.defaultdict(lambda: collections.Counter())
    W = collections.Counter()
    for k, c in samp.items():
        if qsel != "all" and k[1] != qsel:
            continue
        tot = sum(c.values())
        if tot <= 0:
            continue
        b = band(plen[k])
        for g in G:
            acc[b][g] += nh[k] * c[g] / tot
            acc["all"][g] += nh[k] * c[g] / tot
        for s, v in sp[k].items():
            accs[b][s] += nh[k] * v / tot
            accs["all"][s] += nh[k] * v / tot
        W[b] += nh[k]
        W["all"] += nh[k]
    print(f"\nQuarter {qsel}: diet share by mass (N.haul-weighted mean of sample compositions)")
    print(f"  {'band':6s} {'N.haul':>7s} " + " ".join(f"{g:>6s}" for g in G) + "   planktivores by species")
    for b in BANDS + ("all",):
        if W[b] <= 0:
            continue
        pk = ", ".join(f"{s} {accs[b][s] / W[b]:.3f}" for s in ("herring", "sprat", "Norway pout"))
        print(f"  {b:6s} {W[b]:7.0f} " + " ".join(f"{acc[b][g] / W[b]:6.3f}" for g in G) + "   " + pk)

print(f"\nPlanktivore-pool prey eaten by {PNAME}: prey length (cm) and prey/predator length ratio, mass-weighted")
print("(weight = stomcon x N.haul); percentiles p05 / p50 / p95 / p99")


def wq(v, w, ps=(0.05, 0.5, 0.95, 0.99)):
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(v[o][min(np.searchsorted(cw, p), len(v) - 1)]) for p in ps]


for prey in ("herring", "sprat", "Norway pout", "all P"):
    for b in BANDS + ("all",):
        sel = [r for r in lens if (r[3] == prey or (prey == "all P" and r[3] in ("herring", "sprat", "Norway pout")))
               and (b == "all" or band(r[1]) == b)]
        if len(sel) < 3:
            continue
        L = wq([r[4] for r in sel], [r[6] for r in sel])
        Q = wq([r[5] for r in sel], [r[6] for r in sel])
        print(f"  {prey:12s} {b:6s} n {len(sel):4d}  prey L {L[0]:5.1f} {L[1]:5.1f} {L[2]:5.1f} {L[3]:5.1f}"
              f"   ratio {Q[0]:.2f} {Q[1]:.2f} {Q[2]:.2f} {Q[3]:.2f}")
print(f"\nPrey length of herring eaten by quarter (mass-weighted p05/p50/p95), {PNAME} >= 20 cm:")
for q in (1, 2, 3, 4):
    sel = [r for r in lens if r[3] == "herring" and r[2] == q and r[1] >= 20]
    if len(sel) >= 3:
        L = wq([r[4] for r in sel], [r[6] for r in sel], (0.05, 0.5, 0.95))
        print(f"  Q{q}: n {len(sel)}  {L[0]:.1f} / {L[1]:.1f} / {L[2]:.1f} cm")
