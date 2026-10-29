"""Prey-size windows from the SMS 2023 key run stomach file (North Sea, 1981-1991 + a few later years).

Usage: python3 -I sms_prey_size.py <summary_stom.out>

Same weighting as ../../Makrillens bytesstorlek/scripts/mackerel_planktivore_gape.py: fish prey only
(Prey.no > 0), weight = stomcon (stomach content share) x N.haul, step percentile of the weighted
distribution. Duplicate rows (same year, quarter, area, predator, predator mean length, prey, prey
length class, hauls, stomcon) are dropped first (1986-87 saithe duplicates, evidence_saithe.md).
Fish predators: prey/predator length (Prey.length.mean / Predator.length.mean), pooled and per
predator-length band. Mammals (13 grey seal, 14 harbour porpoise; predator length is a dummy):
prey length percentiles in cm. Prey groups: P = herring 21, Norway pout 24, sprat 25 (planktivore
pool); Bf = cod 16, haddock 18, plaice 26, sole 27 (benthivore pool); W = whiting 17; all = every fish.
"""
import sys

import numpy as np

rows = [l.split() for l in open(sys.argv[1])][1:]
seen, R = set(), []
for x in rows:
    # 1986-87 saithe rows repeat under two predator length classes with the same mean length
    k = (x[0], x[1], x[2], x[3], x[7], x[15], x[16], x[24], x[26])
    if k in seen:
        continue
    seen.add(k)
    R.append(x)
print(f"rows {len(rows)}, after dropping exact duplicates {len(R)}")
NAMES = {9: "A.radiata", 10: "gurnards", 11: "W.horse mackerel", 12: "N.horse mackerel", 13: "grey seal",
         14: "harbour porpoise", 15: "hake", 16: "cod", 17: "whiting", 18: "haddock", 19: "saithe", 20: "mackerel"}
PREYN = {16: "cod", 17: "whiting", 18: "haddock", 19: "saithe", 21: "herring", 22: "N.sandeel", 23: "S.sandeel",
         24: "Norway pout", 25: "sprat", 26: "plaice", 27: "sole"}
GROUPS = {"all": None, "P": {21, 24, 25}, "Bf": {16, 18, 26, 27}, "W": {17}}


def wq(v, w, ps=(0.5, 0.9, 0.95, 0.99)):
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(v[o][min(np.searchsorted(cw, p), len(v) - 1)]) for p in ps]


def recs(preds, group, lo=0, hi=1e9):
    out = []
    for x in R:
        p, q = int(x[3]), int(x[15])
        if p not in preds or q <= 0 or float(x[26]) <= 0:
            continue
        if GROUPS[group] is not None and q not in GROUPS[group]:
            continue
        Lp = float(x[7])
        if not (lo <= Lp < hi):
            continue
        out.append((Lp, float(x[17]), float(x[26]) * float(x[24]), q, int(x[0]), int(x[1])))
    return out


def line(lab, rr, mammal=False):
    if len(rr) == 0:
        return f"  {lab:28s} n 0"
    Lp, Lq, w = (np.array([r[i] for r in rr]) for i in range(3))
    if mammal:
        q = wq(Lq / 10, w)
        return (f"  {lab:28s} n {len(rr):3d}  prey cm p50 {q[0]:5.1f} p90 {q[1]:5.1f} p95 {q[2]:5.1f} "
                f"p99 {q[3]:5.1f} max {Lq.max() / 10:5.1f}")
    r = Lq / Lp
    q = wq(r, w)
    qa = wq(Lq / 10, w)
    return (f"  {lab:28s} n {len(rr):3d}  ratio p50 {q[0]:.2f} p90 {q[1]:.2f} p95 {q[2]:.2f} p99 {q[3]:.2f} "
            f"max {r.max():.2f} | prey cm p50 {qa[0]:4.1f} p95 {qa[2]:4.1f} max {Lq.max() / 10:4.1f} | "
            f"pred cm mean(w) {np.sum(Lp * w) / w.sum() / 10:5.1f}")


BANDS = {"saithe": [(300, 450), (450, 650), (650, 1300)], "cod": [(150, 300), (300, 450), (450, 650), (650, 1200)],
         "whiting": [(100, 200), (200, 300), (300, 500)], "hake": [(200, 400), (400, 1100)],
         "haddock": [(150, 300), (300, 450), (450, 650)], "gurnards": [(100, 300), (300, 500)],
         "A.radiata": [(200, 400), (400, 600)]}
for p, name in NAMES.items():
    preds = {11, 12} if p in (11, 12) else {p}
    if p == 12:
        continue
    if p == 11:
        name = "horse mackerel (W+N)"
    mammal = p in (13, 14)
    print(f"\n{name} (SMS {sorted(preds)})")
    yrs = sorted({r[4] for r in recs(preds, 'all')})
    print(f"  years {yrs}")
    for g in GROUPS:
        print(line(f"prey {g}, all sizes", recs(preds, g), mammal))
    for lo, hi in BANDS.get(name, []):
        for g in ("all", "P", "Bf"):
            print(line(f"prey {g}, pred {lo / 10:.0f}-{hi / 10:.0f} cm", recs(preds, g, lo, hi)))
    if mammal:
        for q in sorted({r[3] for r in recs(preds, 'all')}):
            rr = [r for r in recs(preds, 'all') if r[3] == q]
            print(line(f"prey {PREYN.get(q, q)}", rr, True))
