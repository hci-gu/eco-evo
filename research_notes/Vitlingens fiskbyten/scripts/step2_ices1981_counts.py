"""Whiting stomachs of the 1981 Year of the Stomach nearest the box (ICES stomach DB, counts only).

Usage: python3 -I step2_ices1981_counts.py <ices_stomach_dir> <worms_dir> [<worms_dir2> ...]

The 1981 upload has prey COUNTS but no weights. The in-area 1981 hauls are on the eastern North Sea
edge of the Skagerrak (ICES rectangles 42F6-43F8, 56.75-57.25 N, 6.5-8.5 E; Jutland Bank / Hanstholm,
not the deep Skagerrak), quarters 1 and 4. Counts are converted to mass with mean wet weight per item:
  [F] Rindorf 2003 (MEPS 249:265) Table 4 for whiting 20-35 cm: herring 8.26 g, sprat 9.27 g,
      krill 0.12 g, shrimp 0.80 g, crab 0.29 g;
  [D] the 1991 in-area ICES records (weight / count) for categories Rindorf does not give;
  [J] Norway pout 5 g (6-8 cm pout ~3-6 g), sandeel 3 g, other fish 3 g, other invertebrates 0.05 g.
A range is printed with fish weights x0.5 and x2 (J), since whiting eat smaller clupeids in Q1
(8.5 cm herring ~4-5 g) than Rindorf's summer sample.
Predator rows carry a Number (pooled fish per length class, Length in cm); prey counts are per row.
"""
import collections
import csv
import json
import os
import sys

root = sys.argv[1]
wdirs = sys.argv[2:]
WHITING = "126438"
_lin = {}


def lineage(aid):
    if aid in _lin:
        return _lin[aid]
    res = None
    for w in wdirs:
        p = os.path.join(w, f"cls_{aid}.json")
        if os.path.exists(p):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            names = []
            while d:
                names.append(d.get("scientificname", ""))
                d = d.get("child")
            res = names
            break
    _lin[aid] = res
    return res


def item(aid):
    L = lineage(aid) if aid else None
    if not L:
        return "unidentified"
    s = set(L)
    if "Clupea" in s:
        return "herring"
    if "Sprattus" in s:
        return "sprat"
    if "Clupeidae" in s:
        return "clupeid unid"
    if "Trisopterus esmarkii" in s:
        return "Norway pout"
    if "Ammodytidae" in s:
        return "sandeel"
    if "Euphausiacea" in s:
        return "krill"
    if {"Pandalidae", "Crangonidae", "Caridea"} & s:
        return "shrimp"
    if "Brachyura" in s or "Anomura" in s:
        return "crab"
    if "Mysida" in s:
        return "mysid"
    if {"Actinopterygii", "Actinopteri", "Teleostei", "Pisces", "Gnathostomata"} & s:
        return "other fish"
    return "other invertebrate"


W_ITEM = {"herring": 8.26, "sprat": 9.27, "clupeid unid": 8.5, "Norway pout": 5.0, "sandeel": 3.0,
          "other fish": 3.0, "krill": 0.12, "shrimp": 0.80, "crab": 0.29, "mysid": 0.05,
          "other invertebrate": 0.05, "unidentified": 0.05}
FISH = {"herring", "sprat", "clupeid unid", "Norway pout", "sandeel", "other fish"}
PLK = {"herring", "sprat", "clupeid unid", "Norway pout"}

y = "x1981"
H = {r["tblHaulID"]: r for r in csv.DictReader(open(os.path.join(root, y, "HaulInformation.csv"), encoding="latin-1"))}
P = {}
for r in csv.DictReader(open(os.path.join(root, y, "PredatorInformation.csv"), encoding="latin-1")):
    if r["AphiaIDPredator"] != WHITING:
        continue
    h = H.get(r["tblHaulID"])
    try:
        lat, lon = float(h["ShootLat"]), float(h["ShootLong"])
    except Exception:
        continue
    if 56.5 <= lat <= 59.5 and 6.0 <= lon <= 13.0:
        q = (int(h["Month"]) - 1) // 3 + 1
        P[r["tblPredatorInformationID"]] = (q, int(r["Number"] or 1), float(r["Length"] or 0),
                                             h["ICESrectangle"])
print("1981 whiting predator rows near the box:", len(P))
fish_n = collections.Counter()
for q, n, L, rect in P.values():
    fish_n[(q, rect)] += n
print("fish per quarter x rectangle:", dict(sorted(fish_n.items())))
cnt = collections.defaultdict(collections.Counter)
other = collections.defaultdict(collections.Counter)
for r in csv.DictReader(open(os.path.join(root, y, "PreyInformation.csv"), encoding="latin-1")):
    pid = r["tblPredatorInformationID"]
    if pid not in P:
        continue
    q, n, L, rect = P[pid]
    c = float(r["Count"] or 0)
    it = item(r["AphiaIDPrey"])
    cnt[q][it] += c
    if it == "other invertebrate":
        L = lineage(r["AphiaIDPrey"]) or ["?"]
        other[q][L[-1]] += c
for q in sorted(cnt):
    nf = sum(n for qq, n, L, rect in P.values() if qq == q)
    lens = sorted(L for qq, n, L, rect in P.values() if qq == q)
    print(f"\nQ{q}: {nf} whiting, length classes {lens[0]:.0f}-{lens[-1]:.0f} cm; prey counts:")
    for k, v in cnt[q].most_common():
        print(f"   {k:20s} {v:7.0f}")
    print("   'other invertebrate' leaves: " + ", ".join(f"{k} {v:.0f}" for k, v in other[q].most_common(6)))
    for fac, lab in ((1.0, "central"), (0.5, "fish weights x0.5"), (2.0, "fish weights x2")):
        m = {k: v * W_ITEM[k] * (fac if k in FISH else 1.0) for k, v in cnt[q].items()}
        tot = sum(m.values())
        plk = sum(v for k, v in m.items() if k in PLK)
        fish = sum(v for k, v in m.items() if k in FISH)
        mix = ", ".join(f"{k} {m[k] / tot:.2f}" for k in sorted(m, key=lambda k: -m[k])[:6])
        print(f"  mass shares ({lab}): planktivores {plk / tot:.2f}, all fish {fish / tot:.2f}; top: {mix}")
