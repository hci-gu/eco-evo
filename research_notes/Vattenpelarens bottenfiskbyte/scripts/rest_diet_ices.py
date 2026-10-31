"""Diet by mass of the rest water-column group's members (saithe, hake, spurdog, anglerfish, ling, pollack)
from the public ICES stomach database, mapped to the model's prey groups of the rest group's pairs.

Usage: python3 -I rest_diet_ices.py <ices_stomach_dir> <out_dir> <worms_dir> [<worms_dir2> ...]

Adapted from research_notes/"Vitlingens fiskbyten"/scripts/whiting_diet_ices.py (same WoRMS classification,
same areas, mass pooled over stomachs, no catch weighting). Changes: the predator is each member in turn;
whiting is a checked prey of the rest ("wht"); mackerel/horse mackerel are the mackerel group ("mack");
the rest's own members (Pollachius, Merluccius, Squalus, Lophius, Molva) are "own" (unbooked);
length classes per member: < 40, 40-59, >= 60 cm.
"""
import collections
import csv
import json
import os
import sys

root, out = sys.argv[1], sys.argv[2]
wdirs = sys.argv[3:]
os.makedirs(out, exist_ok=True)
PRED = {"126441": "saithe", "126484": "hake", "105923": "spurdog", "126555": "anglerfish",
        "126461": "ling", "126440": "pollack"}
_lin = {}
missing = collections.Counter()


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


FISH = {"Actinopterygii", "Actinopteri", "Elasmobranchii", "Chondrichthyes", "Teleostei", "Pisces",
        "Osteichthyes", "Gnathostomata", "Vertebrata", "Myxini"}
ZOO = {"Euphausiacea", "Copepoda", "Hyperiidea", "Pasiphaeidae", "Sergestidae", "Oplophoridae",
       "Acanthephyridae", "Chaetognatha", "Appendicularia", "Thaliacea", "Ctenophora", "Tomopteridae",
       "Thecosomata", "Gymnosomata", "Cladocera", "Hyperiidae"}
EPI = {"Pandalidae", "Crangonidae", "Hippolytidae", "Processidae", "Palaemonidae", "Thoridae",
       "Mysida", "Mysidae", "Brachyura", "Anomura", "Paguridae", "Galatheidae", "Munididae",
       "Cephalopoda", "Caridea", "Lophogastrida"}


def classify(aid, lname=None):
    if not aid or aid in ("NA", ""):
        return "U", "no_id"
    L = lineage(aid)
    if L is None:
        missing[aid] += 1
        return "U", "no_lineage"
    s = set(L)
    leaf = L[-1]
    if s & FISH:
        if "Trisopterus esmarkii" in s:
            return "plk", "Norway pout"
        if {"Clupea", "Sprattus"} & s:
            return "plk", "herring" if "Clupea" in s else "sprat"
        if "Clupeidae" in s:
            return "plk", "clupeid unid"
        if "Micromesistius" in s:
            return "plk", "blue whiting"
        if "Merlangius" in s:
            return "wht", "whiting"
        if {"Scomber", "Trachurus", "Scombridae", "Carangidae"} & s:
            return "mack", leaf
        if {"Gadus", "Melanogrammus"} & s:
            return "ben", "cod" if "Gadus" in s else "haddock"
        if {"Pleuronectiformes", "Rajiformes", "Triglidae", "Rajidae"} & s:
            return "ben", "flatfish/ray/gurnard"
        if {"Pollachius", "Merluccius", "Squalus", "Lophius", "Molva", "Etmopterus"} & s:
            return "own", leaf
        if "Ammodytidae" in s:
            return "sand", "sandeel"
        if {"Gadidae", "Gadiformes", "Trisopterus"} & s and leaf in ("Gadidae", "Gadiformes", "Trisopterus"):
            return "gadu", leaf
        if leaf in ("Actinopterygii", "Actinopteri", "Teleostei", "Pisces", "Osteichthyes", "Vertebrata",
                    "Gnathostomata"):
            return "fishu", leaf
        return "sfish", leaf
    if "Nephrops" in s:
        return "neph", "Nephrops"
    if s & ZOO:
        if "Euphausiacea" in s:
            return "zoo", "euphausiid"
        if "Copepoda" in s:
            return "zoo", "copepod"
        if "Hyperiidea" in s or "Hyperiidae" in s:
            return "zoo", "hyperiid"
        if "Pasiphaeidae" in s or "Sergestidae" in s:
            return "zoo", "pelagic shrimp"
        return "zoo", "other zoo"
    if s & EPI:
        if "Pandalidae" in s:
            return "epi", "Pandalus"
        if "Crangonidae" in s:
            return "epi", "Crangon"
        if "Mysida" in s or "Mysidae" in s or "Lophogastrida" in s:
            return "epi", "mysid"
        if "Brachyura" in s:
            return "epi", "crab"
        if "Anomura" in s or "Paguridae" in s or "Galatheidae" in s or "Munididae" in s:
            return "epi", "hermit/Munida"
        if "Cephalopoda" in s:
            return "epi", "cephalopod"
        return "epi", "other shrimp"
    if leaf in ("Crustacea", "Arthropoda", "Malacostraca", "Eucarida", "Multicrustacea",
                "Eumalacostraca", "Decapoda", "Pleocyemata", "Peracarida"):
        return "crustu", leaf
    return "inf", leaf


def area(lat, lon):
    if lat is None or lon is None:
        return "unknown"
    if 55.5 <= lat <= 57.75 and 10.3 <= lon <= 13.0:
        return "Kattegat"
    hl = 8.6 + (lat - 57.1) * (7.05 - 8.6) / (58.0 - 57.1)  # Hanstholm-Lindesnes line
    if lat >= 58.0:
        hl = 7.05
    if 57.0 <= lat <= 59.2 and lon >= hl:
        return "Skagerrak"
    if lat >= 57.5:
        return "NNS"
    if 51.0 <= lat < 57.5 and lon < 8.0:
        return "CSNS"
    return "other"


def lclass(L):
    if L is None:
        return "unk"
    if L < 40:
        return "<40"
    if L < 60:
        return "40-59"
    return ">=60"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


CATS = ["plk", "ben", "wht", "mack", "own", "sfish", "sand", "gadu", "fishu", "zoo", "epi", "neph", "inf", "crustu", "U"]
mass = collections.defaultdict(lambda: collections.Counter())
sub = collections.defaultdict(lambda: collections.Counter())
nst = collections.Counter()
nfed = collections.Counter()
ratios = []  # (area, period, pred L cm, prey label, prey L cm, mass g)
years = sorted(d for d in os.listdir(root) if d.startswith("x"))
for yd in years:
    yr = int(yd[1:])
    period = "1981" if yr == 1981 else ("1991" if yr == 1991 else ("1985-99" if yr < 2000 else "2000+"))
    H = {}
    with open(os.path.join(root, yd, "HaulInformation.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            H[r["tblHaulID"]] = (fnum(r["ShootLat"]), fnum(r["ShootLong"]), r["Month"])
    pred = {}
    with open(os.path.join(root, yd, "PredatorInformation.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            sp = PRED.get(r["AphiaIDPredator"].strip('"'))
            if sp is None:
                continue
            L = fnum(r["Length"])
            if L is not None:
                inc = fnum(r["MeasurementIncrement"]) or 10
                L = L / 10.0 if (inc in (1, 5, 10, 50) and L > 60) else L  # mm -> cm when needed
                L = L / 10.0 if L > 100 else L
            lat, lon, mon = H.get(r["tblHaulID"], (None, None, ""))
            q = (int(mon) - 1) // 3 + 1 if mon and mon.isdigit() else 0
            key = (sp, area(lat, lon), period, lclass(L))
            n = int(fnum(r["Number"]) or 1)
            pred[r["tblPredatorInformationID"]] = (key, L, q)
            for k in (key, key[:3] + ("all",), (key[0], key[1], "all", key[3]), (key[0], key[1], "all", "all")):
                nst[k] += n
                if r["StomachEmpty"] not in ("1", '"1"'):
                    nfed[k] += n
    with open(os.path.join(root, yd, "PreyInformation.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            pid = r["tblPredatorInformationID"]
            if pid not in pred:
                continue
            w = fnum(r["Weight"])
            if w is None or w <= 0:
                continue
            if r["UnitWgt"].strip('"') == "mg":
                w /= 1000.0
            cat, lab = classify(r["AphiaIDPrey"].strip('"'))
            key, L, q = pred[pid]
            for k in (key, key[:3] + ("all",), (key[0], key[1], "all", key[3]), (key[0], key[1], "all", "all")):
                mass[k][cat] += w
                sub[k][lab] += w
            pl = fnum(r["Length"])
            if pl is not None and L and cat in ("plk", "ben", "wht", "mack", "sand", "sfish"):
                if r["UnitLngt"].strip('"') == "mm":
                    pl /= 10.0
                if 0.5 <= pl <= 60:
                    ratios.append((key[0], key[1], key[2], L, lab, pl, w, q))

rows = []
for k in sorted(mass):
    tot = sum(mass[k][c] for c in CATS if c != "U")
    if tot <= 0:
        continue
    row = dict(pred=k[0], area=k[1], period=k[2], length=k[3], stomachs=nst[k], fed=nfed[k], g_ident=round(tot, 1),
               g_U=round(mass[k]["U"], 1))
    for c in CATS[:-1]:
        row[c] = round(100 * mass[k][c] / tot, 1)
    top = ", ".join(f"{lab} {100 * v / (tot + mass[k]['U']):.1f}" for lab, v in sub[k].most_common(8))
    row["top_items_pct"] = top
    rows.append(row)
with open(os.path.join(out, "rest_diet_ices.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
with open(os.path.join(out, "rest_prey_lengths_ices.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["pred", "area", "period", "pred_L_cm", "prey", "prey_L_cm", "mass_g", "quarter"])
    wr.writerows(ratios)

print("Rest water-column members: diet by mass (% of identified prey mass), ICES stomach DB; U = unclassified (g)")
print("cats: plk planktivores | ben benthivores | wht whiting | mack mackerel group | own rest members | sfish small demersal fish"
      " | sand sandeel | gadu gadoid unid | fishu fish unid | zoo | epi | neph | inf | crustu crust. unid")
hdr = f"{'pred':10s} {'area':10s} {'period':8s} {'len':6s} {'stom':>5s} {'fed':>5s} {'g':>7s} " + " ".join(f"{c:>6s}" for c in CATS[:-1]) + f" {'gU':>6s}"
print(hdr)
for r in rows:
    if r["area"] in ("other", "unknown"):
        continue
    print(f"{r['pred']:10s} {r['area']:10s} {r['period']:8s} {r['length']:6s} {r['stomachs']:5d} {r['fed']:5d} {r['g_ident']:7.1f} "
          + " ".join(f"{r[c]:6.1f}" for c in CATS[:-1]) + f" {r['g_U']:6.1f}")
print("\nTop items (% of all prey mass incl. U), area x period, length 'all':")
for r in rows:
    if r["length"] == "all" and r["area"] not in ("other", "unknown"):
        print(f"  {r['pred']:10s} {r['area']:10s} {r['period']:8s} n {r['stomachs']} {r['top_items_pct']}")
print(f"\nPrey AphiaIDs without a cached lineage (records): {len(missing)} ids, {sum(missing.values())} records")
print("  " + ", ".join(f"{a}:{n}" for a, n in missing.most_common(40)))
