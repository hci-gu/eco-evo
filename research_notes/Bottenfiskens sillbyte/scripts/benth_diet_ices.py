"""Cod (Gadus morhua) and haddock (Melanogrammus aeglefinus) diet by mass from the public ICES stomach
database, mapped to the model's CURRENT prey groups (sections 209-230) of the benthivore pairs, with the
planktivore pool split by species (Norway pout, herring, sprat, clupeid unid., blue whiting), by area,
period, quarter and predator length class; plus fish-prey / predator length ratios.
Adapted from research_notes/"Vitlingens fiskbyten"/scripts/whiting_diet_ices.py (same classification,
area rule and mass convention). Fix: the misspelt WoRMS name "Trisopterus esmarki" (separate AphiaID in
the 2023 uploads) is Norway pout, as in "Fiskgrupper i rutan"/benthivores evidence C section 2.

Usage: python3 -I benth_diet_ices.py <ices_stomach_dir> <out_dir> <worms_dir> [<worms_dir2> ...]

Reads <ices_stomach_dir>/x<year>/{HaulInformation,PredatorInformation,PreyInformation}.csv as text
(never executed). Prey AphiaIDs are classified from cached WoRMS lineages (cls_<id>.json). Missing
lineages are listed (they go to "U"). Mass = Weight converted to g (UnitWgt mg -> /1000), pooled over
stomachs (no catch weighting; the convention of the earlier notes).

Model prey groups (whiting's pairs):
  plk   planktivorous_fish: Clupea, Sprattus, Clupeidae unid., Trisopterus esmarkii, Micromesistius
  ben   benthivorous_fish: Gadus, Melanogrammus, Pleuronectiformes, rays, gurnards (Triglidae)
  own   whiting (Merlangius): the benthivores -> whiting pair
  wc    other predatory_fish members / mackerel (Pollachius, Merluccius, Scomber, Trachurus, ...)
  sfish small demersal fish (gobies, dragonets, poor cod/bib, rocklings, sea snails, butterfish, ...):
        in the E1 epifauna member "cephalopods and small demersal fish" (0.23 t/km2)
  sand  sandeels (Ammodytidae): no model pool
  gadu  gadoids not identified to species (Gadidae, Trisopterus sp.)
  fishu fish not identified
  zoo   zooplankton: euphausiids, copepods, hyperiids, Pasiphaea/pelagic shrimps, chaetognaths, ...
  epi   mobile_epifauna (exposed): Pandalidae, Crangonidae and other Caridea, Mysida, Brachyura,
        Anomura (hermits, Munida), Cephalopoda
  neph  Nephrops norvegicus
  inf   infauna and other benthos: Polychaeta, Bivalvia, Echinodermata, Amphipoda (non-hyperiid),
        Cumacea, Isopoda, other invertebrates
  crustu crustaceans not identified below Crustacea/Malacostraca/Decapoda
  U     no AphiaID or no lineage
Areas: Skagerrak = N of 57 N, E of the Hanstholm-Lindesnes line (as predator_diet_pools.py), S of
59.2 N, outside Kattegat; Kattegat 55.5-57.75 N, 10.3-13 E; NNS >= 57.5 N W of the Skagerrak;
CSNS 51-57.5 N W of 8 E. Depth is not recorded in the 1981/1991 uploads; area is the depth proxy.
"""
import collections
import csv
import json
import os
import sys

root, out = sys.argv[1], sys.argv[2]
wdirs = sys.argv[3:]
os.makedirs(out, exist_ok=True)
PRED = {"126436": "cod", "126437": "haddock"}
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
        if any(n.startswith("Trisopterus esmark") for n in s):
            return "plk", "Norway pout"
        if {"Clupea", "Sprattus"} & s:
            return "plk", "herring" if "Clupea" in s else "sprat"
        if "Clupeidae" in s:
            return "plk", "clupeid unid"
        if "Micromesistius" in s:
            return "plk", "blue whiting"
        if "Merlangius" in s:
            return "own", "whiting"
        if {"Gadus", "Melanogrammus"} & s:
            return "ben", "cod" if "Gadus" in s else "haddock"
        if {"Pleuronectiformes", "Rajiformes", "Triglidae", "Rajidae"} & s:
            return "ben", "flatfish/ray/gurnard"
        if {"Pollachius", "Merluccius", "Scomber", "Trachurus", "Squalus", "Lophius", "Molva",
            "Scombridae", "Carangidae"} & s:
            return "wc", leaf
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
    """Areas as research_notes/"Fiskgrupper i rutan"/benthivores/scripts/benth_common.py area_of. NOTE: the
    whiting script's area() had no eastern longitude bound, so Baltic hauls (57-59 N, 16-21 E) fell into
    'Skagerrak'; here lon <= 13 E is required."""
    if lat is None or lon is None:
        return "unknown"
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


def areas(lat, lon, depth):
    a = area(lat, lon)
    out = [a]
    if a == "Skagerrak" and depth is not None and depth >= 80:
        out.append("Skag>=80m")
    if a == "NNS" and depth is not None and depth >= 100:
        out.append("NNS>=100m")
    return out


def lclass(L):
    if L is None:
        return "unk"
    if L < 30:
        return "<30"
    if L < 40:
        return "30-39"
    if L < 50:
        return "40-49"
    if L < 70:
        return "50-69"
    return ">=70"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


PLK_SP = ["Norway pout", "herring", "sprat", "clupeid unid", "blue whiting"]
CATS = ["plk", "ben", "own", "wc", "sfish", "sand", "gadu", "fishu", "zoo", "epi", "neph", "inf", "crustu", "U"]
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
            H[r["tblHaulID"]] = (fnum(r["ShootLat"]), fnum(r["ShootLong"]), r["Month"], fnum(r.get("Depth")))
    pred = {}
    with open(os.path.join(root, yd, "PredatorInformation.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            sp = PRED.get(r["AphiaIDPredator"].strip('"'))
            if sp is None:
                continue
            L = fnum(r["Length"])
            if L is not None:
                L = L / 10.0 if L > 150 else L  # cod/haddock lengths are cm; > 150 can only be mm
            lat, lon, mon, dep = H.get(r["tblHaulID"], (None, None, "", None))
            q = (int(mon) - 1) // 3 + 1 if mon and mon.isdigit() else 0
            n = int(fnum(r["Number"]) or 1)
            keys = []
            for ar in areas(lat, lon, dep):
                key = (sp, ar, period, lclass(L))
                keys += [key, (sp, ar, period, "all"), (sp, ar, "all", key[3]), (sp, ar, "all", "all"),
                         (sp, ar, period, "Q%d" % q)]
            pred[r["tblPredatorInformationID"]] = (keys, L, q, areas(lat, lon, dep)[0], period)
            for k in keys:
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
            keys, L, q, ar0, per0 = pred[pid]
            for k in keys:
                mass[k][cat] += w
                sub[k][lab] += w
            pl = fnum(r["Length"])
            if pl is not None and L and cat in ("plk", "ben", "own", "sand", "sfish"):
                if r["UnitLngt"].strip('"') == "mm":
                    pl /= 10.0
                if 0.5 <= pl <= 60:
                    ratios.append((keys[0][0], ar0, per0, L, lab, pl, w, q))

rows = []
for k in sorted(mass):
    tot = sum(mass[k][c] for c in CATS if c != "U")
    if tot <= 0:
        continue
    row = dict(pred=k[0], area=k[1], period=k[2], length=k[3], stomachs=nst[k], fed=nfed[k],
               g_ident=round(tot, 1), g_U=round(mass[k]["U"], 1))
    for c in CATS[:-1]:
        row[c] = round(100 * mass[k][c] / tot, 1)
    for lab in PLK_SP:
        row["plk_" + lab.replace(" ", "_")] = round(100 * sub[k][lab] / tot, 2)
    top = ", ".join(f"{lab} {100 * v / (tot + mass[k]['U']):.1f}" for lab, v in sub[k].most_common(8))
    row["top_items_pct"] = top
    rows.append(row)
with open(os.path.join(out, "benth_diet_ices.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
with open(os.path.join(out, "benth_prey_lengths_ices.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["pred", "area", "period", "pred_L_cm", "prey", "prey_L_cm", "mass_g", "quarter"])
    wr.writerows(ratios)

print("Cod and haddock diet by mass (% of identified prey mass), ICES stomach DB; U = unclassified (g)")
print("cats: plk planktivores | ben benthivores | own whiting | wc water column | sfish small demersal fish"
      " | sand sandeel | gadu gadoid unid | fishu fish unid | zoo | epi | neph | inf | crustu crust. unid")
print("plk split: pout / herring / sprat / clupeid unid / blue whiting (% of identified mass)")
hdr = (f"{'pred':8s} {'area':10s} {'period':8s} {'len':6s} {'stom':>5s} {'fed':>5s} {'g':>7s} "
       + " ".join(f"{c:>6s}" for c in CATS[:-1]) + "  |   pout   herr  sprat  clupu    bwh" + f" {'gU':>6s}")
print(hdr)
for r in rows:
    if r["area"] in ("other", "unknown", "CSNS"):
        continue
    print(f"{r['pred']:8s} {r['area']:10s} {r['period']:8s} {r['length']:6s} {r['stomachs']:5d} {r['fed']:5d} {r['g_ident']:7.1f} "
          + " ".join(f"{r[c]:6.1f}" for c in CATS[:-1]) + "  | "
          + " ".join(f"{r['plk_' + l.replace(' ', '_')]:6.1f}" for l in PLK_SP) + f" {r['g_U']:6.1f}")
print(f"\nPrey AphiaIDs without a cached lineage (records): {len(missing)} ids, {sum(missing.values())} records")
print("  " + ", ".join(f"{a}:{n}" for a, n in missing.most_common(40)))
