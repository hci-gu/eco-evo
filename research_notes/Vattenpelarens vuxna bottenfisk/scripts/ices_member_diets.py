"""ICES stomach database (public download on disk, CC BY 4.0): diets BY MASS of the water-column members that
carry the adult part of the WC -> benthivores pair (hake, anglerfish, ling, spurdog, pollack, velvet belly),
North Sea + Skagerrak (51-62.5 N, 4 W - 13 E). Prey mass split into: the model's benthivore members (haddock,
cod, plaice, witch [J/A by the library's Q3 length-at-age-1 cuts 27 / 27 / 16 / 14.8 cm], dab, long rough dab,
lemon sole, other flatfish [always A, the J/B convention]), whiting, other fish, invertebrates. Prey length in
the PreyInformation table (Length, UnitLngt mm/cm) when given; items without a length are reported apart.
Per-stomach records only; empty stomachs counted; nothing is fitted.

Items without a length: length back-calculated from the prey wet weight with the CA length-weight fits
(research_notes/"Fiskgrupper i rutan"/tables/length_weight_all.csv; flatfish n.i. on plaice) - a LOWER bound,
since digestion only removes mass; marked "w".

Usage: python3 -I ices_member_diets.py <ices_stomach dir with x<year>/ folders>
"""
import csv
import glob
import os
import sys
from collections import defaultdict

D = sys.argv[1]
PRED = {"126484": "hake", "126555": "anglerfish", "126461": "ling", "105923": "spurdog", "126440": "pollack",
        "105913": "velvet belly", "126441": "saithe"}
GAD = {"126437": ("haddock", 27.0), "126436": ("cod", 27.1), "127143": ("plaice", 16.0), "127136": ("witch", 14.8)}
FLAT_A = {"127139": "dab", "127137": "long rough dab", "127140": "lemon sole"}
WHITING = {"126438"}
FLAT_NI = {"10331", "125579"}           # Pleuronectiformes, Pleuronectidae not identified further
UNID_FISH = {"293496", "152352", "11707", "125469"}   # Teleostei, Osteichthyes, ... (fish n.i.)
REGIONS = {"Skagerrak/Trench (>=57 N, >=6 E)": lambda la, lo: la >= 57.0 and lo >= 6.0,
           "rest of North Sea": lambda la, lo: not (la >= 57.0 and lo >= 6.0)}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


LWAB = {"haddock": (0.005451, 3.1482), "cod": (0.005786, 3.1395), "plaice": (0.010625, 2.9573),
        "witch": (0.0013706, 3.4312), "dab": (0.009333, 3.0132), "long rough dab": (0.0105, 3.08),
        "lemon sole": (0.007363, 3.0919), "flatfish n.i.": (0.010625, 2.9573)}
CUTS = {"haddock": 27.0, "cod": 27.1, "plaice": 16.0, "witch": 14.8, "flatfish n.i.": 16.0}


def stage_from(sp, lp, w):
    """(stage, length used, flag): J/A by the cut; dab, long rough dab, lemon sole always A."""
    flag = ""
    if lp is None and w and w > 0:
        a, b = LWAB[sp]
        lp = (w / a) ** (1.0 / b)
        flag = "w"
    if sp not in CUTS:
        return "A", lp, flag
    if lp is None:
        return "?", lp, flag
    return ("J" if lp < CUTS[sp] else "A"), lp, flag


res = defaultdict(lambda: defaultdict(float))
pstom = {}
stomprey = defaultdict(lambda: defaultdict(float))
nstom = defaultdict(lambda: [0, 0])
items = defaultdict(list)
for d in sorted(glob.glob(os.path.join(D, "x*"))):
    yr = os.path.basename(d)[1:]
    H = {}
    with open(os.path.join(d, "HaulInformation.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            H[r["tblHaulID"]] = (fnum(r["ShootLat"]), fnum(r["ShootLong"]), r.get("Month"), fnum(r.get("Depth")))
    P = {}
    with open(os.path.join(d, "PredatorInformation.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["AphiaIDPredator"] not in PRED:
                continue
            la, lo, mo, dep = H.get(r["tblHaulID"], (None, None, None, None))
            if la is None or not (51 <= la <= 62.5 and -4 <= lo <= 13):
                continue
            reg = [k for k, f in REGIONS.items() if f(la, lo)][0]
            L = fnum(r["Length"])
            n = int(fnum(r["Number"]) or 1)
            sp = PRED[r["AphiaIDPredator"]]
            key = (sp, reg)
            P[r["tblPredatorInformationID"]] = (key, L, yr, mo, dep)
            nstom[key][0] += n
            if r["StomachEmpty"] != "1":
                nstom[key][1] += n
    with open(os.path.join(d, "PreyInformation.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            pid = r["tblPredatorInformationID"]
            if pid not in P:
                continue
            key, Lp, yr_, mo, dep = P[pid]
            w = fnum(r["Weight"]) or 0.0
            if r.get("UnitWgt", "").strip().lower() == "mg":
                w /= 1000.0
            pstom[pid] = pstom.get(pid, 0.0)
            a = r["AphiaIDPrey"]
            lp = fnum(r["Length"])
            if lp is not None and r["UnitLngt"].strip().lower() == "mm":
                lp /= 10.0
            name = (r.get("OtherItems") or "").lower()
            if a in GAD:
                sp, cut = GAD[a]
                st, lu, fl = stage_from(sp, lp, w)
                cat = f"ben:{sp}:{st}"
                items[key].append((sp, lu, fl + st, w, Lp, yr_, mo, dep))
            elif a in FLAT_A:
                st, lu, fl = stage_from(FLAT_A[a], lp, w)
                cat = f"ben:{FLAT_A[a]}:A"
                items[key].append((FLAT_A[a], lu, fl + "A", w, Lp, yr_, mo, dep))
            elif a in FLAT_NI:
                st, lu, fl = stage_from("flatfish n.i.", lp, w)
                cat = f"ben:flatfish n.i.:{st}"
                items[key].append(("flatfish n.i.", lu, fl + st, w, Lp, yr_, mo, dep))
            elif a in WHITING:
                cat = "whiting"
            elif a in UNID_FISH:
                cat = "fish n.i."
            elif a:
                cat = f"aphia:{a}"
            else:
                cat = "unidentified"
            res[key][cat] += w
            stomprey[(key, pid)][cat] += w

# coarse classes of the other aphia ids via a small lookup of frequent ones
NAMES = {"126444": "Norway pout", "126417": "herring", "126425": "sprat", "126439": "blue whiting",
         "1066": "Crustacea n.i.", "125739": "Trisopterus", "125464": "Clupeidae", "400596": "Norway pout",
         "126281": "eel", "510534": "sea urchins", "106928": "Polybius", "106674": "Caridea", "150630": "lesser weever",
         "125516": "sandeels", "101417": "Hyperiidae", "107011": "Pontophilus",
         "126735": "mackerel", "126822": "horse mackerel", "126751": "lesser sandeel", "126756": "sandeels",
         "126442": "poor cod", "107254": "Nephrops", "110690": "Pandalus", "11707": "Teleostei",
         "125469": "Actinopterygii", "125732": "Gadidae", "126440": "pollack", "126441": "saithe",
         "126484": "hake", "126555": "anglerfish", "126461": "ling", "105923": "spurdog", "127140": "lemon sole",
         "126792": "dragonet", "126796": "dragonet", "110671": "Crangon", "141433": "Pasiphaea",
         "110689": "Pandalus montagui", "107276": "Pandalidae", "11676": "Cephalopoda", "126433": "silvery pout",
         "126436": "cod", "126714": "argentine", "127023": "argentine", "126758": "greater sandeel"}
for key in sorted(res):
    tot = sum(res[key].values())
    if tot <= 0:
        continue
    ben = {k: v for k, v in res[key].items() if k.startswith("ben:")}
    bt = sum(ben.values())
    ident = tot - res[key].get("unidentified", 0.0)
    print(f"\n== {key[0]} | {key[1]}: stomachs {nstom[key][0]} (non-empty {nstom[key][1]}), prey mass {tot:.1f} g"
          f" (identified to a taxon {ident:.1f} g)")
    print(f"   model benthivores {bt:.1f} g = {bt / tot:.3f} of all prey mass ({bt / ident if ident else 0:.3f} of"
          f" identified); whiting {res[key].get('whiting', 0) / tot:.3f}")
    for k, v in sorted(ben.items(), key=lambda kv: -kv[1]):
        print(f"     {k:30s} {v:8.1f} g  {v / tot:.3f}")
    top = sorted(((v, k) for k, v in res[key].items() if not k.startswith("ben:")), reverse=True)[:8]
    print("   largest other items: " + "; ".join(f"{NAMES.get(k.split(':')[-1], k)} {v / tot:.3f}" for v, k in top))
    if items[key]:
        print("   benthivore items (species, prey L cm, [w = from weight] stage, g, predator L cm, year, month, haul depth m):")
        for it in sorted(items[key], key=lambda x: -x[3])[:25]:
            print("     " + ", ".join(str(round(x, 1)) if isinstance(x, float) else str(x) for x in it))
# stomach-equal weighting (each non-empty stomach's prey proportions averaged): guards against one large meal
print("\n== stomach-equal weighting (mean over non-empty stomachs of the prey-mass proportions)")
agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for (key, pid), d in stomprey.items():
    t = sum(d.values())
    if t <= 0:
        continue
    b = sum(v for c, v in d.items() if c.startswith("ben:"))
    ba = sum(v for c, v in d.items() if c.startswith("ben:") and c.endswith(":A"))
    g = agg[key[0]]
    g[0] += 1
    g[1] += b / t
    g[2] += ba / t
    g[3] += d.get("fish n.i.", 0.0) / t
for sp, (n, b, ba, u) in sorted(agg.items()):
    print(f"  {sp:13s} stomachs with prey mass {n:4d}: benthivores {b / n:.3f} (of it length-known adults {ba / n:.3f}); fish n.i. {u / n:.3f}")
# pooled over regions
print("\n== pooled North Sea + Skagerrak")
for sp in sorted(set(k[0] for k in res)):
    tot = sum(sum(res[k].values()) for k in res if k[0] == sp)
    bt = sum(v for k in res if k[0] == sp for c, v in res[k].items() if c.startswith("ben:"))
    ba = sum(v for k in res if k[0] == sp for c, v in res[k].items() if c.startswith("ben:") and c.endswith(":A"))
    bj = sum(v for k in res if k[0] == sp for c, v in res[k].items() if c.startswith("ben:") and c.endswith(":J"))
    nn = sum(nstom[k][1] for k in nstom if k[0] == sp)
    print(f"  {sp:13s} non-empty {nn:4d}, prey {tot:9.1f} g: benthivores {bt / tot:.3f} (A {ba / tot:.3f}, J {bj / tot:.3f},"
          f" no length {(bt - ba - bj) / tot:.3f})")
