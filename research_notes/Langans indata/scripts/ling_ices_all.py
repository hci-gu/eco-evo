"""Every ling (Molva molva, AphiaID 126461) stomach in the ICES stomach database on disk (all years, ALL areas),
with every prey item: taxon, count, mass, prey length and the prey/predator length ratio (PPR).

Usage: python3 -I ling_ices_all.py <ices_stomach dir with x<year>/ folders> <out_csv>

Data: research_notes/"Torskfiskarnas bytesandelar" (a-umlaut)/data/diet/ices_stomach (public download, CC BY 4.0).
Prey lengths: the PreyInformation Length (UnitLngt mm or cm); where missing, a fish prey length is back-calculated
from the item mass with the CA length-weight fits used in research_notes/"Vattenpelarens vuxna bottenfisk"
(LOWER bound: digestion removes mass) and flagged "w". Predator length: PredatorInformation Length in cm when the
MeasurementIncrement is 0.5 or 1 (cm classes) - checked against the printed range. Nothing is fitted.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

D, OUT = sys.argv[1], sys.argv[2]
LING = "126461"
NAMES = {"126444": "Norway pout", "126417": "herring", "126425": "sprat", "126439": "blue whiting",
         "1066": "Crustacea n.i.", "125739": "Trisopterus", "125464": "Clupeidae", "126281": "eel",
         "125516": "sandeels", "126735": "mackerel", "126822": "horse mackerel", "126751": "lesser sandeel",
         "126756": "sandeels", "126442": "poor cod", "107254": "Nephrops", "11707": "Teleostei",
         "125469": "Actinopterygii", "125732": "Gadidae", "126440": "pollack", "126441": "saithe",
         "126484": "hake", "126555": "anglerfish", "126461": "ling", "105923": "spurdog", "127140": "lemon sole",
         "126792": "dragonet", "110671": "Crangon", "11676": "Cephalopoda", "126433": "silvery pout",
         "126436": "cod", "126437": "haddock", "126438": "whiting", "127143": "plaice", "127136": "witch",
         "127139": "dab", "127137": "long rough dab", "10331": "Pleuronectiformes", "125579": "Pleuronectidae",
         "293496": "Teleostei", "152352": "Osteichthyes", "126715": "argentine", "126758": "greater sandeel",
         "127160": "sole", "126445": "Trisopterus esmarkii", "400596": "Norway pout"}
BEN = {"cod", "haddock", "plaice", "witch", "dab", "long rough dab", "lemon sole", "Pleuronectiformes",
       "Pleuronectidae"}
LWAB = {"haddock": (0.005451, 3.1482), "cod": (0.005786, 3.1395), "plaice": (0.010625, 2.9573),
        "witch": (0.0013706, 3.4312), "dab": (0.009333, 3.0132), "long rough dab": (0.0105, 3.08),
        "lemon sole": (0.007363, 3.0919), "Pleuronectiformes": (0.010625, 2.9573),
        "Pleuronectidae": (0.010625, 2.9573), "herring": (0.0051, 3.15), "Norway pout": (0.0066, 3.12),
        "Trisopterus esmarkii": (0.0066, 3.12), "Trisopterus": (0.0066, 3.12), "whiting": (0.0055, 3.12),
        "blue whiting": (0.0040, 3.15), "sprat": (0.0050, 3.12), "mackerel": (0.0039, 3.2)}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


rows = []
nstom = defaultdict(lambda: [0, 0])
for d in sorted(glob.glob(os.path.join(D, "x*"))):
    H = {}
    with open(os.path.join(d, "HaulInformation.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            H[r["tblHaulID"]] = (fnum(r["ShootLat"]), fnum(r["ShootLong"]), r.get("Depth"), r.get("Gear"),
                                 r.get("ICESrectangle"))
    P = {}
    with open(os.path.join(d, "PredatorInformation.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["AphiaIDPredator"] != LING:
                continue
            la, lo, dep, gear, rect = H.get(r["tblHaulID"], (None,) * 5)
            n = int(fnum(r["Number"]) or 1)
            P[r["tblPredatorInformationID"]] = dict(year=r["Year"], month=r["Month"], lat=la, lon=lo, depth=dep, gear=gear,
                                                     rect=rect, L=fnum(r["Length"]), inc=r["MeasurementIncrement"], n=n,
                                                     empty=r["StomachEmpty"] == "1")
            area = "NS+Sk (51-62.5N, 4W-13E)" if (la is not None and 51 <= la <= 62.5 and -4 <= lo <= 13) else "other"
            nstom[area][0] += n
            if r["StomachEmpty"] != "1":
                nstom[area][1] += n
    with open(os.path.join(d, "PreyInformation.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            p = P.get(r["tblPredatorInformationID"])
            if p is None:
                continue
            w = fnum(r["Weight"])
            if w is not None and r.get("UnitWgt", "").strip().lower() == "mg":
                w /= 1000.0
            lp = fnum(r["Length"])
            if lp is not None and r["UnitLngt"].strip().lower() == "mm":
                lp /= 10.0
            a = r["AphiaIDPrey"]
            name = NAMES.get(a, (r.get("OtherItems") or "").strip() or f"aphia:{a}")
            fl = ""
            if lp is None and w and name in LWAB:
                aa, bb = LWAB[name]
                lp = (w / aa) ** (1 / bb)
                fl = "w"
            rows.append(dict(year=p["year"], month=p["month"], lat=p["lat"], lon=p["lon"], depth=p["depth"],
                             gear=p["gear"], predL=p["L"], inc=p["inc"], pid=r["tblPredatorInformationID"], prey=name,
                             aphia=a, count=r["Count"], g=w, preyL=lp, Lflag=fl,
                             ppr=(lp / p["L"]) if (lp and p["L"]) else None))
print("ling stomachs in the database (all years): " + "; ".join(f"{k}: {v[0]} ({v[1]} non-empty)" for k, v in nstom.items()))
Ls = sorted(p for p in (r["predL"] for r in rows) if p)
print(f"prey rows {len(rows)}; predator length of fed ling (cm): min {Ls[0] if Ls else '-'}, max {Ls[-1] if Ls else '-'}")
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n{'year':4s} {'mo':>2s} {'lat':>6s} {'lon':>6s} {'dep':>4s} {'gear':5s} {'predL':>5s} prey (count, g, prey L cm, PPR)")
for r in sorted(rows, key=lambda r: (r["year"], r["pid"])):
    pl = "" if r["preyL"] is None else "%.1f%s" % (r["preyL"], r["Lflag"])
    pr = "" if r["ppr"] is None else ", PPR %.2f" % r["ppr"]
    g = "-" if r["g"] is None else "%.2f" % r["g"]
    print(f"{r['year']:4s} {r['month']:>2s} {r['lat'] or 0:6.2f} {r['lon'] or 0:6.2f} {str(r['depth']):>4s} {str(r['gear'])[:5]:5s}"
          f" {r['predL'] or 0:5.0f} {r['prey']}, n {r['count']}, {g} g, {pl}{pr}")
# mass and occurrence by class
cls = defaultdict(lambda: [0.0, set()])
fed = set(r["pid"] for r in rows)
for r in rows:
    c = ("benthivore member" if r["prey"] in BEN else "whiting" if r["prey"] == "whiting" else
         "planktivore member" if r["prey"] in ("herring", "sprat", "Norway pout", "blue whiting", "Trisopterus esmarkii",
                                              "Trisopterus") else
         "other/unid fish" if r["prey"] in ("Teleostei", "Osteichthyes", "Actinopterygii", "Gadidae", "Clupeidae",
                                          "poor cod", "argentine", "mackerel", "horse mackerel", "sandeels", "dragonet",
                                          "lesser sandeel", "greater sandeel", "silvery pout", "hake") else "other")
    cls[c][0] += r["g"] or 0.0
    cls[c][1].add(r["pid"])
tot = sum(v[0] for v in cls.values())
print(f"\nby class over all {len(fed)} ling stomachs with prey rows (mass share of {tot:.1f} g; frequency of occurrence):")
for c, (g, s) in sorted(cls.items(), key=lambda kv: -kv[1][0]):
    print(f"  {c:20s} {g:8.1f} g {g / tot if tot else 0:.3f}   FO {len(s) / len(fed):.2f}")
pp = sorted(r["ppr"] for r in rows if r["ppr"] and r["prey"] not in ("Crustacea n.i.", "Nephrops", "Crangon"))
if pp:
    import statistics
    print(f"\nfish-prey PPR (n {len(pp)}, w-flag lower bounds included): median {statistics.median(pp):.2f}, "
          f"max {pp[-1]:.2f}, values {', '.join(f'{x:.2f}' for x in pp)}")
