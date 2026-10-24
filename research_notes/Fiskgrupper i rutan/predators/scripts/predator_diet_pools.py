"""Diet by mass of the predatory_fish members, mapped to the split model pools.

Usage: python3 -I predator_diet_pools.py <ices_stomach_dir> <out_csv> <worms_dir> [<worms_dir2> ...]

Reads the public ICES stomach download (HaulInformation, PredatorInformation, PreyInformation per
year folder; never executes anything from it). Each prey AphiaID is classified from its cached WoRMS
lineage (cls_<id>.json). Pools:
  P  planktivorous_fish (clupeids, Norway pout, blue whiting, sandeels, argentines, pearlside,
     silvery pout, myctophids, anchovy, sticklebacks)
  Bf benthivorous_fish (haddock, cod, flatfish, rays, gurnards, weevers, gobies, dragonets,
     poor cod/bib, rocklings, zoarcids, pholids, cottids, agonids, lumpsuckers, other demersal)
  W  within-group predatory_fish (whiting, saithe, pollack, mackerel, horse mackerel, hake,
     spurdog, anglerfish, ling)
  Gu gadoids not identified to species (Gadidae/Gadiformes/Trisopterus sp.)
  Fu fish not identified below class
  Z  zooplankton (euphausiids, copepods, hyperiids, Pasiphaea and other pelagic shrimps,
     chaetognaths, appendicularians, pteropods, Tomopteris)
  B  benthic_community (mysids, gammarids, decapods incl. Crangon, Pandalus, Nephrops, crabs;
     polychaetes, molluscs, echinoderms, other invertebrates)
  O  other (cephalopods, cnidarians, tunicates)
  U  no AphiaID (digested remains); Uc crustaceans not identified below Crustacea/Malacostraca/Eucarida.
     U and Uc are outside the denominator (pct_U, pct_Uc are relative to identified mass).
  Zpas, Zeuph: Pasiphaea-type pelagic shrimps and euphausiids (sub-shares of Z).
  Areas: Skagerrak = N of 57 N, E of the Hanstholm-Lindesnes line (clamped at 7.05 E north of 58 N),
  S of 59.2 N, outside Kattegat; Kattegat 55.5-57.75 N, 10.3-13 E; NNS >= 57.5 N; CSNS 51-57.5 N.
Mass is prey Weight converted to g by UnitWgt (mg -> /1000). Pooled over stomachs (no catch weighting).
"""
import csv, json, os, sys, collections

root, out = sys.argv[1], sys.argv[2]
wdirs = sys.argv[3:]
PRED = {"126441": "saithe", "126438": "whiting", "127023": "mackerel", "126822": "horse_mackerel",
        "105923": "spurdog", "126555": "anglerfish", "126484": "hake"}

def lineage(aid):
    for w in wdirs:
        p = os.path.join(w, f"cls_{aid}.json")
        if os.path.exists(p):
            try:
                d = json.load(open(p))
            except Exception:
                return None
            names = []
            while d:
                names.append(d.get("scientificname", ""))
                d = d.get("child")
            return names
    return None

PLANK_G = {"Clupea", "Sprattus", "Engraulis", "Sardina", "Alosa", "Micromesistius", "Ammodytes",
           "Hyperoplus", "Gymnammodytes", "Argentina", "Maurolicus", "Gadiculus"}
PLANK_F = {"Clupeidae", "Ammodytidae", "Argentinidae", "Sternoptychidae", "Myctophidae", "Engraulidae",
           "Gasterosteidae"}
WITHIN_G = {"Merlangius", "Pollachius", "Scomber", "Trachurus", "Merluccius", "Squalus", "Lophius",
            "Molva", "Etmopterus", "Brosme"}
WITHIN_F = {"Scombridae", "Carangidae", "Merlucciidae", "Lophiidae", "Squalidae"}
BENTH_G = {"Melanogrammus", "Gadus"}
ZOO = {"Euphausiacea", "Copepoda", "Hyperiidea", "Pasiphaeidae", "Sergestidae", "Oplophoridae",
       "Acanthephyridae", "Luciferidae", "Chaetognatha", "Appendicularia", "Thaliacea", "Ctenophora",
       "Tomopteridae", "Thecosomata", "Limacinidae", "Gymnosomata", "Cladocera", "Ostracoda_pelagic"}
FISH = {"Actinopterygii", "Actinopteri", "Elasmobranchii", "Chondrichthyes", "Myxini", "Pisces",
        "Teleostei", "Osteichthyes", "Gnathostomata", "Myxinidae"}
UNID_INV = {"Crustacea", "Arthropoda", "Malacostraca", "Eucarida", "Multicrustacea", "Eumalacostraca"}

def classify(aid):
    if not aid:
        return "U", "no_id"
    L = lineage(aid)
    if L is None:
        return "U", "no_lineage"
    s = set(L)
    leaf = L[-1]
    if s & FISH or "Vertebrata" in s:
        if "Trisopterus esmarkii" in s or leaf.startswith("Trisopterus esmark"):
            return "P", "Norway pout"
        if s & PLANK_G or s & PLANK_F:
            return "P", leaf
        if s & WITHIN_G or s & WITHIN_F:
            return "W", leaf
        if s & BENTH_G:
            return "Bf", leaf
        if "Trisopterus" in s and leaf in ("Trisopterus minutus", "Trisopterus luscus"):
            return "Bf", leaf
        if {"Gadidae", "Gadiformes", "Trisopterus", "Lotidae", "Gadoidei"} & s:
            # rocklings (Gaidropsarus, Ciliata, Enchelyopus) are benthic
            if {"Gaidropsarus", "Ciliata", "Enchelyopus", "Raniceps"} & s:
                return "Bf", leaf
            return "Gu", leaf
        rank_low = len(L) >= 1 and leaf not in FISH and leaf not in ("Vertebrata", "Chordata")
        if not rank_low:
            return "Fu", leaf
        if leaf in ("Clupeiformes",):
            return "P", leaf
        if leaf in ("Gadiformes", "Perciformes", "Scorpaeniformes"):
            return "Fu", leaf
        return "Bf", leaf  # other identified fish: demersal (flatfish, gobies, dragonets, etc.)
    if "Cephalopoda" in s:
        return "O", "Cephalopoda"
    if {"Hydrozoa", "Scyphozoa", "Tunicata"} & s and not ({"Appendicularia", "Thaliacea"} & s):
        return "O", leaf
    if s & ZOO:
        if {"Pasiphaeidae", "Sergestidae", "Oplophoridae", "Acanthephyridae", "Luciferidae"} & s:
            return "Z", "PAS:" + leaf
        return "Z", leaf
    if {"Nematoda", "Trematoda", "Acanthocephala", "Cestoda", "Plantae", "Chromista", "Bacteria"} & s:
        return "drop", leaf
    if leaf in UNID_INV:
        return "Uc", leaf
    if "Animalia" in s:
        return "B", leaf
    return "drop", leaf

def area_of(lat, lon):
    if lat != lat or lon != lon:
        return "na"
    if 55.5 <= lat < 57.75 and 10.3 <= lon <= 13:
        return "Kattegat"
    # Hanstholm (8.6 E, 57.12 N) - Lindesnes (7.05 E, 58.0 N) line
    if lat >= 57.0 and lon <= 13:
        lon_line = 8.6 + (min(lat, 58.0) - 57.12) * (7.05 - 8.6) / (58.0 - 57.12)  # clamp at Lindesnes
        if lat <= 59.2 and lon >= lon_line:
            return "Skagerrak"
    if lat >= 57.5 and -4 <= lon <= 13:
        return "NNS"
    if 51 <= lat < 57.5 and -4 <= lon <= 9:
        return "CSNS"
    return "other"

def box(lat, lon):
    return 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")

agg = collections.defaultdict(lambda: collections.defaultdict(float))
meta = collections.defaultdict(lambda: {"fed": set(), "hauls": set(), "months": collections.Counter(), "grav": set(), "items": collections.Counter()})
leafmass = collections.defaultdict(float)
cls_cache = {}
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p):
        continue
    hauls = {}
    with open(os.path.join(p, "HaulInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            hauls[r["tblHaulID"]] = r
    preds = {}
    with open(os.path.join(p, "PredatorInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            sp = PRED.get(r["AphiaIDPredator"].strip('"'))
            if sp:
                preds[r["tblPredatorInformationID"]] = (sp, r)
    with open(os.path.join(p, "PreyInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            pid = r["tblPredatorInformationID"]
            if pid not in preds:
                continue
            sp, pr = preds[pid]
            h = hauls.get(pr["tblHaulID"], {})
            lat, lon = fnum(h.get("ShootLat")), fnum(h.get("ShootLong"))
            ar = area_of(lat, lon)
            w = fnum(r["Weight"])
            if w != w or w <= 0:
                continue
            if r["UnitWgt"].strip('"').lower() == "mg":
                w /= 1000.0
            aid = r["AphiaIDPrey"].strip('"')
            if aid not in cls_cache:
                cls_cache[aid] = classify(aid)
            pool, leaf = cls_cache[aid]
            if pool == "drop":
                continue
            L = fnum(pr["Length"])
            if L == L and L > 200:   # mm
                L /= 10.0
            sz = "na" if L != L else f"{int(L // 10) * 10:02d}"
            yr = int(pr["Year"])
            per = "pre2000" if yr < 2000 else str(yr)
            mo = int(pr["Month"]) if pr["Month"] else 0
            for a in (ar, "Box" if box(lat, lon) else None):
                if a is None:
                    continue
                for s in (sz, "all"):
                    k = (sp, a, per, s)
                    agg[k][pool] += w
                    if leaf.startswith("PAS:"):
                        agg[k]["Zpas"] += w
                    elif pool == "Z" and leaf in ("Euphausiacea", "Euphausiidae", "Meganyctiphanes norvegica", "Thysanoessa", "Thysanoessa inermis", "Thysanoessa raschii", "Thysanoessa longicaudata", "Nyctiphanes couchii"):
                        agg[k]["Zeuph"] += w
                    m = meta[k]
                    m["fed"].add(pid); m["hauls"].add(pr["tblHaulID"]); m["months"][mo] += 1
                    m["grav"].add(r["GravMethod"].strip('"')); m["items"][pool] += 1
                    if s == "all":
                        leafmass[(sp, a, per, pool, leaf)] += w
POOLS = ["P", "Bf", "W", "Gu", "Fu", "Z", "B", "O", "U", "Uc"]
for k in list(agg):
    pass
with open(out, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["species", "area", "period", "size_cm", "fed", "hauls", "months", "grav", "g_total"] +
                [f"pct_{q}" for q in POOLS] + ["pct_Zpas", "pct_Zeuph"] + [f"items_{q}" for q in POOLS])
    for k in sorted(agg):
        a = agg[k]; m = meta[k]
        tot = sum(a[q] for q in POOLS if q not in ("U", "Uc"))
        if tot <= 0:
            continue
        months = " ".join(f"{mo}:{n}" for mo, n in sorted(m["months"].items()))
        wr.writerow(list(k) + [len(m["fed"]), len(m["hauls"]), months, "/".join(sorted(m["grav"])), round(sum(a[q] for q in POOLS), 2)] +
                    [round(100 * a[q] / tot, 1) for q in POOLS] + [round(100 * a["Zpas"] / tot, 1), round(100 * a["Zeuph"] / tot, 1)] + [m["items"][q] for q in POOLS])
with open(out.replace(".csv", "_taxa.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["species", "area", "period", "pool", "taxon", "g"])
    for k in sorted(leafmass):
        wr.writerow(list(k) + [round(leafmass[k], 3)])
