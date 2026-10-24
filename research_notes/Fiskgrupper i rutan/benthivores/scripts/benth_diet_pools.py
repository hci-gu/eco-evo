"""Diet by mass of the benthivorous_fish members (and cod fish prey), mapped to the split model pools.

Usage: python3 -I benth_diet_pools.py <ices_stomach_dir> <out_dir> <worms_dir> [<worms_dir2> ...]

Adapted from ../../predators/scripts/predator_diet_pools.py (same lineage logic and areas), with
pools re-cut for this group. Reads the public ICES stomach download (never executes anything from it).
Pools (shares of identified prey mass; U is outside the denominator):
  B    benthic_community (all benthic invertebrates incl. mysids, gammarids, Crangon, Pandalus,
       Nephrops, crabs, polychaetes, molluscs except cephalopods, echinoderms)
  Z    zooplankton (euphausiids, copepods, hyperiids, chaetognaths, appendicularians, pteropods...)
  Pas  pelagic shrimps (Pasiphaea, Sergestidae, Oplophoridae) - judgement case, kept apart
  P    planktivorous_fish (herring, sprat, other clupeids, Norway pout, blue whiting, argentines,
       pearlside, silvery pout, myctophids)
  W    predatory_fish (whiting, saithe, pollack, mackerel, horse mackerel, hake, spurdog, anglerfish, ling)
  Bf   benthivorous_fish = within group (haddock, cod, flatfish, gurnards, rays, weevers, hagfish)
  Sand sandeels (no pool; "other")
  Ofi  other identified fish with no pool (gobies, dragonets, poor cod/bib, rocklings, zoarcids, ...)
  Gu   gadoids not identified to species;  Fu fish not identified below class
  Ceph cephalopods; O other (cnidarians, tunicates)
  U    unidentified (no AphiaID, Crustacea/Arthropoda/Malacostraca not identified further)
Rows with GravMethod DRYWT are dropped. Weight in mg -> g. Pooled over stomachs (no catch weighting).
"fed" = predator records with prey mass; "fed_fish" = sum of the record's Number (pooled records hold
several fish; empties inside a pooled record cannot be separated).
Outputs: diet_pools.csv, diet_pools_taxa.csv, fish_prey_items.csv (fish prey with lengths).
"""
import csv, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benth_common import PRED, area_of, box, fnum

root, outd = sys.argv[1], sys.argv[2]
wdirs = sys.argv[3:]

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

P_G = {"Clupea", "Sprattus", "Engraulis", "Sardina", "Alosa", "Micromesistius", "Argentina",
       "Maurolicus", "Gadiculus"}
P_F = {"Clupeidae", "Argentinidae", "Sternoptychidae", "Myctophidae", "Engraulidae"}
SAND = {"Ammodytidae", "Ammodytes", "Hyperoplus", "Gymnammodytes"}
W_G = {"Merlangius", "Pollachius", "Scomber", "Trachurus", "Merluccius", "Squalus", "Lophius",
       "Molva", "Etmopterus", "Brosme"}
W_F = {"Scombridae", "Carangidae", "Merlucciidae", "Lophiidae", "Squalidae"}
BF_G = {"Melanogrammus", "Gadus", "Myxine", "Amblyraja", "Raja", "Leucoraja", "Dipturus",
        "Eutrigla", "Chelidonichthys", "Trigla", "Trachinus"}
BF_F = {"Pleuronectidae", "Soleidae", "Scophthalmidae", "Bothidae", "Pleuronectiformes", "Rajidae",
        "Triglidae", "Trachinidae", "Myxinidae", "Myxini"}
ZOO = {"Euphausiacea", "Copepoda", "Hyperiidea", "Chaetognatha", "Appendicularia", "Thaliacea",
       "Ctenophora", "Tomopteridae", "Thecosomata", "Limacinidae", "Gymnosomata", "Cladocera"}
PAS = {"Pasiphaeidae", "Sergestidae", "Oplophoridae", "Acanthephyridae", "Luciferidae"}
FISH = {"Actinopterygii", "Actinopteri", "Elasmobranchii", "Chondrichthyes", "Myxini", "Pisces",
        "Teleostei", "Osteichthyes", "Gnathostomata", "Myxinidae", "Teleostei"}
UNID_INV = {"Crustacea", "Arthropoda", "Malacostraca", "Eucarida", "Multicrustacea", "Eumalacostraca"}

def classify(aid):
    """Return (pool, sub, leaf)."""
    if not aid:
        return "U", "no_id", "no_id"
    L = lineage(aid)
    if L is None:
        return "U", "no_lineage", aid
    s = set(L); leaf = L[-1]
    if s & FISH or "Vertebrata" in s:
        if any(n.startswith("Trisopterus esmark") for n in s):   # incl. the misspelt "Trisopterus esmarki" AphiaID
            return "P", "norway_pout", leaf
        if "Clupea" in s:
            return "P", "herring", leaf
        if "Sprattus" in s:
            return "P", "sprat", leaf
        if "Micromesistius" in s:
            return "P", "blue_whiting", leaf
        if "Clupeidae" in s or leaf == "Clupeiformes":
            return "P", "clupeid_unid", leaf
        if s & P_G or s & P_F:
            return "P", "other_P", leaf
        if s & SAND:
            return "Sand", "sandeel", leaf
        if "Merlangius" in s:
            return "W", "whiting", leaf
        if s & W_G or s & W_F:
            return "W", "other_W", leaf
        if "Melanogrammus" in s:
            return "Bf", "haddock", leaf
        if "Gadus" in s:
            return "Bf", "cod", leaf
        if {"Pleuronectiformes"} & s:
            return "Bf", "flatfish", leaf
        if s & BF_G or s & BF_F:
            return "Bf", "other_Bf", leaf
        if {"Gaidropsarus", "Ciliata", "Enchelyopus", "Raniceps"} & s or leaf in ("Trisopterus minutus", "Trisopterus luscus"):
            return "Ofi", "rockling_poorcod", leaf
        if {"Gadidae", "Gadiformes", "Trisopterus", "Lotidae", "Gadoidei"} & s:
            return "Gu", "gadoid_unid", leaf
        if leaf in FISH or leaf in ("Vertebrata", "Chordata", "Perciformes", "Scorpaeniformes", "Eupercaria"):
            return "Fu", "fish_unid", leaf
        if "Gobiidae" in s:
            return "Ofi", "goby", leaf
        if "Callionymidae" in s:
            return "Ofi", "dragonet", leaf
        return "Ofi", "other_fish", leaf
    if "Cephalopoda" in s:
        return "Ceph", "cephalopod", leaf
    if {"Hydrozoa", "Scyphozoa", "Tunicata"} & s and not ({"Appendicularia", "Thaliacea"} & s):
        return "O", "cnid_tunic", leaf
    if s & PAS:
        return "Pas", "pelagic_shrimp", leaf
    if s & ZOO:
        sub = ("euph" if "Euphausiacea" in s else "cop" if "Copepoda" in s else
               "hyp" if "Hyperiidea" in s else "other_Z")
        return "Z", sub, leaf
    if {"Nematoda", "Trematoda", "Acanthocephala", "Cestoda", "Plantae", "Chromista", "Bacteria",
        "Monogenea", "Hirudinea"} & s:
        return "drop", "parasite_plant", leaf
    if leaf in UNID_INV:
        return "U", "crust_unid", leaf
    if "Animalia" in s:
        if "Mysida" in s or "Lophogastrida" in s:
            sub = "mysid"
        elif "Crangon" in s or "Crangonidae" in s:
            sub = "crangonid"
        elif "Pandalidae" in s:
            sub = "pandalid"
        elif "Nephrops" in s:
            sub = "nephrops"
        elif "Brachyura" in s or "Anomura" in s:
            sub = "crab_hermit"
        elif "Amphipoda" in s:
            sub = "amphipod"
        elif "Decapoda" in s:
            sub = "other_decapod"
        elif "Crustacea" in s or "Arthropoda" in s:
            sub = "other_crust"
        elif "Annelida" in s:
            sub = "polychaete"
        elif "Echinodermata" in s:
            sub = "ophiuroid" if "Ophiuroidea" in s else "other_echinoderm"
        elif "Bivalvia" in s:
            sub = "bivalve"
        elif "Mollusca" in s:
            sub = "other_mollusc"
        else:
            sub = "other_inv"
        return "B", sub, leaf
    return "drop", "nonfood", leaf

def sz_of(L, sp):
    if L != L:
        return "na"
    if sp in ("cod", "haddock", "starry_ray"):
        edges = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 70), (70, 999)]
    else:
        edges = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 999)]
    for a, b in edges:
        if a <= L < b:
            return f"{a:02d}-{b if b < 999 else ''}"
    return "na"

agg = collections.defaultdict(lambda: collections.defaultdict(float))
meta = collections.defaultdict(lambda: {"fed": set(), "hauls": set(), "months": collections.Counter(),
                                        "items": collections.Counter(), "nfish": {}})
leafmass = collections.defaultdict(float)
fishrows = []
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
            if r["GravMethod"].strip('"').upper() == "DRYWT":
                continue
            sp, pr = preds[pid]
            h = hauls.get(pr["tblHaulID"], {})
            lat, lon = fnum(h.get("ShootLat")), fnum(h.get("ShootLong"))
            depth = fnum(h.get("Depth"))
            ar = area_of(lat, lon)
            w = fnum(r["Weight"])
            if w != w or w <= 0:
                continue
            if r["UnitWgt"].strip('"').lower() == "mg":
                w /= 1000.0
            aid = r["AphiaIDPrey"].strip('"')
            if aid not in cls_cache:
                cls_cache[aid] = classify(aid)
            pool, sub, leaf = cls_cache[aid]
            if pool == "drop":
                continue
            L = fnum(pr["Length"])
            if L == L and L > 200:   # mm
                L /= 10.0
            yr = int(pr["Year"])
            per = "pre2000" if yr < 2000 else "2000+"
            mo = int(pr["Month"]) if pr["Month"] else 0
            areas = [ar]
            if box(lat, lon):
                areas.append("Box")
            if ar == "NNS" and depth == depth and depth >= 150 and lon > 2.5:
                areas.append("NNS_trench")
            if pool in ("P", "W", "Bf", "Sand", "Ofi", "Gu", "Fu"):
                pl = fnum(r["Length"]); u = r["UnitLngt"].strip('"').lower()
                if pl == pl and u == "mm":
                    pl /= 10.0
                elif u not in ("cm", "mm"):
                    pl = float("nan")
                fishrows.append([sp, ar, "Box" if box(lat, lon) else "", yr, mo, L, pool, sub, leaf,
                                 round(pl, 1) if pl == pl else "", round(w, 3), r["Count"], r["DigestionStage"].strip('"')])
            for a in areas:
                for per2 in (per, str(yr)):
                    for s in (sz_of(L, sp), "all"):
                        k = (sp, a, per2, s)
                        agg[k][pool] += w
                        agg[k]["sub:" + sub] += w
                        m = meta[k]
                        m["fed"].add(pid); m["hauls"].add(pr["tblHaulID"]); m["months"][mo] += 1
                        m["items"][pool] += 1
                        nf = fnum(pr.get("Number"))
                        m["nfish"][pid] = int(nf) if nf == nf and nf >= 1 else 1
                        if s == "all" and per2 == per:
                            leafmass[(sp, a, per, pool, sub, leaf)] += w
POOLS = ["B", "Z", "Pas", "P", "W", "Bf", "Sand", "Ofi", "Gu", "Fu", "Ceph", "O", "U"]
SUBS = ["euph", "cop", "hyp", "other_Z", "herring", "sprat", "clupeid_unid", "norway_pout", "blue_whiting",
        "other_P", "whiting", "other_W", "haddock", "cod", "flatfish", "other_Bf", "goby", "dragonet",
        "rockling_poorcod", "other_fish", "mysid", "crangonid", "pandalid", "nephrops", "crab_hermit",
        "amphipod", "other_decapod", "other_crust", "polychaete", "ophiuroid", "other_echinoderm",
        "bivalve", "other_mollusc", "other_inv"]
os.makedirs(outd, exist_ok=True)
with open(os.path.join(outd, "ices_benth_diet_pools.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["species", "area", "period", "size_cm", "fed", "fed_fish", "hauls", "months", "g_total"] +
                [f"pct_{q}" for q in POOLS] + [f"pct_{q}" for q in SUBS] + [f"items_{q}" for q in POOLS])
    for k in sorted(agg):
        a = agg[k]; m = meta[k]
        tot = sum(a[q] for q in POOLS if q != "U")
        if tot <= 0:
            continue
        months = " ".join(f"{mo}:{n}" for mo, n in sorted(m["months"].items()))
        wr.writerow(list(k) + [len(m["fed"]), sum(m["nfish"].values()), len(m["hauls"]), months, round(sum(a[q] for q in POOLS), 2)] +
                    [round(100 * a[q] / tot, 1) for q in POOLS] +
                    [round(100 * a["sub:" + q] / tot, 1) for q in SUBS] + [m["items"][q] for q in POOLS])
with open(os.path.join(outd, "ices_benth_diet_taxa.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["species", "area", "period", "pool", "sub", "taxon", "g"])
    for k in sorted(leafmass):
        wr.writerow(list(k) + [round(leafmass[k], 3)])
with open(os.path.join(outd, "ices_fish_prey_items.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["predator", "area", "box", "year", "month", "pred_len_cm", "pool", "sub", "taxon",
                 "prey_len_cm", "g", "count", "digestion"])
    wr.writerows(fishrows)
print("done", len(agg), len(fishrows))
