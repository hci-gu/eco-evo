"""Nephrops norvegicus in fish stomachs: scan of the public ICES Stomach Content database extracts.

Usage:
  python3 -I ices_nephrops_scan.py <ices_stomach_dir> <out_dir> <worms_dir> [<worms_dir> ...]

<ices_stomach_dir> holds the unzipped public downloads (x1981, x1991, x2000 ... x2025; each with
HaulInformation.csv, PredatorInformation.csv, PreyInformation.csv) already fetched for
../../Torskfiskarnas bytesandelar (stomachdata.ices.dk API, CC BY 4.0). Read as text only.
<worms_dir>: cached WoRMS classification JSON files (cls_<AphiaID>.json) from the sibling notes.

All predator species are scanned (not only the benthivores), so whiting, saithe, hake etc. are covered.
Nephrops prey = any prey AphiaID whose WoRMS lineage contains "Nephrops" (107254, 107253).

Per predator record (one row of PredatorInformation; 1981/1991 records can pool several fish, the
`Number` column) the script keeps: StomachEmpty, Regurgitated, whether prey rows with mass exist.
Food mass = all prey rows with a weight, except parasites, plants and non-food (by WoRMS lineage);
unidentified prey (no AphiaID) stays in the denominator ("of total food"); a second share uses
identified prey only. DRYWT rows are dropped. mg -> g.

Outputs (out_dir):
  ices_nephrops_items.csv     every Nephrops prey row (predator, area, haul depth, predator length, prey
                              count, mass, prey length, digestion stage)
  ices_nephrops_shares.csv    per predator x area x period x size class: records, fish, empty,
                              regurgitated, fed records, total food g, Nephrops g, % mass (total and
                              identified), % occurrence among fed records, haul depth quartiles
  ices_coverage.csv           predator records by species x area x year (what the public subset holds)
"""
import csv, json, os, sys, collections, statistics

root, outd = sys.argv[1], sys.argv[2]
wdirs = sys.argv[3:]

PRED = {"126436": "cod", "126437": "haddock", "126438": "whiting", "126441": "saithe",
        "126484": "hake", "127137": "long_rough_dab", "127139": "dab", "127143": "plaice",
        "127136": "witch", "127140": "lemon_sole", "127141": "flounder", "150637": "grey_gurnard",
        "105865": "starry_ray", "105883": "thornback_ray", "105923": "spurdog", "126555": "anglerfish",
        "126461": "ling", "126439": "blue_whiting", "126444": "norway_pout", "126440": "pollack",
        "126822": "herring", "127023": "mackerel", "127262": "turbot?", "127149": "sole?",
        "127150": "solea?", "101170": "hagfish", "127082": "greater_weever", "126447": "tusk",
        "127138": "halibut?", "126415": "sprat?", "105876": "ray?", "105821": "shark?",
        "127160": "four_spot_megrim?", "126735": "sea_scorpion?"}
GADOID_LIKE = {"cod", "haddock", "whiting", "saithe", "hake", "pollack", "ling", "starry_ray",
               "thornback_ray", "spurdog", "anglerfish", "tusk", "blue_whiting", "norway_pout"}
NONFOOD = {"Nematoda", "Trematoda", "Acanthocephala", "Cestoda", "Plantae", "Chromista", "Bacteria",
           "Monogenea", "Hirudinea"}


def fnum(x):
    try:
        return float(str(x).strip('"'))
    except (TypeError, ValueError):
        return float("nan")


def area_of(lat, lon):
    """Areas as in ../../Fiskgrupper i rutan/benthivores/scripts/benth_common.py."""
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
    if lon > 13 or (lat < 57.5 and lon > 9):
        return "Baltic_or_Belts"
    return "other"


def in_box(lat, lon):
    return 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3


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
                break
            names = []
            while d:
                names.append(d.get("scientificname", ""))
                d = d.get("child")
            res = names
            break
    _lin[aid] = res
    return res


def prey_kind(aid):
    """'neph', 'food', 'unid' (no AphiaID), 'drop'."""
    if not aid:
        return "unid"
    if aid in ("107254", "107253"):
        return "neph"
    L = lineage(aid)
    if L is None:
        return "food_nolin"
    s = set(L)
    if "Nephrops" in s:
        return "neph"
    if s & NONFOOD:
        return "drop"
    if "Animalia" not in s:
        return "drop"
    return "food"


def size_class(L, sp):
    if L != L:
        return "na"
    if sp in GADOID_LIKE:
        edges = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 70), (70, 999)]
    else:
        edges = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 999)]
    for a, b in edges:
        if a <= L < b:
            return f"{a:02d}-{b if b < 999 else ''}"
    return "na"


def depth_bin(d):
    if d != d:
        return "depth_na"
    if d < 50:
        return "d<50"
    if d < 100:
        return "d50-100"
    if d < 200:
        return "d100-200"
    return "d200+"


# accumulators
recs = {}                      # pid -> dict
items = []
cover = collections.Counter()
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p):
        continue
    hauls = {}
    with open(os.path.join(p, "HaulInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            hauls[r["tblHaulID"]] = r
    with open(os.path.join(p, "PredatorInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            aid = r["AphiaIDPredator"].strip('"')
            sp = PRED.get(aid, "aphia_" + aid)
            h = hauls.get(r["tblHaulID"], {})
            lat, lon = fnum(h.get("ShootLat")), fnum(h.get("ShootLong"))
            dep = fnum(h.get("Depth"))
            L = fnum(r["Length"])
            if L == L and L > 200:      # mm
                L /= 10.0
            elif L == L and str(r.get("MeasurementIncrement", "")).strip('"') in ("1",) and L > 150:
                L /= 10.0
            nf = fnum(r.get("Number"))
            yr = int(fnum(r["Year"])) if r["Year"] else 0
            mo = int(fnum(r["Month"])) if r["Month"] and fnum(r["Month"]) == fnum(r["Month"]) else 0
            ar = area_of(lat, lon)
            recs[d + ":" + r["tblPredatorInformationID"]] = dict(
                sp=sp, area=ar, box=in_box(lat, lon), lat=lat, lon=lon, depth=dep, L=L, year=yr, month=mo,
                n=int(nf) if nf == nf and nf >= 1 else 1,
                empty=str(r.get("StomachEmpty", "")).strip('"') in ("1", "Y", "y", "TRUE", "True"),
                regurg=str(r.get("Regurgitated", "")).strip('"') not in ("", "0", "N", "n", "NA"),
                food=0.0, food_id=0.0, neph=0.0, neph_items=0.0, haul=d + ":" + r["tblHaulID"])
            cover[(sp, ar, yr)] += 1
    with open(os.path.join(p, "PreyInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            key = d + ":" + r["tblPredatorInformationID"]
            rec = recs.get(key)
            if rec is None:
                continue
            if r["GravMethod"].strip('"').upper() == "DRYWT":
                continue
            aid = r["AphiaIDPrey"].strip('"')
            kind = prey_kind(aid)
            if kind == "drop":
                continue
            w = fnum(r["Weight"])
            if w != w or w <= 0:
                continue
            if r["UnitWgt"].strip('"').lower() == "mg":
                w /= 1000.0
            rec["food"] += w
            if kind != "unid":
                rec["food_id"] += w
            if kind == "neph":
                c = fnum(r["Count"])
                rec["neph"] += w
                rec["neph_items"] += c if c == c else 1
                pl = fnum(r["Length"]); u = r["UnitLngt"].strip('"').lower()
                if pl == pl and u == "cm":
                    pl *= 10.0
                items.append([rec["sp"], rec["area"], "Box" if rec["box"] else "", rec["year"], rec["month"],
                              round(rec["lat"], 3), round(rec["lon"], 3),
                              rec["depth"] if rec["depth"] == rec["depth"] else "",
                              rec["L"] if rec["L"] == rec["L"] else "", rec["n"],
                              r["Count"].strip('"'), round(w, 3), round(pl, 1) if pl == pl else "",
                              r["UnitLngt"].strip('"'), r["DigestionStage"].strip('"'), aid])

os.makedirs(outd, exist_ok=True)
with open(os.path.join(outd, "ices_nephrops_items.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["predator", "area", "box", "year", "month", "lat", "lon", "haul_depth_m", "pred_length_cm",
                 "pred_records_pooled_n", "nephrops_count", "nephrops_g", "prey_length_mm_as_reported",
                 "prey_length_unit_raw", "digestion_stage", "aphia_prey"])
    wr.writerows(sorted(items, key=lambda x: (x[0], x[1], x[3])))

# aggregate
agg = collections.defaultdict(lambda: {"rec": 0, "fish": 0, "empty": 0, "regurg": 0, "fed": 0, "fed_fish": 0,
                                       "food": 0.0, "food_id": 0.0, "neph": 0.0, "neph_rec": 0,
                                       "neph_items": 0.0, "depths": [], "hauls": set(), "months": collections.Counter()})
for rec in recs.values():
    sp, ar, yr = rec["sp"], rec["area"], rec["year"]
    areas = [ar]
    if rec["box"]:
        areas.append("Box")
    if ar == "NNS" and rec["depth"] == rec["depth"] and rec["depth"] >= 150 and rec["lon"] > 2.5:
        areas.append("NNS_trench")
    if ar in ("Skagerrak", "NNS", "Kattegat", "CSNS"):
        areas.append(ar + "_" + depth_bin(rec["depth"]))
    if ar in ("Skagerrak", "Kattegat"):
        areas.append("SkagKat")
    periods = [str(yr), "pre2000" if yr < 2000 else "2000+", "all_years"]
    if 2015 <= yr:
        periods.append("2015+")
    for a in areas:
        for per in periods:
            for s in (size_class(rec["L"], sp), "all"):
                g = agg[(sp, a, per, s)]
                g["rec"] += 1; g["fish"] += rec["n"]
                g["empty"] += rec["empty"]; g["regurg"] += rec["regurg"]
                g["hauls"].add(rec["haul"]); g["months"][rec["month"]] += 1
                if rec["food"] > 0:
                    g["fed"] += 1; g["fed_fish"] += rec["n"]
                    g["food"] += rec["food"]; g["food_id"] += rec["food_id"]
                    if rec["depth"] == rec["depth"]:
                        g["depths"].append(rec["depth"])
                if rec["neph"] > 0:
                    g["neph"] += rec["neph"]; g["neph_rec"] += 1; g["neph_items"] += rec["neph_items"]


def q(v, p):
    if not v:
        return ""
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]


with open(os.path.join(outd, "ices_nephrops_shares.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["predator", "area", "period", "size_cm", "records", "fish", "empty_flag", "regurg_flag",
                 "fed_records", "fed_fish", "hauls", "months", "food_g", "food_identified_g", "nephrops_g",
                 "pct_mass_of_total", "pct_mass_of_identified", "records_with_nephrops",
                 "pct_occurrence_of_fed", "nephrops_items", "depth_q25", "depth_median", "depth_q75"])
    for k in sorted(agg):
        g = agg[k]
        if g["fed"] == 0:
            continue
        months = " ".join(f"{m}:{n}" for m, n in sorted(g["months"].items()))
        wr.writerow(list(k) + [g["rec"], g["fish"], g["empty"], g["regurg"], g["fed"], g["fed_fish"],
                               len(g["hauls"]), months, round(g["food"], 2), round(g["food_id"], 2),
                               round(g["neph"], 2),
                               round(100 * g["neph"] / g["food"], 2) if g["food"] else "",
                               round(100 * g["neph"] / g["food_id"], 2) if g["food_id"] else "",
                               g["neph_rec"], round(100 * g["neph_rec"] / g["fed"], 2), g["neph_items"],
                               q(g["depths"], 0.25), q(g["depths"], 0.5), q(g["depths"], 0.75)])

with open(os.path.join(outd, "ices_coverage.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["predator", "area", "year", "records"])
    for (sp, ar, yr), n in sorted(cover.items()):
        wr.writerow([sp, ar, yr, n])

# console summary: every group with Nephrops, and all-size rows of the key predators in the key areas
print("Groups with Nephrops (all sizes, by year and period):")
for k in sorted(agg):
    g = agg[k]
    if k[3] == "all" and g["neph"] > 0 and not k[2] in ("all_years",):
        print(f"  {k[0]:14s} {k[1]:20s} {k[2]:9s} fed {g['fed']:4d} ({g['fed_fish']:4d} fish) of {g['rec']:4d} "
              f"rec  food {g['food']:8.1f} g  neph {g['neph']:7.1f} g = {100*g['neph']/g['food']:5.2f} %  "
              f"occ {g['neph_rec']}/{g['fed']}  depth med {q(g['depths'],0.5)}")
print("\nKey predators in Skagerrak/Kattegat/NNS/Box (all sizes, all periods), incl. zero Nephrops:")
for k in sorted(agg):
    g = agg[k]
    if k[3] == "all" and k[0] in ("cod", "haddock", "whiting", "long_rough_dab", "dab", "plaice", "witch",
                                   "lemon_sole", "saithe", "hake") \
            and k[1] in ("Skagerrak", "Kattegat", "NNS", "Box", "NNS_trench", "SkagKat") \
            and k[2] in ("pre2000", "2000+"):
        print(f"  {k[0]:14s} {k[1]:11s} {k[2]:7s} rec {g['rec']:5d} empty {g['empty']:4d} regurg {g['regurg']:3d} "
              f"fed {g['fed']:4d} ({g['fed_fish']:4d} fish) food {g['food']:8.1f} g  neph {g['neph']:6.1f} g "
              f"= {(100*g['neph']/g['food'] if g['food'] else 0):5.2f} %  occ {g['neph_rec']}")
