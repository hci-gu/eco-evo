"""Split the benthivorous_fish benthos diet into infauna and mobile epifauna (no model output used).

Usage: python3 -I benth_diet_split.py <ices_benth_diet_pools.csv> <length_class_pct.csv> \
           <qb_species.csv> <diet_species.csv> <out_csv> [cod_nns]

Inputs (all from research_notes/"Fiskgrupper i rutan"/benthivores and "Torskfiskarnas bytesandelar"):
  ices_benth_diet_pools.csv : ICES stomach DB, % of identified prey mass by sub-pool per species/area/period/size.
  length_class_pct.csv      : box length mix (variant A_floor, quarter pooled) for cod and haddock.
  qb_species.csv            : group biomass share and central Q/B per species.
  diet_species.csv          : central benthos share B (% of the diet) per species.

Sub-pools: EPI = mysid, crangonid, pandalid, nephrops, crab_hermit, other_decapod (mobile epifauna);
INF = polychaete, ophiuroid, other_echinoderm, bivalve, other_mollusc, amphipod, other_crust, other_inv.
Per species: epifauna share of its benthos diet e_i, and the composition of each part.
Group: consumption-weighted, w_i = share_i x QB_i x B_i, so the result is the split of the benthos
the GROUP eats (what the prey M2 needs).
Species without stomach data in the database get a judgement e_i (see JUDGE) with a stated basis.
"""
import csv
import sys

pools, lcp, qbf, dietf, out = sys.argv[1:6]
EPI = ["mysid", "crangonid", "pandalid", "nephrops", "crab_hermit", "other_decapod"]
INF = ["polychaete", "ophiuroid", "other_echinoderm", "bivalve", "other_mollusc", "amphipod",
       "other_crust", "other_inv"]
SUBS = EPI + INF
rows = list(csv.DictReader(open(pools)))
mix = {}
for r in csv.DictReader(open(lcp)):
    if r["variant"] == "A_floor" and r["quarter"] == "pooled" and r["species"] in ("cod", "haddock"):
        g = lambda k: float(r[k])
        mix[r["species"]] = {"00-20": g("[0, 10)") + g("[10, 15)") + g("[15, 20)"),
                             "20-30": g("[20, 25)") + g("[25, 30)"), "30-40": g("[30, 40)"),
                             "40-50": g("[40, 50)"), "50-70": g("[50, 60)") + g("[60, 70)"),
                             "70-": g("[70, 200)")}
ORDER = ["00-20", "20-30", "30-40", "40-50", "50-70", "70-"]


def sub_vec(r):
    return {s: float(r["pct_" + s]) for s in SUBS}


def length_weighted(sp, areas, period):
    """box length mix x size-class sub-pool shares (of total diet), classes with < 5 fed borrow a neighbour."""
    D = {}
    for r in rows:
        if r["species"] == sp and r["period"] == period and r["area"] in areas and r["size_cm"] != "all":
            k = r["size_cm"]
            v = sub_vec(r)
            g = float(r["g_total"])
            fed = int(r["fed"])
            if k in D:
                old, og, of = D[k]
                D[k] = ({s: (old[s] * og + v[s] * g) / (og + g) for s in SUBS}, og + g, of + fed)
            else:
                D[k] = (v, g, fed)
    acc = {s: 0.0 for s in SUBS}
    used = []
    for i, k in enumerate(ORDER):
        j = i
        cand = D.get(k)
        while (cand is None or cand[2] < 5) and j < len(ORDER) - 1:
            j += 1
            cand = D.get(ORDER[j])
        jj = i
        while (cand is None or cand[2] < 5) and jj > 0:
            jj -= 1
            cand = D.get(ORDER[jj])
        used.append(f"{k}({cand[2]})")
        for s in SUBS:
            acc[s] += mix[sp][k] * cand[0][s]
    w = sum(mix[sp].values())
    return {s: acc[s] / w for s in SUBS}, " ".join(used)


def pooled_all(sp, areas, periods):
    """mass-weighted pool of the 'all' size rows (g_total weights)."""
    acc = {s: 0.0 for s in SUBS}
    gt = 0.0
    fed = 0
    for r in rows:
        if r["species"] == sp and r["area"] in areas and r["period"] in periods and r["size_cm"] == "all":
            g = float(r["g_total"])
            for s in SUBS:
                acc[s] += g * float(r["pct_" + s])
            gt += g
            fed += int(r["fed"])
    return {s: acc[s] / gt for s in SUBS}, f"{fed} fed, {gt:.0f} g"


DATA = {
    "haddock": ("len", ["NNS", "NNS_trench"], "2025", "ICES-DB NNS + trench Jan-Feb 2025, box length mix"),
    "cod": ("len", ["Skagerrak", "Box"], "2000+", "ICES-DB Skagerrak + box 2023-25, box length mix"),
    "long_rough_dab": ("all", ["CSNS", "NNS"], ["1991"], "ICES-DB 1991 CSNS + NNS (PRE-2000)"),
    "plaice": ("all", ["Skagerrak", "Box"], ["2024"], "ICES-DB Skagerrak + box 2024"),
    "starry_ray": ("all", ["Skagerrak", "Box"], ["2022", "2024"], "ICES-DB Skagerrak + box 2022-24"),
    "grey_gurnard": ("all", ["CSNS"], ["2022"], "ICES-DB CSNS 2022 (24 fed)"),
}
# judgement e_i (low, central, high) for species with no stomach data here
JUDGE = {
    "dab": ((0.2, 0.35, 0.5), "J: M&D 1991 (De Clerck & Torreele 1988) B 98.7 of which 'small mobile epifauna' 33 "
            "(that M&D group includes amphipods, infauna here); Crangon and hermit crabs in dab diets"),
    "witch": ((0.02, 0.05, 0.15), "J: Bergstad et al. 2003 Skagerrak: polychaetes dominate"),
    "lemon_sole": ((0.02, 0.05, 0.15), "J: Rae 1965 via FishBase: polychaetes, few crustaceans"),
    "hagfish": ((0.1, 0.3, 0.5), "J: carrion + polychaetes; shrimp in Myxine guts"),
    "greater_weever": ((0.4, 0.6, 0.8), "J: weevers take Crangon, mysids and small fish"),
    "others": ((0.15, 0.3, 0.45), "J: group mean"),
}
if len(sys.argv) > 6 and sys.argv[6] == "cod_nns":
    # variant: cod diet from the northern North Sea 2023 (Nephrops and Pandalus grounds)
    DATA["cod"] = ("len", ["NNS"], "2000+", "VARIANT ICES-DB NNS 2023, box length mix")
qb = {r["species"]: r for r in csv.DictReader(open(qbf))}
Bsh = {}
for r in csv.DictReader(open(dietf)):
    if r["variant"] in ("central",):
        Bsh[r["species"]] = float(r["B"])
Bsh["cod"] = 49.3  # skag_gu central variant
res = {}
for sp, q in qb.items():
    if sp in DATA:
        kind, areas, per, basis = DATA[sp]
        v, info = length_weighted(sp, areas, per) if kind == "len" else pooled_all(sp, areas, per)
        b = sum(v.values())
        comp = {s: v[s] / b for s in SUBS}
        e = sum(comp[s] for s in EPI)
        res[sp] = (e, e, e, comp, basis + "; " + info)
    else:
        (lo, c, hi), basis = JUDGE[sp]
        res[sp] = (lo, c, hi, None, basis)
print(f"{'species':<16}{'share%':>7}{'QB':>6}{'B%':>6}{'e_epi':>7}  basis")
W = {}
for sp, (lo, c, hi, comp, basis) in res.items():
    w = float(qb[sp]["share_pct"]) / 100 * float(qb[sp]["qb_central"]) * Bsh[sp] / 100
    W[sp] = w
    print(f"{sp:<16}{float(qb[sp]['share_pct']):7.1f}{float(qb[sp]['qb_central']):6.2f}{Bsh[sp]:6.1f}{c:7.3f}  {basis}")
    if comp:
        print("    " + ", ".join(f"{s} {100 * comp[s]:.1f}" for s in SUBS if comp[s] > 0.0005))
Wt = sum(W.values())
for k, lab in ((0, "low J"), (1, "central"), (2, "high J")):
    e = sum(W[sp] * res[sp][k] for sp in res) / Wt
    print(f"GROUP epifauna share of benthos consumption ({lab}): {e:.3f}")
# group composition by sub-pool (data species with their compositions; judged species split pro rata)
grp = {s: 0.0 for s in SUBS}
data_w = sum(W[sp] for sp in res if res[sp][3])
for sp in res:
    if res[sp][3]:
        for s in SUBS:
            grp[s] += W[sp] * res[sp][3][s]
for s in SUBS:
    grp[s] /= data_w
e_grp = sum(W[sp] * res[sp][1] for sp in res) / Wt
epi_d = sum(grp[s] for s in EPI)
with open(out, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["sub_pool", "part", "pct_of_benthos_data_species", "pct_of_part_data_species"])
    for s in SUBS:
        part = "epifauna" if s in EPI else "infauna"
        tot = epi_d if s in EPI else 1 - epi_d
        wr.writerow([s, part, round(100 * grp[s], 2), round(100 * grp[s] / tot, 2)])
    wr.writerow(["GROUP_epifauna_share_central", "", round(100 * e_grp, 2), ""])
print(f"composition of the data species' benthos (consumption weighted, {data_w / Wt:.2f} of the weight):")
print("  EPI " + ", ".join(f"{s} {100 * grp[s]:.1f}" for s in EPI) + f"  = {100 * epi_d:.1f}")
print("  INF " + ", ".join(f"{s} {100 * grp[s]:.1f}" for s in INF) + f"  = {100 * (1 - epi_d):.1f}")
