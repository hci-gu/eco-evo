"""Prey-group densities in the porpoise stomach years (1988-96) relative to today, from NS-IBTS GOV.

Usage: python3 -I then_now_densities.py <extract_1991_96> <extract_2000_20> <extract_2021_25> <lw_csv>

Inputs (untrusted data, parsed only): HH.csv.gz and CPUEL.csv.gz written by
"../../Torskfiskarnas artsammansättning/scripts/extract_datras.py" (region 56.5-59.5 N, 6.5-12.5 E),
and "../../Fiskgrupper i rutan/tables/length_weight_all.csv" (W = a L^b, g and cm; CA fits or
form defaults). A few gadoids missing there take the CA fits of
"../../Torskfiskarnas artsammansättning/tables/length_weight.csv" (hard-coded below).

Method:
- valid GOV hauls (HaulVal V, Gear GOV) in Q1 and Q3; a species absent from a haul is a zero;
- kg/h per haul from CPUE per length class (DATRAS lower bound in mm; weight at the class midpoint);
- Q1 and Q3 means are averaged with equal weight (annual-mean proxy), per region and period;
- "lt40" = length classes below 40 cm (the porpoise prey window used by Ransijn et al. 2021 after
  Aarefjord et al. 1995);
- regions: Skagerrak (east of the Hanstholm-Lindesnes line, as gadoid_mix.py), Kattegat north of
  56.5 N (the extract's edge; lat < 57.75 east of 10.6 E), IIIa = both, box = the four ICES
  rectangles 44G0, 44G1, 45G0, 45G1;
- the pool ratio then/now weights each member's GOV ratio by its share of today's pool biomass
  (the library's composition), so the acoustic herring level of the box is scaled by GOV herring's
  own change, never compared with GOV gadoids in absolute terms (GOV herring is a 3-7x floor,
  "../../Ekolodsdata planktivorer").
"""
import sys

import numpy as np
import pandas as pd

ext91, ext00, ext21, lw_csv = sys.argv[1:5]

SP = {  # AphiaID: (name, pool)
    126417: ("herring", "plk"), 126425: ("sprat", "plk"), 126444: ("Norway pout", "plk"),
    126439: ("blue whiting", "plk"),
    126437: ("haddock", "ben"), 126436: ("cod", "ben"), 127137: ("long rough dab", "ben"),
    127139: ("dab", "ben"), 127143: ("plaice", "ben"), 127136: ("witch", "ben"),
    127140: ("lemon sole", "ben"),
    126438: ("whiting", "wht"),
    126441: ("saithe", "wc"), 126484: ("hake", "wc"), 126440: ("pollack", "wc"), 126461: ("ling", "wc"),
}
# today's pool composition (library / research notes, annual-mean biomass shares)
COMP = {
    "plk": {"herring": 0.79, "blue whiting": 0.105, "Norway pout": 0.075, "sprat": 0.03},
    "ben": {"haddock": 0.386, "cod": 0.314, "long rough dab": 0.126, "dab": 0.045, "plaice": 0.038,
            "witch": 0.017, "lemon sole": 0.009},
    "wht": {"whiting": 1.0},
    # eaten-possible members of the water-column rest (sharks and anglerfish excluded: never in stomachs)
    "wc": {"saithe": 0.797, "hake": 0.048, "ling": 0.014, "pollack": 0.008},
}
EXTRA_LW = {126438: (0.006509, 3.0575), 126441: (0.006494, 3.0788), 126484: (0.005175, 3.0646),
            126440: (0.00692, 3.08), 126461: (0.003710, 3.0882), 126439: (0.003098, 3.1856)}

lw = pd.read_csv(lw_csv, index_col=0)
LW = {a: (lw.loc[a, "a"], lw.loc[a, "b"]) if a in lw.index else EXTRA_LW[a] for a in SP}
missing = [SP[a][0] for a in SP if a not in lw.index]
print("length-weight from length_weight_all.csv; fallback (Torskfiskarnas CA fits / FishBase) used for:",
      ", ".join(missing) or "none")


def load(ext):
    hh = pd.read_csv(f"{ext}/HH.csv.gz")
    hh = hh[(hh.HaulVal.astype(str).str.strip() == "V") & hh.Quarter.isin([1, 3])
            & (hh.Gear.astype(str).str.strip() == "GOV")].copy()
    hh["Ship"] = hh.Ship.astype(str).str.strip()
    hh["StatRec"] = hh.StatRec.astype(str).str.strip()
    hh = hh.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo"])
    lat, lon = hh.ShootLat, hh.ShootLong
    x1, y1, x2, y2 = 8.60, 57.12, 7.05, 57.98
    east = ((x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)) < 0
    hh["skag"] = east & (lat < 59.2) & ~((lon > 10.6) & (lat < 57.75)) & (lon < 12.5) & (lat > 57.0)
    hh["kat"] = (lon > 10.6) & (lat < 57.75) & (lat >= 56.5) & (lon < 12.8)
    hh["iiia"] = hh.skag | hh.kat
    hh["box"] = hh.StatRec.isin(["44G0", "44G1", "45G0", "45G1"])
    hh = hh[hh.iiia | hh.box]
    cp = pd.read_csv(f"{ext}/CPUEL.csv.gz")
    cp = cp[cp.AphiaID.isin(SP)].copy()
    cp["Ship"] = cp.Ship.astype(str).str.strip()
    L = cp.LngtClas / 10.0 + 0.5
    a = cp.AphiaID.map(lambda x: LW[x][0])
    b = cp.AphiaID.map(lambda x: LW[x][1])
    cp["kg_h"] = cp.CPUE_number_per_hour * a * L ** b / 1000.0
    cp["lt40"] = cp.LngtClas < 400
    cp["name"] = cp.AphiaID.map(lambda x: SP[x][0])
    key = ["Year", "Quarter", "Ship", "HaulNo"]
    out = {}
    for tag, sel in [("all", cp), ("lt40", cp[cp.lt40])]:
        m = sel.groupby(key + ["name"]).kg_h.sum().unstack(fill_value=0.0)
        H = hh.set_index(key).join(m, how="left")
        for n in [v[0] for v in SP.values()]:
            if n not in H:
                H[n] = 0.0
        H[[v[0] for v in SP.values()]] = H[[v[0] for v in SP.values()]].fillna(0.0)
        out[tag] = H.reset_index()
    return out


def region_means(H, region, years):
    d = H[H[region] & H.Year.between(*years)]
    names = [v[0] for v in SP.values()]
    q = [d[d.Quarter == k][names].mean() for k in (1, 3)]
    n = [int((d.Quarter == k).sum()) for k in (1, 3)]
    return (q[0] + q[1]) / 2.0, n


D91, D00, D21 = load(ext91), load(ext00), load(ext21)
PERIODS = {"1991-96": (D91, (1991, 1996)), "2001-20": (D00, (2001, 2020)), "2021-25": (D21, (2021, 2025))}


def ref_mean(tag, region, which):
    if which == "2001-25":
        m1, n1 = region_means(D00[tag], region, (2001, 2020))
        m2, n2 = region_means(D21[tag], region, (2021, 2025))
        return (m1 * 20 + m2 * 5) / 25.0, [n1[0] + n2[0], n1[1] + n2[1]]
    data, yrs = PERIODS[which]
    return region_means(data[tag], region, yrs)


for region in ["skag", "iiia", "kat", "box"]:
    print(f"\n=== region {region} ===")
    for tag in ["all", "lt40"]:
        m91, n91 = region_means(D91[tag], region, (1991, 1996))
        print(f"-- {tag}: GOV kg/h (Q1/Q3 mean), hauls Q1/Q3 1991-96 {n91}")
        refs = {}
        for which in ["2001-25", "2021-25"]:
            refs[which] = ref_mean(tag, region, which)
        print(f"   {'species':16s} {'1991-96':>9s} {'2001-25':>9s} {'2021-25':>9s} {'r(01-25)':>9s} {'r(21-25)':>9s}")
        for n in [v[0] for v in SP.values()]:
            r1 = m91[n] / refs["2001-25"][0][n] if refs["2001-25"][0][n] > 0 else np.nan
            r2 = m91[n] / refs["2021-25"][0][n] if refs["2021-25"][0][n] > 0 else np.nan
            print(f"   {n:16s} {m91[n]:9.2f} {refs['2001-25'][0][n]:9.2f} {refs['2021-25'][0][n]:9.2f} {r1:9.2f} {r2:9.2f}")
        for which in ["2001-25", "2021-25"]:
            ref = refs[which][0]
            parts = []
            for pool, comp in COMP.items():
                num = sum(w * (m91[s] / ref[s]) for s, w in comp.items() if ref[s] > 0)
                den = sum(w for s, w in comp.items() if ref[s] > 0)
                parts.append(f"{pool} {num / den:5.2f}")
            print(f"   pool ratio 1991-96 / {which} (today's composition weights): " + ", ".join(parts))
        # composition of the <40 / all within each pool, 1991-96 vs ref
    m91a, _ = region_means(D91["all"], region, (1991, 1996))
    m91s, _ = region_means(D91["lt40"], region, (1991, 1996))
    refa, _ = ref_mean("all", region, "2001-25")
    refs_, _ = ref_mean("lt40", region, "2001-25")
    print("   share of GOV biomass < 40 cm, 1991-96 | 2001-25: " + ", ".join(
        f"{n} {m91s[n] / m91a[n]:.2f}|{refs_[n] / refa[n]:.2f}" for n in ["herring", "cod", "haddock", "whiting", "saithe", "hake", "pollack"]
        if m91a[n] > 0 and refa[n] > 0))

# ---- length structure: share of GOV biomass below L cm, per region and period
print("\nSaithe / hake / whiting / cod length structure: share of GOV biomass below L cm (1991-96 | 2001-20)")
LIMS = (20, 25, 30, 35, 40)
for region in ["skag", "box", "kat"]:
    for n in ["saithe", "hake", "whiting", "cod"]:
        aph = [a for a, v in SP.items() if v[0] == n][0]
        outs = []
        for ext, yrs, D in [(ext91, (1991, 1996), D91), (ext00, (2001, 2020), D00)]:
            cp = pd.read_csv(f"{ext}/CPUEL.csv.gz")
            cp = cp[(cp.AphiaID == aph) & cp.Year.between(*yrs) & cp.Quarter.isin([1, 3])].copy()
            cp["Ship"] = cp.Ship.astype(str).str.strip()
            H = D["all"]
            keys = set(map(tuple, H[H[region] & H.Year.between(*yrs)][["Year", "Quarter", "Ship", "HaulNo"]]
                           .values.tolist()))
            cp = cp[[k in keys for k in zip(cp.Year, cp.Quarter, cp.Ship, cp.HaulNo)]]
            L = cp.LngtClas / 10.0 + 0.5
            w = cp.CPUE_number_per_hour * LW[aph][0] * L ** LW[aph][1]
            tot = w.sum()
            outs.append([w[cp.LngtClas < lim * 10].sum() / tot if tot > 0 else np.nan for lim in LIMS])
        print(f"  {region:5s} {n:8s} " + " ".join(f"<{lim}: {a:.3f}|{b:.3f}" for lim, a, b in zip(LIMS, *outs)))
