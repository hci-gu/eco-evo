"""Box share of each member stock's survey index: all sizes (biomass), spawners, age-0 and age-1
(WP3, box_share.md).

Usage: python3 -I share_survey.py <cpuel_members.csv.gz> <hh_2000_20.csv.gz> <hh_2021_25.csv.gz>
                                   <length_weight_all.csv> <etopo_csv> <depth_npz> <out_dir>

Inputs (untrusted data, parsed only): share_extract.py output (NS-IBTS CPUE per length, member
species, whole survey area); the HH haul extracts of "Fiskgrupper i rutan" (2000-20) and
"Torskfiskarnas artsammansattning" (2021-25) (Swedish letters); the length-weight table of
"Fiskgrupper i rutan"; an ETOPO1 elevation grid (2 arc-min) for the sea area of each ICES
rectangle; maps/depth.npz for the box depth bins.

Method:
- valid GOV hauls, Q1 and Q3, 2001-2025; haul key includes the shoot position (the 26D4 collision);
- per haul and species: kg/h (W = a L^b), and numbers/h of the stage windows below;
- rectangle mean per year and quarter -> mean over the years with data (climatology per quarter);
- stock-area index = sum over rectangles of (rectangle mean x rectangle sea area);
- box (R) = mean of the two box rectangles 44G0 and 45G0 x 3600 km2;
  box (S) = Skagerrak + box-rectangle hauls in depth bins, weighted by the box area per bin, x 3600;
- share = box / stock-area index. Within one survey q cancels except through depth and length mix.
- per-year shares (same estimator, rectangles sampled that year) give the year-to-year range.
"""
import os
import sys

import numpy as np
import pandas as pd

cp_csv, hh1, hh2, lw_csv, etopo_csv, depth_npz, out = sys.argv[1:8]
os.makedirs(out, exist_ok=True)
BOX_AREA = 3600.0
YEARS = (2001, 2025)
HALF_CM = {126417, 126425}

# stage windows (cm, lower inclusive, upper exclusive) and the quarter they are read in [J]
# age0: Q3 (August) young of the year; age1: Q1 (Jan-Feb) one-year-olds; mature: Q1 pre-spawning
# (Q3 for the autumn-spawning herring), length at ~50 % maturity.
SP = {
    126417: ("herring", dict(age0=(3, 3, 16), age1=(1, 10, 20), mature=(3, 24, 99))),
    126425: ("sprat", dict(age0=(3, 3, 9), age1=(1, 6, 11), mature=(1, 11, 99))),
    126444: ("Norway pout", dict(age0=(3, 3, 11), age1=(1, 8, 14), mature=(1, 14, 99))),
    126439: ("blue whiting", dict(mature=(3, 22, 99))),
    126437: ("haddock", dict(age0=(3, 4, 20), age1=(1, 10, 25), mature=(1, 30, 999))),
    126436: ("cod", dict(age0=(3, 4, 20), age1=(1, 10, 27), mature=(1, 50, 999))),
    126438: ("whiting", dict(age0=(3, 4, 19), age1=(1, 8, 22), mature=(1, 22, 999))),
    126441: ("saithe", dict(age0=(3, 4, 20), age1=(1, 10, 30), mature=(1, 60, 999))),
    127143: ("plaice", dict(age1=(1, 5, 15), mature=(1, 27, 999))),
    127139: ("dab", dict(mature=(1, 15, 999))),
    127137: ("long rough dab", dict(mature=(1, 18, 999))),
    127136: ("witch", dict(mature=(1, 32, 999))),
    126484: ("hake", dict(age0=(3, 4, 20), mature=(1, 45, 999))),
    127023: ("mackerel", dict()),
    126822: ("horse mackerel", dict()),
    105923: ("spurdog", dict(mature=(1, 80, 999))),
}
# stock areas in survey terms: divisions (from rectangle centre) that the ICES stock covers
AREAS = {
    "4+3a": lambda d, la, lo: d in ("4a", "4b", "4c", "3a20", "3a21"),
    "4+3a20": lambda d, la, lo: d in ("4a", "4b", "4c", "3a20"),
    "3a (20+21)": lambda d, la, lo: d in ("3a20", "3a21"),
    "3a20 (Skagerrak)": lambda d, la, lo: d == "3a20",
    # cod Viking substock proxy [J]: 3a.20 + 4a/4b east of 2 E (or 4 E) and north of 56 N
    "Viking>2E": lambda d, la, lo: d == "3a20" or (d in ("4a", "4b") and lo > 2.0 and la > 56.0),
    "Viking>4E": lambda d, la, lo: d == "3a20" or (d in ("4a", "4b") and lo > 4.0 and la > 56.0),
}


def rect_of(lat, lon):
    """ICES statistical rectangle code (B0 = 40-39 W ... G0 = 10-11 E; letter I unused)."""
    row = np.floor((np.asarray(lat) - 36.0) * 2).astype(int) + 1
    k = np.floor(np.asarray(lon) + 40.0).astype(int)
    return [f"{r:02d}{'BCDEFGHJKLM'[c // 10]}{c % 10}" if c >= 0 else f"{r:02d}A{c + 4}"
            for r, c in zip(row, k)]


def rect_centre(code):
    row, letter, digit = int(code[:2]), code[2], int(code[3])
    lat = 36.0 + (row - 1) * 0.5 + 0.25
    k = ("BCDEFGHJKLM".index(letter)) * 10 + digit  # B0 = -40 E
    lon = -40.0 + k + 0.5
    return lat, lon


def division(lat, lon):
    """ICES division of a rectangle centre (North Sea / 3a only; others -> 'other')."""
    if lon < -4 or lat < 51.0 or lat > 62.0:
        return "other"
    # 3a/4 boundary: Hanstholm (8.60 E, 57.12 N) - Lindesnes (7.05 E, 57.98 N)
    x1, y1, x2, y2 = 8.60, 57.12, 7.05, 57.98
    east = (x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1) < 0
    if lon > 10.0 and 56.0 <= lat < 57.75:
        return "3a21"  # Kattegat
    if lon >= 11.0 and 57.5 <= lat < 58.0:
        return "3a21"  # 44G1: mostly Kattegat sea area
    if east and lon > 7.0 and lat > 56.9:
        return "3a20"
    if lon > 9.5 and lat < 57.0:
        return "other"  # Belt Sea / Baltic side
    if lat >= 57.5:
        return "4a"
    if lat >= 53.5:
        return "4b"
    return "4c"


# ---------------------------------------------------------------- sea area per rectangle
E = pd.read_csv(etopo_csv, skiprows=[1])
E = E[E.altitude < 0]
dlat = np.diff(np.unique(E.latitude))[0]
dlon = np.diff(np.unique(E.longitude))[0]
E["km2"] = (dlat * 111.2) * (dlon * 111.2 * np.cos(np.radians(E.latitude)))
E["rect"] = rect_of(E.latitude.values, E.longitude.values)
SEA = E.groupby("rect").km2.sum()

# ---------------------------------------------------------------- hauls
H = pd.concat([pd.read_csv(hh1), pd.read_csv(hh2)], ignore_index=True)
H = H[(H.HaulVal.str.strip() == "V") & H.Quarter.isin([1, 3]) & (H.Gear.str.strip() == "GOV")
      & H.Year.between(*YEARS)].copy()
H["Ship"] = H.Ship.astype(str).str.strip()
H["la"], H["lo"] = H.ShootLat.round(2), H.ShootLong.round(2)
H = H.drop_duplicates(["Year", "Quarter", "Ship", "HaulNo", "la", "lo"])
H["rect"] = rect_of(H.ShootLat.values, H.ShootLong.values)
H["div"] = [division(*rect_centre(r)) for r in H.rect]
lat, lon = H.ShootLat, H.ShootLong
x1, y1, x2, y2 = 8.60, 57.12, 7.05, 57.98
east = (x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1) < 0
H["skag"] = east & (lat < 59.2) & ~((lon > 10.6) & (lat < 57.75)) & (lon < 12.5) & (lat > 57.0)
H["boxrect"] = H.rect.isin(["44G0", "45G0"])
H["region"] = H.skag | H.rect.isin(["44G0", "44G1", "45G0", "45G1"])
H["dbin"] = pd.cut(H.Depth, [0, 100, 200, 300, 900], labels=["<100", "100-200", "200-300", ">=300"],
                   right=False)
ws = H.WingSpread.where(H.WingSpread > 0, np.nan)
dist = H.Distance.where(H.Distance > 0, np.nan)
dist = dist.fillna(H.GroundSpeed.where(H.GroundSpeed > 0) * 1852 * H.HaulDur / 60)
H["km2_per_h"] = (ws.fillna(ws.median()) / 1000) * (dist.fillna(dist.median()) / 1000) / (H.HaulDur / 60)
H["key"] = list(zip(H.Year, H.Quarter, H.Ship, H.HaulNo, H.la, H.lo))
H = H.set_index("key")

# ---------------------------------------------------------------- CPUE -> per-haul stage indices
LW = pd.read_csv(lw_csv, index_col=0)
C = pd.read_csv(cp_csv)
C["Ship"] = C.Ship.astype(str).str.strip()
C = C[C.Year.between(*YEARS)]
C["key"] = list(zip(C.Year, C.Quarter, C.Ship, C.HaulNo, C.ShootLat.round(2), C.ShootLon.round(2)))
C = C[C.key.isin(set(H.index))].copy()
C["L"] = C.LngtClas / 10.0 + np.where(C.AphiaID.isin(HALF_CM), 0.25, 0.5)
C["kg_h"] = C.CPUE_number_per_hour * C.AphiaID.map(LW.a) * C.L ** C.AphiaID.map(LW.b) / 1000
cols = {}
for aph, (name, st) in SP.items():
    c = C[C.AphiaID == aph]
    cols[(name, "kg")] = c.groupby("key").kg_h.sum()
    for s, (q, lo, hi) in st.items():
        k = c[(c.Quarter == q) & (c.L >= lo) & (c.L < hi)]
        cols[(name, s)] = k.groupby("key").CPUE_number_per_hour.sum()
        cols[(name, s + "_kg")] = k.groupby("key").kg_h.sum()
X = pd.DataFrame(cols).reindex(H.index).fillna(0.0)
X.columns = pd.MultiIndex.from_tuples(X.columns)

z = np.load(depth_npz)["depth"].astype(float)
z = np.abs(z[np.isfinite(z)])
BINS = ["<100", "100-200", "200-300", ">=300"]
barea = pd.cut(pd.Series(z), [0, 100, 200, 300, 900], labels=BINS, right=False).value_counts(
    normalize=True).reindex(BINS)


def box_S(sel):
    """depth-stratified box mean (Skagerrak + box-rectangle hauls); >= 300 m takes 200-300 m if empty."""
    d = X.loc[sel].groupby(H.loc[sel, "dbin"], observed=False).mean().reindex(BINS)
    if d.loc[">=300"].isna().all():
        d.loc[">=300"] = d.loc["200-300"]
    return d.mul(barea, axis=0).sum()


def rect_clim(q, years=None):
    sel = H.Quarter == q
    if years is not None:
        sel &= H.Year.isin(years)
    m = X.loc[sel].groupby([H.loc[sel, "rect"], H.loc[sel, "Year"]]).mean()
    return m.groupby(level=0).mean(), m.groupby(level=0).size()


rows = []
yrows = []
rect_div = {r: division(*rect_centre(r)) for r in SEA.index}
for q in (1, 3):
    clim, nyr = rect_clim(q)
    sea = SEA.reindex(clim.index).fillna(0.0)
    divs = pd.Series({r: rect_div.get(r, division(*rect_centre(r))) for r in clim.index})
    box_r = clim.loc[["44G0", "45G0"]].mean() * BOX_AREA
    box_s = box_S((H.Quarter == q) & H.region) * BOX_AREA
    for an, dl in AREAS.items():
        inarea = pd.Series([dl(divs[r], *rect_centre(r)) for r in clim.index], index=clim.index)
        idx = clim.loc[inarea].mul(sea[inarea], axis=0).sum()
        all_sea = SEA[[dl(rect_div.get(r), *rect_centre(r)) for r in SEA.index]].sum()
        cov = sea[inarea].sum() / all_sea
        for col in X.columns:
            if idx[col] <= 0:
                continue
            rows.append(dict(quarter=q, area=an, species=col[0], stage=col[1], box_R=box_r[col],
                             box_S=box_s[col], index=idx[col], share_R=box_r[col] / idx[col],
                             share_S=box_s[col] / idx[col], area_km2=sea[inarea].sum(), sea_cov=cov))
    # per-year shares (R estimator) over 4+3a and 3a20
    for y in range(YEARS[0], YEARS[1] + 1):
        sel = (H.Quarter == q) & (H.Year == y)
        if sel.sum() == 0:
            continue
        m = X.loc[sel].groupby(H.loc[sel, "rect"]).mean()
        if not {"44G0", "45G0"} & set(m.index):
            continue
        b = m.loc[[r for r in ("44G0", "45G0") if r in m.index]].mean() * BOX_AREA
        dv = pd.Series({r: rect_div.get(r, division(*rect_centre(r))) for r in m.index})
        s = SEA.reindex(m.index).fillna(0.0)
        for an in ("4+3a", "4+3a20", "3a (20+21)", "Viking>2E"):
            ia = pd.Series([AREAS[an](dv[r], *rect_centre(r)) for r in m.index], index=m.index)
            idx = m.loc[ia].mul(s[ia], axis=0).sum()
            for col in X.columns:
                if idx[col] > 0:
                    yrows.append(dict(quarter=q, year=y, area=an, species=col[0], stage=col[1],
                                      share=b[col] / idx[col]))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(out, "share_survey.csv"), index=False)
Y = pd.DataFrame(yrows)
YS = Y.groupby(["quarter", "area", "species", "stage"]).share.describe(percentiles=[0.1, 0.5, 0.9])
YS.to_csv(os.path.join(out, "share_survey_by_year_summary.csv"))
Y.to_csv(os.path.join(out, "share_survey_by_year.csv.gz"), index=False, compression="gzip")

# sea area per division (survey sea-area coverage)
dsea = pd.Series({d: SEA[[rect_div.get(r) == d for r in SEA.index]].sum() for d in
                  ["4a", "4b", "4c", "3a20", "3a21"]})
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
print("sea area by division (km2, ETOPO)\n", dsea.round(0).to_string())
print("box rect sea areas", SEA.reindex(["44G0", "45G0", "44G1", "45G1"]).round(0).to_dict())
print("hauls by quarter", H.groupby("Quarter").size().to_dict(), "box rect hauls",
      H[H.boxrect].groupby("Quarter").size().to_dict(), "region hauls", H[H.region].groupby("Quarter").size().to_dict())
print("box depth-bin area shares", barea.round(3).to_dict())
show = R[~R.stage.str.endswith("_kg")]
print("\nsurvey sea-area coverage by area\n", R.groupby("area")[["area_km2", "sea_cov"]].first().round(3).to_string())
print("\nclimatological shares (R = two box rectangles, S = depth-stratified)\n",
      show.pivot_table(index=["species", "stage", "quarter"], columns="area",
                       values=["share_R", "share_S"]).round(4).to_string())
print("\nper-year shares, 4+3a, R estimator: p10 / median / p90\n",
      YS.xs("4+3a", level="area")[["count", "10%", "50%", "90%"]].round(4).to_string())
