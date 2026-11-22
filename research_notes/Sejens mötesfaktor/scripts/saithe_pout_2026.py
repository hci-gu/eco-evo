"""Saithe on Norway pout, NS-IBTS Q1 2026 (February, R/V G.O. Sars): stomach content of each saithe against the
Norway pout caught IN THE SAME HAUL (the Rindorf & Gislason 2005 design, on saithe).

Field side [D on F]:
- Stomachs: ICES stomach DB download held in research_notes/"Födonivå och mättnad i fält"/data/A_ices_stomach/all
  (ICES data licence). Saithe (AphiaID 126441), 2026, individual fish (length 0.5-cm, IndWgt g), prey weights in mg.
  Norway pout = AphiaID 126444. Rows with StomachEmpty == 1 and no prey rows are empty stomachs (S = 0).
  Regurgitated fish (4) are dropped.
- Ration (gastric evacuation, WGSAM 2024 Table 3-7 cylinder model, saithe = whiting parameters, as in
  research_notes/"Fiskgrupper i rutan"/predators/scripts/calc_ger_cmax.py):
      R_tot (g/h) = rho e^(delta T) L^lambda E^-0.85 S_tot^0.5,  C_pout (g/d) = 24 R_tot S_pout / S_tot.
  E = pout energy density 4.6 kJ/g [J] (4.2-5.4), T = 7.0 C [J] (6-8.5; February bottom water at 80-170 m in the
  northern North Sea; BotTemp is not reported for these hauls).
- Pout density in the haul: DATRAS HL numbers at length x the haul's raise (SubFactor) x W(L) (0.0065 L^3.17 g),
  divided by the wing swept area (Distance x WingSpread; gear mean where missing) and by the Walker et al. 2017
  GOV q at length (0.066 <= 11 cm, 0.19 >= 12 cm; the convention behind the library's 1.64 t/km2 pout,
  research_notes/"Bottenfiskens bytesbalans"). Also printed at q = 1.

Fit [D]: per haul, the mean pout ration P_h (g/g/d) of the saithe sampled, against D_h (t/km2):
  (i) ratio estimator sum P / sum D (a lower bound on the low-density slope when the response saturates);
  (ii) Type II P = s D / (1 + s D / Pmax) by least squares on the haul means (s = low-density slope);
  (iii) haul bootstrap (2000 draws) for both.

Chain side [D]: library pre-296 a of predatory_fish_preys_on_planktivorous_fish (0.02348 km2/t/tick per t of the
WHOLE group, per t/km2 accessible planktivore) x saithe's share of the pair's a / its biomass share (0.92 / 0.80,
library comment and AGENTS.md) x pout access 0.95 (saithe's 19 cm gape, "Makrillens bytesstorlek" section 5) x the
pair's February daylight multiplier m (dark_ratio 0.14, -5 deg, 58.15 N) x the water column's q(T) at the field T
(ingestion_q10 2.25 normalised over its at_50m layer) x the pair's depth_access factor at each haul's bed depth
(normalised over the box's cells as the engine does). Factor = field slope / chain slope, per day.

Usage: python3 -I saithe_pout_2026.py <repo root>
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(sys.argv[1])
NOTE = os.path.join(ROOT, "research_notes", "Sejens mötesfaktor")
DB = os.path.join(ROOT, "research_notes", "Födonivå och mättnad i fält", "data", "A_ices_stomach", "all")
DAT = os.path.join(NOTE, "data", "datras")
RHO, DELTA, LAM = 0.00171, 0.081, 1.35          # saithe = whiting GER parameters (WGSAM 2024)
E_POUT = (4.2, 4.6, 5.4)
T_FIELD = (6.0, 7.0, 8.5)
LW_POUT = (0.0065, 3.17)
A_PRE296 = 0.02348
SAITHE_SCALE = 0.92 / 0.80
POUT_ACCESS = 0.95
rng = np.random.default_rng(1)

# ---------------------------------------------------------------- stomachs
pr = pd.read_csv(os.path.join(DB, "PredatorInformation.csv"), dtype=str, on_bad_lines="skip")
pr = pr[(pr.AphiaIDPredator == "126441") & (pr.Year == "2026")].copy()
py = pd.read_csv(os.path.join(DB, "PreyInformation.csv"), dtype=str, on_bad_lines="skip")
py = py[(py.AphiaIDPredator == "126441") & (py.Year == "2026")].copy()
hi = pd.read_csv(os.path.join(DB, "HaulInformation.csv"), dtype=str, on_bad_lines="skip")
py["g"] = pd.to_numeric(py.Weight, errors="coerce") / np.where(py.UnitWgt == "mg", 1000.0, 1.0)
tot = py.groupby("tblPredatorInformationID").g.sum()
pout = py[py.AphiaIDPrey == "126444"].groupby("tblPredatorInformationID").g.sum()
fishp = py[py.AphiaIDPrey.isin(["126444", "293496", "125469", "127023", "127312", "126417", "126437",
                                 "126426", "127419"])].groupby("tblPredatorInformationID").g.sum()
pout_dig = py[(py.AphiaIDPrey == "126444") & (py.DigestionStage != "0")].groupby("tblPredatorInformationID").g.sum()
print("pout prey mass by DigestionStage (g):", py[py.AphiaIDPrey == "126444"].groupby("DigestionStage").g.sum().round(1).to_dict(),
      "; all other prey:", py[py.AphiaIDPrey != "126444"].groupby("DigestionStage").g.sum().round(1).to_dict())
pr["S_tot"] = pr.tblPredatorInformationID.map(tot).fillna(0.0)
pr["S_pout_dig"] = pr.tblPredatorInformationID.map(pout_dig).fillna(0.0)
pr["S_pout"] = pr.tblPredatorInformationID.map(pout).fillna(0.0)
pr["S_fish"] = pr.tblPredatorInformationID.map(fishp).fillna(0.0)
pr["L"] = pd.to_numeric(pr.Length, errors="coerce")
pr["W"] = pd.to_numeric(pr.IndWgt, errors="coerce")
n0 = len(pr)
pr = pr[pr.Regurgitated.fillna("0") != "1"]
pr = pr.drop(columns=["Ship", "Gear", "HaulNo", "StationNumber", "Day", "Time"]).merge(hi[["tblHaulID", "Ship", "Gear", "HaulNo", "StationNumber", "Day", "Time", "ShootLat", "ShootLong",
                  "ICESrectangle", "Depth"]], on="tblHaulID", how="left")
print(f"saithe 2026 rows {n0}, after dropping regurgitated {len(pr)}; hauls {pr.tblHaulID.nunique()}")
print(f"  length {pr.L.min():.1f}-{pr.L.max():.1f} cm (median {pr.L.median():.1f}); weight median {pr.W.median():.0f} g")
print(f"  empty (S_tot 0) {int((pr.S_tot == 0).sum())}; with pout {int((pr.S_pout > 0).sum())}; "
      f"pout share of all prey mass {pr.S_pout.sum() / pr.S_tot.sum():.3f}; of fish prey {pr.S_pout.sum() / pr.S_fish.sum():.3f}")
print(f"  'Number' field values {pr.Number.value_counts().to_dict()} (read as a code; rows are single fish:"
      f" individual 0.5-cm lengths and weights) [J]")


def ration(row, T, E, col="S_pout"):
    if row.S_tot <= 0:
        return 0.0
    R = RHO * math.exp(DELTA * T) * row.L ** LAM * E ** -0.85 * row.S_tot ** 0.5
    return 24.0 * R * row[col] / row.S_tot / row.W


def cmax(L, W, T, E=4.184, smax=0.025):
    return 24.0 * RHO * math.exp(DELTA * T) * L ** LAM * E ** -0.85 * (smax * W) ** 0.5 / W


# ---------------------------------------------------------------- pout density in the same haul
hh = pd.read_csv(os.path.join(DAT, "HH_NS-IBTS_2026_Q1.csv"))
hl = pd.read_csv(os.path.join(DAT, "HL_NS-IBTS_2026_Q1.csv"))
hh = hh[hh.Ship == "58G2"].copy()
wmean = hh[hh.WingSpread > 0].groupby("Gear").WingSpread.mean().to_dict()
hh["wing"] = np.where(hh.WingSpread > 0, hh.WingSpread, hh.Gear.map(wmean))
hh["area_km2"] = hh.Distance * hh.wing / 1e6
lp = hl[(hl.Ship == "58G2") & (hl.SpecCode == 126444)].copy()
lp["n"] = lp.HLNoAtLngt * lp.SubFactor.where(lp.SubFactor > 0, 1.0)
lp["L"] = lp.LngtClass / np.where(lp.LngtCode.astype(str) == ".", 10.0, 1.0)  # code 1 = cm
lp["w_lw"] = lp.n * LW_POUT[0] * (lp.L + 0.5) ** LW_POUT[1]
# length-weight gives the shape only; each catch category's weight is scaled to its recorded CatCatchWgt
# (the LW overstates the measured mean weight ~1.6x in these hauls)
cat_sum = lp.groupby(["HaulNo", "CatIdentifier"]).w_lw.transform("sum")
lp["w_t"] = lp.w_lw / cat_sum * lp.CatCatchWgt / 1e6
lp["q"] = np.where(lp.L <= 11, 0.066204, 0.191855)
lp["w_tq"] = lp.w_t / lp.q
dens = lp.groupby("HaulNo").agg(t_raw=("w_t", "sum"), t_q=("w_tq", "sum"), n=("n", "sum"))
dens["catch_g"] = lp.groupby(["HaulNo", "CatIdentifier"]).CatCatchWgt.first().groupby(level=0).sum()
print("pout LW-implied / recorded catch weight, median over hauls: "
      f"{float((lp.groupby('HaulNo').w_lw.sum() / dens.catch_g).median()):.2f}")
hh = hh.set_index("HaulNo").join(dens)
hh[["t_raw", "t_q", "catch_g", "n"]] = hh[["t_raw", "t_q", "catch_g", "n"]].fillna(0.0)
hh["D_q1"] = hh.t_raw / hh.area_km2
hh["D"] = hh.t_q / hh.area_km2
print(f"\nG.O. Sars Q1 2026 hauls {len(hh)}; swept area median {hh.area_km2.median():.4f} km2 "
      f"(wing {hh.wing.median():.1f} m)")

# ---------------------------------------------------------------- chain
lib = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml")))
man = yaml.safe_load(open(os.path.join(ROOT, "mareld2.yaml")))
sys.path.insert(0, ROOT)
from lib.world import daylight  # noqa: E402
from lib.environments.ecosystem_env import open_pool  # noqa: E402
P = lib["interaction_definitions"]["predatory_fish_preys_on_planktivorous_fish"]
lat = float(man["simulation_settings"]["daylight"]["latitude_deg"])
m = daylight.attack_multiplier(lat, 6, float(P["dark_ratio"]), float(P.get("light_threshold_deg", -5.0)))
m_day = m.reshape(365, 4).mean(axis=1)
m_feb = float(m_day[31:59].mean())
q10 = float(lib["species_definitions"]["predatory_fish"]["ingestion_q10"])
Tl = np.array(man["simulation_settings"]["temperature"]["layers"]["at_50m"], float)
qnorm = float(np.mean(q10 ** (Tl / 10.0)))
depth = np.load(os.path.join(ROOT, "maps", "depth.npz"))["depth"].astype(float)
dtab = open_pool._parse_rows(P["depth_access_table"], "factor", "pair")
raw_box = np.interp(depth, dtab[0], dtab[1], left=dtab[1][0], right=dtab[1][-1])
dnorm = float(raw_box.mean())


def dfac(d):
    return float(np.interp(d, dtab[0], dtab[1], left=dtab[1][0], right=dtab[1][-1])) / dnorm


print(f"\nchain: a_pre296 {A_PRE296} x saithe scale {SAITHE_SCALE:.3f} x pout access {POUT_ACCESS} x m_Feb {m_feb:.3f}"
      f" x q(T) {q10 ** (T_FIELD[1] / 10) / qnorm:.3f} at {T_FIELD[1]} C; depth factor <100 {dfac(90):.2f}, "
      f"100-150 {dfac(125):.2f}, 150-250 {dfac(170):.2f} (box-normalised)")

# ---------------------------------------------------------------- per haul
res = {}
for T in T_FIELD:
    for E in E_POUT:
        pr[f"C_{T}_{E}"] = pr.apply(lambda r: ration(r, T, E), axis=1)
key = f"C_{T_FIELD[1]}_{E_POUT[1]}"
# net-feeding bound: only pout past DigestionStage 0 (0 read as fresh/undigested [J]); the stomach's other
# contents keep their own evacuation (S_tot unchanged)
# consistency: summed prey weight vs the recorded FullStomWgt (stomach incl. wall); where prey exceed it, the cap
# scales S_tot and S_pout down to FullStomWgt (prey in the gullet or mouth, or a recording error) [J]
pr["F_stom"] = pd.to_numeric(pr.FullStomWgt, errors="coerce")
fed = pr[pr.S_tot > 0]
print(f"summed prey / FullStomWgt per fed fish: median {(fed.S_tot / fed.F_stom).median():.2f}; fish with prey > "
      f"FullStomWgt {int((fed.S_tot > fed.F_stom).sum())} of {len(fed)}; of the fish > 5 %BW pout "
      f"{int(((fed.S_tot > fed.F_stom) & (100 * fed.S_pout / fed.W > 5)).sum())} of {int((100 * fed.S_pout / fed.W > 5).sum())}")
capf = np.where((pr.S_tot > pr.F_stom) & (pr.F_stom > 0), pr.F_stom / pr.S_tot, 1.0)
pr["S_tot_c"], pr["S_pout_c"] = pr.S_tot * capf, pr.S_pout * capf


def ration_c(row, T, E):
    if row.S_tot_c <= 0:
        return 0.0
    R = RHO * math.exp(DELTA * T) * row.L ** LAM * E ** -0.85 * row.S_tot_c ** 0.5
    return 24.0 * R * row.S_pout_c / row.S_tot_c / row.W


pr["C_cap"] = pr.apply(lambda r: ration_c(r, T_FIELD[1], E_POUT[1]), axis=1)
pr["C_dig"] = pr.apply(lambda r: ration(r, T_FIELD[1], E_POUT[1], "S_pout_dig"), axis=1)
g = pr.groupby("HaulNo")
H = pd.DataFrame({"n_fish": g.size(), "L": g.L.mean(), "P": g[key].mean(),
                  "P_lo": g[f"C_{T_FIELD[0]}_{E_POUT[2]}"].mean(), "P_dig": g.C_dig.mean(), "P_cap": g.C_cap.mean(), "P_hi": g[f"C_{T_FIELD[2]}_{E_POUT[0]}"].mean(),
                  "S_pout_pct": g.apply(lambda d: 100 * (d.S_pout / d.W).mean()),
                  "cmax": g.apply(lambda d: np.mean([cmax(L, W, T_FIELD[1]) for L, W in zip(d.L, d.W)]))})
H.index = H.index.astype(int)
H = H.join(hh[["Gear", "Depth", "DayNight", "D", "D_q1", "catch_g", "area_km2"]], how="left")
H["chain_day"] = [4.0 * A_PRE296 * SAITHE_SCALE * POUT_ACCESS * m_feb * q10 ** (T_FIELD[1] / 10) / qnorm * dfac(d)
                  for d in H.Depth]
H = H.sort_values("D")
print("\nper haul (sorted by pout density; P = pout ration g/g/d at T 7, E 4.6; D t/km2 Walker-q; chain per day)")
print(H[["Gear", "Depth", "n_fish", "L", "S_pout_pct", "P", "cmax", "catch_g", "D_q1", "D", "chain_day"]]
      .to_string(float_format=lambda x: f"{x:.4g}"))
missing = H.D.isna().sum()
H = H.dropna(subset=["D"])
print(f"hauls without a DATRAS match: {missing}")


def fit(Hs, col="P"):
    P_, D_, c_ = Hs[col].values, Hs.D.values, Hs.chain_day.values
    ratio = P_.sum() / D_.sum()
    ratio_f = P_.sum() / (c_ * D_).sum()
    best = (np.inf, None, None)
    for ls in np.linspace(math.log(1e-5), math.log(10.0), 160):
        s = math.exp(ls)
        for lp in np.linspace(math.log(1e-4), math.log(0.2), 120):
            pm = math.exp(lp)
            pred = s * D_ / (1 + s * D_ / pm)
            sse = float(((P_ - pred) ** 2).sum())
            if sse < best[0]:
                best = (sse, s, pm)
    _, s, pm = best
    # factor in the same Type II form with the haul's own chain slope
    bestf = (np.inf, None)
    for lf in np.linspace(math.log(1e-4), math.log(100.0), 300):
        f = math.exp(lf)
        pred = f * c_ * D_ / (1 + f * c_ * D_ / pm)
        sse = float(((P_ - pred) ** 2).sum())
        if sse < bestf[0]:
            bestf = (sse, f)
    return ratio, ratio_f, s, pm, bestf[1]


r, rf, s, pm, f2 = fit(H)
cbar = float(np.average(H.chain_day, weights=H.D))
print(f"\nfit, haul means (n {len(H)}):")
print(f"  ratio estimator sum P / sum D = {r:.4g} g/g/d per t/km2 -> factor {rf:.3g} (vs each haul's chain)")
print(f"  Type II: low-density slope s {s:.4g} per t/km2/d, plateau Pmax {pm:.4g} g/g/d "
      f"({pm / H.cmax.mean():.2f} of GER Cmax); factor s / density-weighted chain {s / cbar:.3g}; "
      f"factor fitted on each haul's chain {f2:.3g}")
boots = []
idx = np.arange(len(H))
for _ in range(400):
    b = H.iloc[rng.choice(idx, len(idx))]
    if b.D.sum() <= 0:
        continue
    rb, rfb, sb, pmb, f2b = fit(b)
    boots.append((rfb, f2b, sb, pmb))
B = np.array(boots)
for j, lab in enumerate(["ratio factor", "Type II factor (haul chain)", "Type II slope", "Pmax"]):
    print(f"  bootstrap {lab}: median {np.median(B[:, j]):.3g}, 5-95 % {np.percentile(B[:, j], 5):.3g}-"
          f"{np.percentile(B[:, j], 95):.3g}")
for lab, col in (("T 6 / E 5.4 (low ration)", "P_lo"), ("T 8.5 / E 4.2 (high ration)", "P_hi"),
                 ("pout past DigestionStage 0 only (net-feeding bound)", "P_dig"),
                 ("stomach content capped at FullStomWgt", "P_cap")):
    r2, rf2, s2, pm2, f22 = fit(H, col)
    print(f"  {lab}: ratio factor {rf2:.3g}, Type II factor {f22:.3g}")
Hq1 = H.copy()
Hq1["D"] = Hq1.D_q1
r3, rf3, s3, pm3, f23 = fit(Hq1)
print(f"  at GOV q = 1 (density x{(H.D.sum() / H.D_q1.sum()):.2f} lower): ratio factor {rf3:.3g}, Type II factor {f23:.3g}"
      " (not the library's convention)")
lowD = H[H.D <= H.D.median()]
r4, rf4, *_ = fit(lowD)
print(f"  low-density half (D <= {H.D.median():.3g}, n {len(lowD)}): ratio factor {rf4:.3g}")
print(f"\ncorr(P, D) Spearman {H[['P', 'D']].corr(method='spearman').iloc[0, 1]:.2f}; "
      f"hauls with pout in stomachs {int((H.P > 0).sum())}, with pout in the catch {int((H.D > 0).sum())}")
H.to_csv(os.path.join(NOTE, "tables", "saithe_pout_2026_hauls.csv"))
