"""Otolith FMR of the benthivore members (Alewijnse 2022 dataset; Jones et al. 2023 plaice).

1. Barents Sea site (72.797 N, 23.288 E; SUMIE; outer-edge samples; 2018-2019): cod, haddock and
   long rough dab sampled at one place with one diet/DIC baseline per species. Fit
       ln VO2 = a_species + b ln(mass) + c T
   on fish >= 100 g (adults) and report species offsets relative to cod: this ratio does not depend
   on the absolute calibration as long as one calibration holds for all species (Jones 2023 argue it does).
2. Absolute FMR at the model's sizes and 7.5 C, with the fitted c and with imposed Q10 2 and 3.
3. Multipliers FMR/SMR per species with stated SMR denominators (prior notes' SMR values).
VO2 from C_resp: Chung et al. 2019, C_resp = 0.243 (1 - exp(-0.0088 VO2)), VO2 mg O2/kg/h.
1 mg O2/kg/h = 13.6 J/mg x 24 h = 0.3264 kJ/kg/d = 119.1 MJ/t/yr.
Usage: python3 -I otolith_fmr.py <alewijnse_master_data.csv>
"""
import csv, math, sys
import numpy as np

C, K = 0.243, 8.80e-3
MJ_PER_MGKGH = 13.6e-3 * 24 * 365  # kJ/kg/yr per mg/kg/h = MJ/t/yr
TICK = 13.6e-3 * 6  # MJ/t per 6 h tick per mg/kg/h

def vo2(c):
    return -math.log(1 - c / C) / K

rows = [r for r in csv.DictReader(open(sys.argv[1]))]
site = [r for r in rows if r["source"] == "SUMIE" and r["lat"] not in ("", "NA") and abs(float(r["lat"]) - 72.797) < 0.01
        and r["sample_type"] == "outer"]
SP = {"Gadus_morhua": "cod", "Melanogrammus_aeglefinus": "haddock", "Hippoglossoides_platessoides": "LRD"}
print(f"conversion: 1 mg O2/kg/h = {MJ_PER_MGKGH:.1f} MJ/t/yr = {TICK:.4f} MJ/t/tick\n")

def fit(data, label, fixed_c=None):
    names = ["cod", "haddock", "LRD"]
    X, y = [], []
    for sp, m, t, v in data:
        row = [1.0 if sp == n else 0.0 for n in names] + [math.log(m)]
        if fixed_c is None:
            row.append(t)
        X.append(row)
        y.append(math.log(v) - (fixed_c * t if fixed_c is not None else 0.0))
    X, y = np.array(X), np.array(y)
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    print(f"--- {label}: n = {len(y)}, residual SD (ln) = {math.sqrt(s2):.3f}")
    for i, n in enumerate(names):
        print(f"  a_{n} = {beta[i]:.3f} +- {se[i]:.3f}")
    print(f"  b (ln mass) = {beta[3]:.3f} +- {se[3]:.3f}")
    c = fixed_c if fixed_c is not None else beta[4]
    if fixed_c is None:
        print(f"  c (per C) = {beta[4]:.3f} +- {se[4]:.3f}  -> Q10 {math.exp(10*beta[4]):.2f}")
    else:
        print(f"  c fixed = {c:.4f} (Q10 {math.exp(10*c):.2f})")
    for j, n in enumerate(["haddock", "LRD"], start=1):
        d = beta[j] - beta[0]
        v = cov[j, j] + cov[0, 0] - 2 * cov[0, j]
        lo, hi = math.exp(d - 1.96 * math.sqrt(v)), math.exp(d + 1.96 * math.sqrt(v))
        print(f"  {n}/cod FMR ratio at equal mass and T = {math.exp(d):.2f} (95 % CI {lo:.2f}-{hi:.2f})")
    return beta, c

data_all, data_ad = [], []
summary = {}
for r in site:
    sp = SP.get(r["species"])
    if not sp:
        continue
    c = float(r["c_resp_mean"])
    if c >= C:
        continue
    m, t = float(r["body_mass"]), float(r["temp_mean"])
    rec = (sp, m, t, vo2(c))
    data_all.append(rec)
    if m >= 100:
        data_ad.append(rec)
for sp in ["cod", "haddock", "LRD"]:
    for lab, d in [("all", data_all), (">=100 g", data_ad)]:
        s = [x for x in d if x[0] == sp]
        if s:
            print(f"{sp:8s} {lab:8s} n {len(s):3d}  mass {min(x[1] for x in s):.0f}-{max(x[1] for x in s):.0f} g "
                  f"(median {np.median([x[1] for x in s]):.0f})  T {np.mean([x[2] for x in s]):.1f} "
                  f"({min(x[2] for x in s):.1f}-{max(x[2] for x in s):.1f}) C  VO2 mean {np.mean([x[3] for x in s]):.0f} "
                  f"median {np.median([x[3] for x in s]):.0f} mg O2/kg/h")
print()
results = {}
results["fitted"] = fit(data_ad, "adults >= 100 g, T coefficient fitted")
results["q10_2"] = fit(data_ad, "adults >= 100 g, Q10 2 imposed", fixed_c=math.log(2) / 10)
results["q10_3"] = fit(data_ad, "adults >= 100 g, Q10 3 imposed", fixed_c=math.log(3) / 10)
results["all_fitted"] = fit(data_all, "all sizes incl. juveniles, T fitted")
print()

# Absolute FMR at the model's sizes and 7.5 C (extrapolation from 0.4-7.8 C experienced T)
sizes = {"cod": 1200.0, "haddock": 275.0, "LRD": 170.0}
idx = {"cod": 0, "haddock": 1, "LRD": 2}
print("Predicted FMR_oto at the model's effective mass and 7.5 C (extrapolated):")
for key, (beta, c) in results.items():
    if key == "all_fitted":
        continue
    out = []
    for sp, m in sizes.items():
        v = math.exp(beta[idx[sp]] + beta[3] * math.log(m) + c * 7.5)
        out.append(f"{sp} {m:.0f} g {v:.0f} mg/kg/h = {v*MJ_PER_MGKGH:.0f} MJ/t/yr = {v*TICK:.2f} MJ/t/tick")
    print(f"  [{key}] " + "; ".join(out))
print()

# Plaice (Jones et al. 2023, Table 3), witch, dab, lemon sole (Alewijnse, Cefas outer samples)
print("Other flatfish (no within-site cod comparator):")
print(f"  plaice, Jones 2023 Table 3: mean 54 mg/kg/h at mean 8 C, 200-800 g -> {54*MJ_PER_MGKGH:.0f} MJ/t/yr; "
      f"per-fish SD 36")
for spn in ["Glyptocephalus_cynoglossus", "Limanda_limanda", "Microstomus_kitt"]:
    s = [r for r in rows if r["species"] == spn]
    v = [vo2(float(r["c_resp_mean"])) for r in s]
    t = [float(r["temp_mean"]) for r in s]
    m = [float(r["body_mass"]) for r in s]
    print(f"  {spn}: n {len(s)}, VO2 mean {np.mean(v):.0f} (range {min(v):.0f}-{max(v):.0f}) mg/kg/h at "
          f"T {np.mean(t):.1f} C, mass {min(m):.0f}-{max(m):.0f} g -> {np.mean(v)*MJ_PER_MGKGH:.0f} MJ/t/yr at that T")
print()

# Multipliers against SMR denominators (mg O2/kg/h)
def q(v, t0, t1, q10):
    return v * q10 ** ((t1 - t0) / 10)
print("SMR denominators and multipliers:")
# cod: Schurmann & Steffensen 1997 35.5 at 5 C, 57.0 at 10 C (juveniles; mass not in abstract)
smr_cod_75 = 35.5 * (57.0 / 35.5) ** 0.5
print(f"  cod SMR at 7.5 C (Schurmann & Steffensen 1997, interpolated): {smr_cod_75:.1f}; prior-note group table 3.53 MJ/t/tick "
      f"at 7 C = {3.53/TICK:.1f} mg/kg/h (1.2 kg)")
smr = {"cod": 3.53 / TICK, "haddock": 2.50 / TICK, "LRD": 1.50 / TICK}
for key in ["fitted", "q10_2", "q10_3"]:
    beta, c = results[key]
    s = []
    for sp, m in sizes.items():
        v = math.exp(beta[idx[sp]] + beta[3] * math.log(m) + c * 7.5)
        s.append(f"{sp} {v/smr[sp]:.2f}")
    print(f"  [{key}] FMR_oto / prior-note SMR (cod 3.53, haddock 2.50, LRD 1.50 MJ/t/tick at ~7 C): " + ", ".join(s))
# Multipliers at the model masses need SMR ratios at the model masses. Equal-mass SMR ratios
# (Norin 63-68 g; Perez-Casanova 40-50 g) are moved to 275 g vs 1200 g with SMR ~ W^-0.2 (x1.34).
print("  haddock multiplier at 275 g = cod multiplier 2.2 x FMR ratio(275 g / 1200 g) / SMR ratio(275 g / 1200 g):")
mass_term = (275 / 1200) ** -0.2
for key in ["fitted", "q10_2", "q10_3"]:
    beta, c = results[key]
    fr = math.exp(beta[1] - beta[0]) * (275 / 1200) ** beta[3]
    for lab, sr in [("Norin 0.54 x1.34", 0.54 * mass_term), ("Perez-Casanova 0.64 x1.34", 0.64 * mass_term),
                    ("Perez-Casanova 0.75 x1.34", 0.75 * mass_term), ("prior table 2.50/3.53", 2.50 / 3.53)]:
        print(f"    [{key}] FMR ratio {fr:.2f}; SMR ratio {lab} = {sr:.2f}: haddock multiplier {2.2*fr/sr:.2f}")

# Diet-baseline sensitivity: recompute C_resp = (oto - DIC)/(diet - DIC) with one common diet value
print()
print("Diet-baseline sensitivity (C_resp recomputed by two-source mixing; check against the Bayesian value):")
diet_sp = {}
for r in site:
    if r["species"] in SP:
        diet_sp[SP[r["species"]]] = float(r["diet_d13c_mean"])
print("  species diet d13C used by Alewijnse at the site:", {k: round(v, 2) for k, v in diet_sp.items()})
for common_lab, common in [("own species values", None)] + [(f"common = {k} {v:.2f}", v) for k, v in diet_sp.items()]:
    dd = []
    for r in site:
        sp = SP.get(r["species"])
        if not sp:
            continue
        m, t = float(r["body_mass"]), float(r["temp_mean"])
        if m < 100:
            continue
        oto, dic = float(r["oto_d13c_mean"]), float(r["dic_d13c_mean"])
        diet = float(r["diet_d13c_mean"]) if common is None else common
        cr = (oto - dic) / (diet - dic)
        if cr <= 0 or cr >= C:
            continue
        dd.append((sp, m, t, vo2(cr)))
    fit(dd, f"adults, Q10 2 imposed, diet {common_lab}", fixed_c=math.log(2) / 10)
print()
plaice_smr = {"Fonds 400 g": 29.1 * 2.01 ** 0.1, "Fonds 150 g": 36.1 * 2.01 ** 0.1}
for k, v in plaice_smr.items():
    print(f"  plaice: 54 / SMR {k} at 8 C ({v:.1f}) = {54/v:.2f}")
witch_smr = {"Moser 1996 400 g at 8.4 C (Q10 2.2)": 11.1 * 2.2 ** 0.24,
             "Moser 1996 150 g at 8.4 C": 14.9 * 2.2 ** 0.24, "plaice Fonds 400 g at 8.4 C": 32.1}
for k, v in witch_smr.items():
    print(f"  witch: 35 / SMR {k} ({v:.1f}) = {35/v:.2f}")
dab_smr = 44.6 * 2.01 ** ((17.8 - 10) / 10)
print(f"  dab: 176 / plaice SMR 150 g at 17.8 C ({dab_smr:.0f}) = {176/dab_smr:.2f} (summer, English Channel; a bound only)")
ls_smr = 44.6 * 2.01 ** ((14.6 - 10) / 10)
print(f"  lemon sole: 114 / plaice SMR 150 g at 14.6 C ({ls_smr:.0f}) = {114/ls_smr:.2f}")
