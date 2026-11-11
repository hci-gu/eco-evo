"""Otolith field metabolism of blue whiting (Alewijnse 2022 dataset), and why sprat/anchovy give none.

1. Barents site (72.797 N, 23.288 E; SUMIE; outer-edge samples): blue whiting sampled at the same site,
   with the same seawater-DIC baseline, as the cod and haddock of the sibling note
   (research_notes/"Faltmetabolism kolja och plattfisk", a-umlaut, scripts/otolith_fmr.py). Fit
       ln VO2 = a_species + b ln(mass) + c T
   on cod + blue whiting (+ haddock) >= 100 g and report blue whiting / cod at equal mass and T. The ratio
   does not depend on the absolute calibration if one calibration holds for both species.
2. Absolute FMR: (a) cod-anchored: the sibling's cod value 11 338 MJ/t/yr at 1.2 kg and 7.5 C
   (= 2.2 x SMR, Chung 2019) x the ratio, moved to blue whiting's box size with the fitted b;
   (b) face value of the Chung calibration at each site.
3. Sprat and anchovy: C_resp above the calibration's asymptote 0.243 -> no VO2 (listed only).
VO2 from C_resp: Chung et al. 2019, C_resp = 0.243 (1 - exp(-0.0088 VO2)), VO2 mg O2/kg/h.
1 mg O2/kg/h = 13.6 J/mg x 24 h x 365 = 119.1 MJ/t/yr.
Usage: python3 -I otolith_bw.py <alewijnse_master_data.csv>
"""
import csv, math, sys
import numpy as np

C, K = 0.243, 8.80e-3
MJ = 13.6e-3 * 24 * 365


def vo2(c):
    return -math.log(1 - c / C) / K


rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8", errors="replace")))
site = [r for r in rows if r["source"] == "SUMIE" and r["lat"] not in ("", "NA")
        and abs(float(r["lat"]) - 72.797) < 0.01 and r["sample_type"] == "outer"]
SP = {"Gadus_morhua": "cod", "Micromesistius_poutassou": "bw", "Melanogrammus_aeglefinus": "haddock"}

data = []
for r in site:
    sp = SP.get(r["species"])
    if not sp:
        continue
    c = float(r["c_resp_mean"])
    if c >= C:
        continue
    data.append((sp, float(r["body_mass"]), float(r["temp_mean"]), vo2(c), c))

print("Barents site (72.797 N), outer otolith edge, per species:")
for sp in ("cod", "haddock", "bw"):
    s = [x for x in data if x[0] == sp]
    s100 = [x for x in s if x[1] >= 100]
    print(f"  {sp:8s} n {len(s):3d} (>=100 g: {len(s100):3d}); mass {min(x[1] for x in s):.0f}-{max(x[1] for x in s):.0f} g; "
          f"T {min(x[2] for x in s):.1f}-{max(x[2] for x in s):.1f} C (mean {np.mean([x[2] for x in s]):.1f}); "
          f"C_resp {np.mean([x[4] for x in s]):.3f}; VO2 mean {np.mean([x[3] for x in s]):.0f} mg O2/kg/h")


def fit(d, names, label, fixed_c=None):
    X, y = [], []
    for sp, m, t, v, _ in d:
        row = [1.0 if sp == n else 0.0 for n in names] + [math.log(m)]
        if fixed_c is None:
            row.append(t)
        X.append(row)
        y.append(math.log(v) - (fixed_c * t if fixed_c is not None else 0.0))
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    c = fixed_c if fixed_c is not None else beta[len(names) + 1]
    j = names.index("bw")
    d_ = beta[j] - beta[0]
    v_ = cov[j, j] + cov[0, 0] - 2 * cov[0, j]
    lo, hi = math.exp(d_ - 1.96 * math.sqrt(v_)), math.exp(d_ + 1.96 * math.sqrt(v_))
    print(f"--- {label}: n {len(y)}, resid SD {math.sqrt(s2):.3f}, b {beta[len(names)]:.3f}, "
          f"c {c:.3f} (Q10 {math.exp(10 * c):.2f}); bw/cod at equal mass and T = {math.exp(d_):.2f} (95 % CI {lo:.2f}-{hi:.2f})")
    return beta[len(names)], c, math.exp(d_), lo, hi


ad = [x for x in data if x[1] >= 100]
res = {}
res["fit_cb"] = fit([x for x in ad if x[0] in ("cod", "bw")], ["cod", "bw"], "cod + bw >= 100 g, T fitted")
res["fit_cbh"] = fit(ad, ["cod", "bw", "haddock"], "cod + bw + haddock >= 100 g, T fitted")
res["q10_2"] = fit([x for x in ad if x[0] in ("cod", "bw")], ["cod", "bw"], "cod + bw >= 100 g, Q10 2 imposed",
                   fixed_c=math.log(2) / 10)
res["q10_3"] = fit([x for x in ad if x[0] in ("cod", "bw")], ["cod", "bw"], "cod + bw >= 100 g, Q10 3 imposed",
                   fixed_c=math.log(3) / 10)
print()

# Cod-anchored absolute value at the box's blue whiting (Q3 visitor, 20-30 cm; 50 m and bottom ~7.5-10 C).
COD_ANCHOR = 11338.0  # MJ/t/yr, cod 1.2 kg, 7.5 C (sibling note, Chung 2019 2.2 x SMR)
print("Cod-anchored blue whiting FMR at 7.5 C (the sibling's cod 11 338 MJ/t/yr at 1.2 kg x ratio x (m/1200)^b):")
for key, (b, c, ratio, lo, hi) in res.items():
    out = []
    for m in (60.0, 100.0, 150.0):
        v = COD_ANCHOR * ratio * (m / 1200.0) ** b
        out.append(f"{m:.0f} g {v:.0f} ({COD_ANCHOR * lo * (m / 1200.0) ** b:.0f}-{COD_ANCHOR * hi * (m / 1200.0) ** b:.0f})")
    print(f"  [{key}] b {b:.3f}: " + "; ".join(out) + " MJ/t/yr")
print()

print("Face value of the Chung calibration, blue whiting by site (all rows; C_resp < 0.243):")
for lat in sorted({r["lat"] for r in rows if r["species"] == "Micromesistius_poutassou"}):
    s = [r for r in rows if r["species"] == "Micromesistius_poutassou" and r["lat"] == lat]
    v = [vo2(float(r["c_resp_mean"])) for r in s]
    t = [float(r["temp_mean"]) for r in s]
    m = [float(r["body_mass"]) for r in s]
    print(f"  lat {float(lat):.2f}: n {len(s)}, mass {min(m):.0f}-{max(m):.0f} g, T {np.mean(t):.1f} C "
          f"({min(t):.1f}-{max(t):.1f}), VO2 {np.mean(v):.0f} ({min(v):.0f}-{max(v):.0f}) mg/kg/h = {np.mean(v) * MJ:.0f} MJ/t/yr")
print()

print("Small pelagics with C_resp at or above the calibration asymptote 0.243 (no VO2 obtainable):")
for spn in ("Sprattus_sprattus", "Engraulis_encrasicolus"):
    s = [r for r in rows if r["species"] == spn]
    cs = [float(r["c_resp_mean"]) for r in s]
    print(f"  {spn}: n {len(s)}, C_resp {min(cs):.3f}-{max(cs):.3f}, T {min(float(r['temp_mean']) for r in s):.1f}-"
          f"{max(float(r['temp_mean']) for r in s):.1f} C, mass {min(float(r['body_mass']) for r in s):.0f}-"
          f"{max(float(r['body_mass']) for r in s):.0f} g, sample {set(r['sample_type'] for r in s)}, source {set(r['source'] for r in s)}")
print("  Clupea harengus, Trisopterus esmarkii: not in the dataset.")
present = sorted({r["species"] for r in rows})
print(f"  species in the dataset: {len(present)}; clupeoids/planktivore members present: "
      f"{[s for s in present if s.split('_')[0] in ('Clupea', 'Sprattus', 'Trisopterus', 'Micromesistius', 'Engraulis', 'Sardina')]}")
