"""Biomass-weighted species fields and the derived pair fields of predatory_fish.

Usage: python3 -I weighted_fields.py <predators_dir>

Reads inputs/species_values.csv (per species: central, low, high; sources in the evidence files)
and tables/pred_season.csv (long-run 2001-25 depth-stratified GOV shares, pooled Q1+Q3).
Ranges of weighted means are the weighted lows and highs (all species at their bound at once),
i.e. corner bounds, not confidence intervals. Writes tables/weighted_fields.csv and prints the
derived quantities: per-tick conversions, the energy ceiling and h per prey, the attack rate
anchor, M2 on each prey pool, K and the trawl tables.
"""
import os
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
V = pd.read_csv(os.path.join(d, "inputs", "species_values.csv")).set_index("species")
S = pd.read_csv(os.path.join(d, "tables", "pred_season.csv")).set_index("common")
w = S.loc[V.index, "pooled"]
w = w / w.sum()
TICKS = 1460.0
rows = []


def wm(col):
    return float((w * V[col]).sum())


def put(field, c, lo, hi, note=""):
    rows.append(dict(field=field, central=c, low=lo, high=hi, note=note))
    print(f"{field:38s} {c:12.6g} ({lo:.6g} - {hi:.6g}) {note}")


print("weights:", w.round(4).to_dict())
r, rlo, rhi = wm("r_now"), wm("r_lo"), wm("r_hi")
put("r_max (/yr)", r, rlo, rhi)
Fb, Fp = wm("F_bot"), wm("F_pel")
F = Fb + Fp
# saithe F range 0.11-0.38 carries the group range
Flo = F - w["saithe"] * (0.30 - 0.11)
Fhi = F + w["saithe"] * (0.38 - 0.30)
put("F total (/yr)", F, Flo, Fhi, f"bottom {Fb:.4f} pelagic {Fp:.4f}")
nm, nlo, nhi = wm("natmort"), wm("nm_lo"), wm("nm_hi")
put("natural_mortality M1 (/yr)", nm, nlo, nhi)
put("natural_mortality M1 (/tick)", nm / TICKS, nlo / TICKS, nhi / TICKS)
B_star = 1.20
K = B_star / (1 - F / r)
Klo = B_star / (1 - Flo / rhi)
Khi = B_star / (1 - Fhi / rlo)
put("regional_carrying_capacity K (t/km2)", K, Klo, Khi, "K = 1.20 / (1 - F/r)")
cm, clo, chi = wm("cmax_kJkgd"), wm("cmax_lo"), wm("cmax_hi")
put("Cmax (kJ/kg/d)", cm, clo, chi)
ceil, ceil_lo, ceil_hi = cm / 4.0, clo / 4.0, chi / 4.0  # 1 MJ/t/tick = 4 kJ/kg/d
put("energy ceiling (MJ/t/tick)", ceil, ceil_lo, ceil_hi)
sm, slo, shi = wm("smr"), wm("smr_lo"), wm("smr_hi")
put("resting_metabolism (MJ/t/tick)", sm, slo, shi)
put("metabolism_q10", wm("q10"), 2.0, 2.6)
el = wm("e_lean")
put("energy_content lean (MJ/t)", el, el - 300, el + 300)
rm, rmean = wm("res_max"), wm("res_mean")
put("max_energy_reserve (MJ/t)", rm, rm * 0.75, rm * 1.25)
put("reserve_reference_fill", rmean / rm, 0.4, 0.55, f"whole body as eaten {el + rmean:.0f} MJ/t")
qb, qlo, qhi = wm("qb"), wm("qb_lo"), wm("qb_hi")
put("Q/B (/yr), biomass-weighted", qb, qlo, qhi)
# starvation: routine (1.5 x SMR) / lean energy / u, u = 0.15
u = 0.15
put("starve_rate (/tick)", 1.5 * sm / el / u, sm / el / u, 2.5 * sm / el / u, "loss at s = 0 equals routine burn of lean tissue")

# prey pools: energy as eaten (MJ/t), biomass (t/km2), accessible fraction, diet share (consumption-weighted,
# evidence C; W kept) with range, and the share renormalised without W
PREY = {
    "planktivorous_fish": dict(E=7125, B=3.3, f=1.0, s=0.41, slo=0.30, shi=0.55, sren=0.45),
    "zooplankton": dict(E=4500, B=35000 / 3600, f=1.0, s=0.31, slo=0.15, shi=0.45, sren=0.35),
    "benthic_community": dict(E=3000, B=95.0, f=0.2, s=0.11, slo=0.05, shi=0.15, sren=0.12),
    "benthivorous_fish": dict(E=4500, B=1.53, f=0.15, s=0.07, slo=0.04, shi=0.15, sren=0.08),
}
Q = qb * B_star
print(f"\nconsumption Q = {qb:.3f} x {B_star} = {Q:.3f} t/km2/yr; ration {qb / 365 * 100:.2f} %BW/d")
mix_E = sum(p["sren"] * p["E"] for p in PREY.values()) / sum(p["sren"] for p in PREY.values())
print(f"field ration in energy {qb * mix_E / 365:.1f} kJ/kg/d "
      f"(diet energy {mix_E:.0f} MJ/t) = {qb * mix_E / 365 / cm:.2f} of Cmax")
for k, p in PREY.items():
    h = p["E"] / ceil
    hlo, hhi = p["E"] / ceil_hi, p["E"] / ceil_lo
    Bacc = p["B"] * p["f"]
    a = 1.0 / (h * Bacc)
    m2 = Q * p["s"] / p["B"]
    m2lo = qlo * B_star * p["slo"] / p["B"]
    m2hi = qhi * B_star * p["shi"] / p["B"]
    m2ren = Q * p["sren"] / p["B"]
    put(f"{k}: handling_time h (ticks)", h, hlo, hhi, f"E {p['E']} / ceiling")
    put(f"{k}: a for 1/(a h) = B_acc", a, a / 3, a * 3, f"B_acc {Bacc:.3g} t/km2")
    put(f"{k}: predation_mortality M2 (/yr)", m2, m2lo, m2hi, f"renormalised without W: {m2ren:.4f}")

# trawl tables (library construction, section 161): mean effort of today's maps
bot_mean, pel_mean = 3.0142, 0.02496
qb_bot = Fb / TICKS / bot_mean
qb_pel = Fp / TICKS / pel_mean
put("bottom_trawling biomass_factor at 100", qb_bot * 100, Flo * Fb / F / TICKS / bot_mean * 100,
    Fhi * Fb / F / TICKS / bot_mean * 100, f"q {qb_bot:.4e} per unit")
put("pelagic_trawling biomass_factor at 10", qb_pel * 10, qb_pel * 10 * 0.5, qb_pel * 10 * 1.5,
    f"q {qb_pel:.4e} per unit")
pd.DataFrame(rows).to_csv(os.path.join(d, "tables", "weighted_fields.csv"), index=False)
