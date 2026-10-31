"""The booked whiting -> planktivorous_fish M2 0.203 (section 223.37) split by prey species, and what its
box scaling assumes.

Usage: python3 -I step2_sms_m2_species.py <summary.out> <partial_m2.out> <who_eats_whom_level2.csv>
  (research_notes/Fiskgrupper i rutan/planktivores/data/sms2023/; SMS 2023 key run, untrusted, parsed only)

Same construction as ../../Bestandsreglering (a-ring)/scripts/sms_m2_check.py, whiting only, 2013-22:
partial M2 at the box stage mix J/B 0.38, scaled per prey species by
    f_sp = (B_whiting,box / B_sp,box) / (B_whiting,NS / B_sp,NS),
i.e. the box whiting are given the North Sea whiting's consumption PER WHITING TONNE of each species
(a fixed-ration, fixed-diet transfer). Two alternatives are printed: (b) linear clearance (the North Sea
per-capita M2 on each species x the box/North Sea whiting density: Type II far below saturation on that
prey); (c) a fixed total ration with the diet re-weighted by each species' box/North Sea availability
(multi-prey Type II: share_i ~ a_i B_i; other food assumed equally available in both, J). North Sea area
575 000 km2 (ICES IV) [F, standard figure], 632 000 km2 with IIIa [J] - used only for densities in (b), (c). Printed: the North Sea and box densities behind f_sp, the scaled M2 per
species (weighted by the pool shares), the implied box consumption C_sp = M2_sp x B_sp,box
(t/km2/yr) and the share of the whiting ration (Q/B 4.2, 3.3-5.6 [F, params_split_223.md]) it implies.
"""
import sys

import numpy as np
import pandas as pd

summ, part, who = sys.argv[1:4]
Y0, Y1 = 2013, 2022
PREY = {21: ("Herring", 0.79 / 0.895), 24: ("Nor.pout", 0.075 / 0.895), 25: ("Sprat", 0.03 / 0.895)}
JB = 0.38
B_PLK, B_WHI = 3.3, 0.446
BOX_PREY = {21: B_PLK * 0.79, 24: B_PLK * 0.075, 25: B_PLK * 0.03}
S = pd.read_csv(summ, sep=r"\s+")
for c in ("N.bar", "west", "M2"):
    S[c] = pd.to_numeric(S[c], errors="coerce")
S["B"] = S["N.bar"] * S.west
P = pd.read_csv(part, sep=r"\s+")
W = pd.read_csv(who)
Bidx = S.set_index(["Year", "Quarter", "Species.n", "Age"]).B
P = P[P["Prey.no"].isin(PREY)].copy()
P["B"] = [Bidx.get((y, q, s, a), np.nan) for y, q, s, a in zip(P.Year, P.Quarter, P["Prey.no"], P["Prey.age"])]
P["eaten"] = P["Part.M2"] * P.B
# name the whiting predator number by eaten-biomass match, as sms_m2_check.py
best, err = None, np.inf
for no, d in P.groupby("Predator.no"):
    e = d.groupby("Year").eaten.sum()
    w = W[(W.Predator == "Whiting") & W.Prey.isin([v[0] for v in PREY.values()])]
    x = w.groupby("Year").eatenW.sum().reindex(e.index).fillna(0)
    r = float(np.abs(np.log((x + 1) / (e + 1))).mean())
    if r < err:
        best, err = no, r
print(f"whiting = SMS predator number {best} (mean |log ratio| {err:.3f})")
P = P[(P["Predator.no"] == best) & (P.Year >= Y0) & (P.Year <= Y1)]
Sx = S[(S.Year >= Y0) & (S.Year <= Y1)]
nq = len(Sx.groupby(["Year", "Quarter"]))
Bq = Sx.groupby("Species.n").B.sum() / nq
whi_no = 17 if 17 in Bq.index else None
B_whi_ns = Bq[whi_no]
NS_AREA = 575000.0  # km2, SMS North Sea + 3a (J: area for densities only; the ratio f does not use it)
print(f"SMS 2013-22 mean biomass (t): whiting {B_whi_ns:,.0f}")
tot_m2 = tot_c = 0.0
M2NS = {}
print(f"{'prey':9s} {'NS B (t)':>12s} {'NS whi/prey':>12s} {'box whi/prey':>13s} {'f':>6s} {'M2 NS':>7s} {'M2 box':>7s}"
      f" {'pool w':>7s} {'w x M2box':>9s} {'C box t/km2/yr':>15s}")
for sp, (nm, wgt) in PREY.items():
    bJ = Sx[(Sx["Species.n"] == sp) & (Sx.Age <= 1)].B.sum()
    bA = Sx[(Sx["Species.n"] == sp) & (Sx.Age >= 2)].B.sum()
    d = P[P["Prey.no"] == sp]
    mJ = 4 * d[d["Prey.age"] <= 1].eaten.sum() / bJ
    mA = 4 * d[d["Prey.age"] >= 2].eaten.sum() / bA
    m = JB * mJ + (1 - JB) * mA
    f = (B_WHI / BOX_PREY[sp]) / (B_whi_ns / Bq[sp])
    mbox = m * f
    c_box = mbox * BOX_PREY[sp]
    tot_m2 += wgt * mbox
    M2NS[sp] = m
    tot_c += c_box * (wgt / (BOX_PREY[sp] / B_PLK))  # the pool-weighted M2 x 3.3 is the booked flow
    print(f"{nm:9s} {Bq[sp]:12,.0f} {B_whi_ns / Bq[sp]:12.3f} {B_WHI / BOX_PREY[sp]:13.3f} {f:6.2f} {m:7.3f} {mbox:7.3f}"
          f" {wgt:7.3f} {wgt * mbox:9.3f} {c_box:15.3f}")
print(f"pool-weighted box M2 (the booked value): {tot_m2:.3f} /yr -> flow {tot_m2 * B_PLK:.3f} t/km2/yr")
print("  NB the pool-weighted sum renormalises the three members to the pool (blue whiting has no SMS M2), so the")
print("  booked flow is not the sum of the per-species box flows; per-species flows (t/km2/yr) are the column above.")
for qb in (3.3, 4.2, 5.6):
    print(f"  implied planktivore share of the whiting ration at Q/B {qb}: {tot_m2 * B_PLK / (qb * B_WHI):.2f}")

# ---------------------------------------------------------------- alternative transfers to the box
# NS whiting diet at the box length mix (tables/sms_whiting_diet_stdout.txt 'all' row, scaled to the
# box-weighted planktivore share 0.255 of clearance_test_stdout.txt) [D]:
D_NS = {21: 0.255 * 0.034 / 0.284, 24: 0.255 * 0.174 / 0.284, 25: 0.255 * 0.075 / 0.284}
Q_B, Q_RANGE = 4.2, (3.3, 5.6)
print("\nAlternative transfers of the North Sea whiting predation to the box [D]:")
print(f"  NS whiting diet (box length mix): herring {D_NS[21]:.3f}, pout {D_NS[24]:.3f}, sprat {D_NS[25]:.3f}; "
      f"other food {1 - sum(D_NS.values()):.3f}")
for area in (575000.0, 632000.0):
    dens_ratio_whi = B_WHI / (B_whi_ns / area)
    m2c = sum(PREY[sp][1] * mJA * dens_ratio_whi for sp, mJA in M2NS.items())
    print(f"  (b) linear clearance (per-capita M2 x box/NS whiting density; NS area {area:,.0f} km2, whiting "
          f"{B_whi_ns / area:.3f} t/km2): M2 {m2c:.3f} /yr, flow {m2c * B_PLK:.3f} t/km2/yr")
for lab, ratio in (("per km2 (area 575 000 km2)", {sp: BOX_PREY[sp] / (Bq[sp] / 575000.0) for sp in PREY}),
                   ("per km2 (area 632 000 km2)", {sp: BOX_PREY[sp] / (Bq[sp] / 632000.0) for sp in PREY}),
                   ("per whiting tonne", {sp: (BOX_PREY[sp] / B_WHI) / (Bq[sp] / B_whi_ns) for sp in PREY})):
    plk = {sp: D_NS[sp] * ratio[sp] for sp in PREY}
    other = 1 - sum(D_NS.values())
    share = sum(plk.values()) / (sum(plk.values()) + other)
    m2 = share * Q_B * B_WHI / B_PLK
    lo = share * Q_RANGE[0] * B_WHI / B_PLK
    hi = share * Q_RANGE[1] * B_WHI / B_PLK
    print(f"  (c) fixed ration, diet re-weighted by prey availability {lab}: box/NS prey ratio " +
          ", ".join(f"{PREY[sp][0]} {ratio[sp]:.2f}" for sp in PREY) +
          f" -> planktivore share {share:.3f}, flow {share * Q_B * B_WHI:.3f} t/km2/yr, M2 {m2:.3f} ({lo:.3f}-{hi:.3f}) /yr")
