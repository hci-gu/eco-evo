"""Whiting krill rations in the Oslofjord krill layer in November 2000, March 2001 AND June 2001 (Onsrud et al.
2004, ICES J Mar Sci 61:1278, Table 3 [F]; the June row was not used in "Krillparen mot vintermagar").

Same method as research_notes/"Krillparen mot vintermagar"/scripts/onsrud_rations.py [S], re-run here for the
three months (no change of method):
- krill content per fish = share with krill (mean of day and night) x krill per krill-bearing stomach x the fresh
  mass of one adult M. norvegica (0.15 / 0.23 / 0.30 g WW);
- ration: ICES gadoid GER (WGSAM 2024 Table 3-7, whiting set) 24 rho e^(0.081 T) L^1.35 E^-0.85 sqrt(S) on
  krill-bearing stomachs x the share with krill; E 5.0 kJ/g; T 7 C (deep water homogeneous below ~30 m [F];
  6-8 C [J]); whiting 21-28 cm [J] (lengths not reported).
- model scale: ration / 2.51 %BW/d (the model ceiling 125.4 kJ/kg/d at 5 kJ/g krill).
Usage: python3 -I onsrud_three_months.py
"""
import math

T3 = {"Nov 2000": ((42.31, 35.19), (1.82, 3.47), (2.67, 3.00), (25.00, 14.81)),
      "Mar 2001": ((75.51, 79.69), (3.95, 5.92), (3.28, 3.25), (8.16, 6.25)),
      "Jun 2001": ((42.86, 34.62), (2.17, 1.44), (3.08, 2.23), (7.14, 26.92))}
RHO, LAM, DEL, MU, E, T = 0.00171, 1.35, 0.081, 0.85, 5.0, 7.0
CEIL_KRILL = 2.51

for mo, ((wd, wn), (nd, nn), full, empty) in T3.items():
    P = (wd + wn) / 200.0
    n_with = (wd / 100 * nd + wn / 100 * nn) / (2 * P)
    rs = []
    for L in (21, 28):
        W = 0.00603 * L ** 3.077
        for wk in (0.15, 0.30):
            S_with = n_with * wk
            R = 24 * RHO * math.exp(DEL * T) * L ** LAM * E ** (-MU) * math.sqrt(S_with) * P
            rs.append(100 * R / W)
    lo, hi = min(rs), max(rs)
    print(f"whiting {mo}: krill FO {100*P:4.1f} %, {n_with:4.2f} krill per krill-bearing stomach, all-fish mean "
          f"{P*n_with:4.2f}; fullness d/n {full}; empty d/n {empty} %; krill ration {lo:.2f}-{hi:.2f} %BW/d"
          f" -> {lo/CEIL_KRILL:.2f}-{hi/CEIL_KRILL:.2f} of the model ceiling")
