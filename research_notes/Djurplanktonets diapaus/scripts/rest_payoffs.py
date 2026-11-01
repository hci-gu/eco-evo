"""Derived table: what Rest costs in the model vs what dormancy / DVM resting costs in the field.

Usage: python3 -I rest_payoffs.py

No input files; every number is typed here with its source (see evidence.md):
- model: fg_library.yaml zooplankton (resting_metabolism 22.5 MJ/t/tick,
  feeding_cost 4.0, movement_cost 1.2, resting_cost default 1.0,
  energy_content 4500 MJ/t, max_energy_reserve 675 MJ/t, maintenance_level
  0.5, starve_rate 0.2, natural_mortality 0.015812 /tick, metabolism_q10 2.0,
  t_ref = annual mean of the layer) and mareld2.yaml temperature layers
  mean_0_50m and at_50m.
- deep water: SHARK A17 1991-2020 150-300 m monthly means
  (tables/a17_depth_temperature_stdout.txt).
- dormancy respiration factor (same temperature): 0.15-0.25 (Hirche 1983
  Table 1, Korsfjord deep vs surface CV at the same date and 6 C ETS:
  0.11/0.72, 0.18/0.79; Maps et al. 2014 meta-analysis ~1/4;
  Ingvarsdottir 1999 via Grigor 2022 10-20 %).
- potential diapause duration of C. finmarchicus CV: 280 d at 0 C, 90 d
  at 11 C (Saumweber & Durbin 2006 abstract); exponential interpolation.
"""
import math

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
T_MIX = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]
T_50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]
# A17 mean of the 150, 200 and 300 m rows, 1991-2020
T_DEEP = [(8.0 + 7.8 + 7.4) / 3, (7.2 + 7.1 + 6.9) / 3, (6.9 + 6.6 + 6.6) / 3,
          (6.9 + 6.8 + 6.5) / 3, (6.9 + 6.8 + 6.7) / 3, (7.0 + 7.0 + 6.8) / 3,
          (7.2 + 7.2 + 6.9) / 3, (7.3 + 7.3 + 6.9) / 3, (7.5 + 7.4 + 7.0) / 3,
          (7.7 + 7.6 + 7.2) / 3, (7.9 + 7.8 + 7.2) / 3, (8.6 + 8.2 + 7.4) / 3]
Q10 = 2.0
T_REF = sum(T_MIX) / 12.0
REST_TICK = 22.5           # MJ/t/tick at T_REF
E = 4500.0
RESERVE_MAX = 675.0
U = 0.5
STARVE = 0.2
M1_TICK = 0.015812


def q(t):
    return Q10 ** ((t - T_REF) / 10.0)


print(f"t_ref (annual mean of mean_0_50m) = {T_REF:.2f} C")
print("Model rest cost at t_ref: "
      f"{REST_TICK * 4 / E * 100:.2f} % body energy/d; eat x4 = "
      f"{REST_TICK * 4 * 4 / E * 100:.2f} %/d; move x1.2 = "
      f"{REST_TICK * 4 * 1.2 / E * 100:.2f} %/d")
buffer = (1.0 - U) * RESERVE_MAX
print(f"Reserve buffer full -> maintenance line: {buffer:.0f} MJ/t "
      f"= {buffer / E * 100:.1f} % of body energy")
print(f"Below the line biomass is lost at starve_rate x (s - u) per tick; at "
      f"an empty reserve {STARVE * U:.2f}/tick = "
      f"{1 - (1 - STARVE * U) ** 4:.2f}/d")
print(f"M1 = {M1_TICK:.6f}/tick = {M1_TICK * 4:.4f}/d = "
      f"{M1_TICK * 1460:.1f}/yr (constant)")
print()
hdr = ("month  T_mix  T_50  T_deep  q(mix)  q(50)  q(deep)  rest%/d  eat%/d"
       "  days_rest  dorm%/d(0.15-0.25)  days_dorm  rest@50/rest")
print(hdr)
for i, m in enumerate(MONTHS):
    rest = REST_TICK * 4 * q(T_MIX[i])            # MJ/t/d
    eat = rest * 4.0
    dorm_lo = REST_TICK * 4 * 0.15 * q(T_DEEP[i])
    dorm_hi = REST_TICK * 4 * 0.25 * q(T_DEEP[i])
    print(f"{m:5s}  {T_MIX[i]:5.1f}  {T_50[i]:4.1f}  {T_DEEP[i]:6.2f}"
          f"  {q(T_MIX[i]):6.2f}  {q(T_50[i]):5.2f}  {q(T_DEEP[i]):7.2f}"
          f"  {rest / E * 100:7.2f}  {eat / E * 100:6.2f}"
          f"  {buffer / rest:9.1f}"
          f"  {dorm_lo / E * 100:7.2f}-{dorm_hi / E * 100:.2f}"
          f"        {buffer / dorm_hi:4.0f}-{buffer / dorm_lo:3.0f}"
          f"       {q(T_50[i]) / q(T_MIX[i]):5.2f}")

print()
k = math.log(280.0 / 90.0) / 11.0
for t in (6.0, 7.0, 7.4, 8.0):
    print(f"Saumweber & Durbin interpolation: potential C5 diapause at {t:.1f} C"
          f" = {280.0 * math.exp(-k * t):.0f} d")
print()
print("Mortality per day (field, see evidence.md) vs model M1 "
      f"{M1_TICK * 4:.4f}/d:")
rows = [
    ("diapause, oceanic basins SW Iceland, Nov-Apr [S]", 0.004),
    ("diapause, deep fjord without mesopelagic fish (Bagoien 2001)", 0.0085),
    ("diapause, deep clear fjord with Maurolicus/Benthosema (Bagoien 2001)",
     0.0255),
    ("active C5, Fladen Ground spring (Eiane & Ohman 2004)", 0.05),
    ("active C. helgolandicus C5-F, L4, 13.8 C (Hirst et al. 2007)", 0.099),
    ("H&K 2002 broadcast spawners at 13.8 C", 0.089),
    ("H&K 2002 broadcast spawners at 5.3 C", 0.048),
    ("H&K 2002 predation-free (lab) at 5 C", 0.019),
]
for name, m in rows:
    print(f"  {name:72s} {m:6.4f}/d  = {m / (M1_TICK * 4):5.2f} x model M1")
lo, hi = 0.05, 0.18
print(f"  S. elegans removal of deep Calanus, Gullmar fjord autumn: "
      f"{lo:.2f}-{hi:.2f}/d of standing stock = "
      f"{-math.log(1 - lo):.3f}-{-math.log(1 - hi):.3f}/d instantaneous")
