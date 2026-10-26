"""Herring encounter factors: per-individual clearance rates and validation conversions.

Usage: python3 -I herring_encounter_checks.py > ../tables/herring_encounter_checks_stdout.txt

No downloads are read; every input is typed in below with its source key (see
../evidence_herring.md). Per-individual values only: numbers per tonne, overlap thickness
and the group weighting are done in the caller's script.

Conventions
- TL total length; SL = 0.87 TL for herring (J, typical clupeid ratio; Hamann et al. 2023
  give SL only).
- W(TL) = a TL^b, NS-IBTS CA fit for Clupea harengus (Fiskgrupper i rutan/tables/
  length_weight_all.csv): a 0.00223, b 3.370 (g, cm).
- Particulate (visual cruise) clearance: Q = f_field * pi R^2 v * p_capture, with f_field = 0.5
  (the 1/2 pi R^2 v of Varpe & Fiksen 2010 as reported by van Deurs et al. 2015, a forward
  half-field) - stated per case.
- Filter clearance: Q_f = A_gape * v_filter * retention.
"""
import math

A_LW, B_LW = 0.0022305830962786404, 3.3701933505454997
SL_PER_TL = 0.87
GAPE_PER_SL2 = 292.0 / 248.2 ** 2   # Hamann et al. 2023 Table 1: 292 mm2 at SL 248.2 mm


def weight_g(tl_cm):
    return A_LW * tl_cm ** B_LW


def gape_m2(tl_cm):
    sl_mm = SL_PER_TL * tl_cm * 10.0
    return GAPE_PER_SL2 * sl_mm ** 2 * 1e-6


def particulate(tl_cm, r_bl, v_bls, p_cap, half=0.5):
    tl = tl_cm / 100.0
    r, v = r_bl * tl, v_bls * tl
    return half * math.pi * r * r * v * p_cap   # m3/s


print("== 0. Size, weight, gape (Hamann 2023 scaling, isometric)")
for tl in (10, 15, 20, 25, 30):
    print(f"TL {tl:4.0f} cm  W {weight_g(tl):6.1f} g  gape {gape_m2(tl)*1e6:6.1f} mm2"
          f"  ind/t {1e6/weight_g(tl):9.0f}")
print(f"gape/SL^2 = {GAPE_PER_SL2:.5f}")

print("\n== 1. Particulate (day, saturating light) clearance per individual")
print("cases: (R in body lengths, v in BL/s, capture prob.)")
cases = {"low": (0.3, 0.6, 0.3), "central": (0.7, 1.0, 0.6), "high": (1.3, 1.4, 0.9)}
for tl in (15, 20, 25):
    for k, (r, v, p) in cases.items():
        q = particulate(tl, r, v, p)
        w = weight_g(tl)
        print(f"TL {tl} {k:8s} R {r:.1f} BL v {v:.1f} BL/s p {p:.1f}: Q {q*1e3:8.3f} L/s"
              f" = {q*3600:7.2f} m3/h = {q*21600:7.1f} m3/6h;  per g {q*3600/w*1e3:7.1f} L/g/h"
              f"  per t {q*21600/w*1e6:.3e} m3/t/6h")

print("\n== 2. Prey-size scaling of R in the Aksnes-Utne small-range limit, R ~ sqrt(A), A = 0.75 L W")
prey = {"Calanus CV/adult (2.5 x 0.8 mm)": (2.5, 0.8), "Calanus CIII (1.5 x 0.5)": (1.5, 0.5),
        "small copepod (1.0 x 0.4)": (1.0, 0.4), "Langbehn ref 4 mm (A 3e-6 m2)": None,
        "van Deurs ref 1.5 mm, W L/2": (1.5, 0.75)}
A_ref_L = 3e-6
A_ref_vD = 0.75 * 1.5e-3 * 0.75e-3
for k, lw in prey.items():
    A = A_ref_L if lw is None else 0.75 * lw[0] * 1e-3 * lw[1] * 1e-3
    print(f"{k:34s} A {A:.2e} m2  R/BL if 1 BL at Langbehn ref: {math.sqrt(A/A_ref_L):.2f};"
          f" if 1 BL at van Deurs ref: {math.sqrt(A/A_ref_vD):.2f}")

print("\n== 3. Filter clearance per individual (gape x speed x retention 0.9)")
for tl in (15, 20, 25):
    for lab, v in (("dark filtering 0.11 m/s (Batty 1986)", 0.11),
                   ("light 0.34 m/s (Batty 1986/Hamann 2023)", 0.34)):
        q = gape_m2(tl) * v * 0.9
        print(f"TL {tl} {lab:42s}: {q*1e3:6.3f} L/s = {q*3600*1e3:7.0f} L/h")

print("\n== 4. Night/day clearance ratio if night = dark filtering only, day = particulate central")
for tl in (15, 20, 25):
    qd = particulate(tl, *cases["central"])
    qn = gape_m2(tl) * 0.11 * 0.9
    qd_lo = particulate(tl, *cases["low"])
    print(f"TL {tl}: night/day = {qn/qd:.4f} (vs central day), {qn/qd_lo:.3f} (vs low day)")

print("\n== 5. Biting vs filtering crossover density N* = max bite rate / filter clearance")
for tl in (11.5, 15, 20, 25):
    for v in (0.11, 0.34):
        qf = gape_m2(tl) * v * 0.9 * 1e3   # L/s
        for srmax in (0.85, 1.0):
            print(f"TL {tl:5.1f} v {v:.2f} SRmax {srmax:.2f}/s: filter {qf:6.3f} L/s -> N* {srmax/qf:6.1f} prey/L")
print("Compare: Gibson & Ezzi 1985 ~50 Artemia nauplii/L; 1990 50 % filtering at 18 C. finmarchicus/L"
      " and 114 medium Artemia/L; Bernreuther 2007 filtering from >= 900 small copepods/L (9.9-13 cm)")

print("\n== 6. Is the bite rate a ceiling? 0.85 bites/s x 12 h feeding")
for name, mg_ww in (("small copepod 0.011 mg DW -> ~0.06 mg WW (DW/WW 0.18)", 0.011 / 0.18),
                    ("Calanus CV ~0.15 mg DW -> ~0.8 mg WW", 0.15 / 0.18)):
    for tl in (15, 20):
        g = 0.85 * 12 * 3600 * mg_ww / 1000.0
        print(f"{name:55s} TL {tl}: {g:6.2f} g/d = {100*g/weight_g(tl):6.1f} %BW/d")
print("Compare: weighted Cmax ceiling at h 40 = 10 %BW/d (params_planktivorous_fish.md section 2)")

print("\n== 7. Validation: measured clearance (initial slopes) vs theory")
# Bernreuther 2007 ms 4: SR = 0.849 N/(77.6 + N), snatches/s per fish, N copepods/L; 9.9-13 cm TL
tl_b = 11.5
q_b = 0.849 / 77.6          # L/s per fish
w_b = weight_g(tl_b)
print(f"Bernreuther lab: initial slope {q_b:.4f} L/s = {q_b*3600:.1f} L/h per fish;"
      f" W({tl_b} cm) {w_b:.1f} g -> {q_b*3600/w_b:.2f} L/g/h")
for k, (r, v, p) in {"theory R0.3 v1 p1": (0.3, 1.0, 1.0), "theory R0.6 v1 p1": (0.6, 1.0, 1.0)}.items():
    qt = particulate(tl_b, r, v, p) * 1e3
    print(f"   {k}: {qt:.3f} L/s -> measured/theory {q_b/qt:.3f}")
# Arrhenius & Hansson 1999: WSC = 0.21 Z/(1 + 26 Z); WSC g DW food / g DW fish / daylight h; Z g DW m-3
slope = 0.21                 # m3 per g DW fish per h
fish_dw = 0.22
q_ah = slope * fish_dw       # m3 per g WW fish per h
print(f"Arrhenius & Hansson field YOY: initial slope {slope} m3/gDW/h = {q_ah*1e3:.1f} L/gWW/h"
      f" = {q_ah*1e6*6:.3e} m3/t/6h (daylight)")
for tl_y in (4.0, 6.0, 8.0):
    w_y = weight_g(tl_y)
    q_y = q_ah * w_y / 3600 * 1e3   # L/s per fish
    for r in (0.5, 1.0):
        qt = particulate(tl_y, r, 1.0, 1.0) * 1e3
        print(f"   TL {tl_y} cm W {w_y:.2f} g: measured {q_y:.4f} L/s; theory R {r} BL v 1 BL/s p 1:"
              f" {qt:.4f} L/s -> ratio {q_y/qt:.2f}")
print("   half-saturation 1/26 g DW m-3 = 0.038 g DW m-3 = 0.30 g WW m-3 (DW/WW 0.13, their value)")
for h in (25.0, 50.0):
    print(f"   area form at overlap thickness {h:.0f} m: a = {q_ah*1e6*6/h/1e6:.4f} km2/t/6h (daylight)")

print("\n== 8. Light: biting thresholds (Batty et al. 1990) in umol photons (1 lx ~ 0.0185 umol m-2 s-1, J)")
for lx in (0.001, 0.01, 1.0, 10.0):
    print(f"{lx:8.3f} lx ~ {lx*0.0185:.2e} umol m-2 s-1")
