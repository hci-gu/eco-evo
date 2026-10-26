"""Encounter-model factors for the non-herring members of planktivorous_fish
(sprat, Norway pout, blue whiting): per-individual clearance volume rate and per-tonne
clearance per 6 h tick, BEFORE division by any vertical overlap thickness.

Usage: python3 -I members_encounter.py <length_weight_all.csv>
Output (stdout) is kept in ../tables/members_encounter_stdout.txt.

Model (sources in ../evidence_sprat_pout_bluewhiting.md):
  saturated-light visual range  r = k_v * sqrt(A_prey),  A_prey = L * (w L) * 0.75
      (Aksnes & Utne 1997 form at small c; image-area rule of Langbehn et al. 2019 S1)
  clearance  beta = f_field * pi r^2 v * p_capture   [m3 s-1 per fish]
      f_field = 1 (full circle) or 0.5 (half the visual area scanned; Varpe & Fiksen 2010
      as used by Langbehn et al. 2019) - both are printed.
k_v anchors [D]: Langbehn Benthosema 0.06 m on 2.7 mm copepod and piscivore 1.5 m on
6 cm fish -> ~60; Utne 1997 Gobiusculus 18.9 cm on C. finmarchicus -> ~155; Kulke 2018
sprat 8.7 cm on Artemia, back-calculated -> ~90-120.
"""
import csv
import math
import sys

lw = {}
with open(sys.argv[1], newline="") as fh:
    for row in csv.DictReader(fh):
        lw[row["species"]] = (float(row["a"]), float(row["b"]))


def area(L_mm, wl):
    L = L_mm / 1000.0
    return L * (wl * L) * 0.75


print("== k_v anchors (r = k_v sqrt(A))")
for name, r, Lmm, wl in [("Benthosema 6 cm on 2.7 mm copepod (Langbehn S1)", 0.06, 2.7, 0.2),
                         ("piscivore 50 cm on 6 cm fish (Langbehn S1)", 1.5, 60.0, 0.2),
                         ("Gobiusculus on C. finmarchicus 2.5 mm (Utne 1997)", 0.189, 2.5, 0.33),
                         ("Gobiusculus on Acartia 1.0 mm (Utne 1997, 8 umol)", 0.129, 1.0, 0.33)]:
    print(f"  {name:55s} k_v = {r / math.sqrt(area(Lmm, wl)):6.0f}")
# Kulke 2018: 8.7 cm sprat, 16 C, Artemia nauplii 0.5-0.9 mm; lab clearance BRmax/k
for br, k, lab in [(2.04, 28.42, "6.3 cm sprat, Artemia (ms1)"), (1.06, 25.42, "6.3 cm sprat, A. tonsa (ms1)"),
                   (2.0, 10.7, "8.7 cm sprat, Artemia, 16 C (ms2, k = -0.11 T + 12.45)"),
                   (2.17, 36.36, "8 cm herring, Artemia (ms1)"), (0.95, 12.01, "8 cm herring, A. tonsa (ms1)")]:
    beta = br / k / 1000.0  # m3 s-1
    print(f"  Kulke lab clearance {lab:52s} {beta * 1e3:.3f} L/s = {beta * 3600:.2f} m3/h")
for v_bl in (0.3, 0.5):
    v = v_bl * 0.087
    beta = 2.0 / 10.7 / 1000.0
    for p in (1.0, 0.7):
        r = math.sqrt(beta / (math.pi * v * p))
        print(f"  back-calc 8.7 cm sprat: v {v_bl} BL/s, p {p}: r = {r * 100:.1f} cm,"
              f" k_v = {r / math.sqrt(area(0.7, 0.4)):.0f} (Artemia 0.7 mm, w/L 0.4)")

# member parameter sets: (length cm, prey length mm, prey w/L, k_v, v BL/s, p_capture)
members = {
    "sprat":        ("Sprattus sprattus",        {"low": (10, 1.0, 0.33, 60, 0.3, 0.33),
                                                  "central": (12, 1.2, 0.33, 100, 0.6, 0.7),
                                                  "high": (14, 1.5, 0.33, 155, 1.0, 0.86)}),
    "norway_pout":  ("Trisopterus esmarkii",     {"low": (11, 1.2, 0.33, 60, 0.2, 0.33),
                                                  "central": (15, 2.0, 0.33, 100, 0.33, 0.7),
                                                  "high": (20, 2.5, 0.33, 155, 0.6, 0.86)}),
    "blue_whiting": ("Micromesistius poutassou", {"low": (20, 1.2, 0.33, 60, 0.2, 0.33),
                                                  "central": (25, 2.0, 0.33, 100, 0.33, 0.7),
                                                  "high": (30, 2.5, 0.33, 155, 0.6, 0.86)}),
}
print("\n== Members on copepod-sized pool prey, SATURATING light, before overlap thickness")
print("member        case     L_cm  W_g   N_per_t   prey_mm  r_cm   v_cm_s  p     beta_m3_h(f=1)"
      "  per_t_m3_tick(f=1)  per_t_m3_tick(f=0.5)")
for key, (sp, cases) in members.items():
    a, b = lw[sp]
    for case, (L, Lp, wl, kv, vbl, p) in cases.items():
        W = a * L ** b
        n_t = 1e6 / W
        r = kv * math.sqrt(area(Lp, wl))
        v = vbl * L / 100.0
        beta = math.pi * r * r * v * p  # m3 s-1
        tick = beta * 6 * 3600 * n_t
        print(f"{key:13s} {case:8s} {L:4.0f} {W:6.1f} {n_t:9.0f} {Lp:7.1f} {r * 100:6.1f} {v * 100:7.1f}"
              f" {p:5.2f} {beta * 3600:14.3f} {tick:19.3e} {0.5 * tick:20.3e}")

print("\n== Krill-sized prey (20 mm, w/L 0.2) for comparison (NOT in the 0-25 m WP2 pool)")
for key, L, vbl in [("norway_pout", 15, 0.33), ("blue_whiting", 25, 0.33)]:
    sp = members[key][0]
    a, b = lw[sp]
    W = a * L ** b
    for kv in (60, 100, 155):
        r = kv * math.sqrt(area(20.0, 0.2))
        beta = math.pi * r * r * vbl * L / 100.0 * 0.5
        print(f"  {key:13s} k_v {kv:3d}: r = {r:.2f} m, beta(p 0.5) = {beta * 3600:.1f} m3/h,"
              f" per t per tick = {beta * 21600 * 1e6 / W:.3e} m3")

print("\n== Validation: sprat field clearance (Kulke 2018 Fig 5.5, k = 291 L^-1.5) per tonne")
a, b = lw["Sprattus sprattus"]
for L in (10, 12, 14):
    k = 291.0 * L ** -1.5
    W = a * L ** b
    for brmax in (1.0, 2.0):
        beta = brmax / k / 1000.0
        print(f"  L {L} cm (W {W:.1f} g): k {k:.1f} /L, BRmax {brmax}: clearance {beta * 3600:.2f} m3/h,"
              f" per t per tick {beta * 21600 * 1e6 / W:.3e} m3; a at H 25 m = {beta * 21600 * 1e6 / W / 25 / 1e6:.4f},"
              f" H 50 m = {beta * 21600 * 1e6 / W / 50 / 1e6:.4f} km2/t/tick")
