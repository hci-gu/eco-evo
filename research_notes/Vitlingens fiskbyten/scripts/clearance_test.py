"""Clearance-ratio test of whiting's four checked pairs against field diets (no model output read).

Usage: python3 -I clearance_test.py

Concurrent search (Holling multi-prey with one handling budget): intake_i proportional to
a_i f_i B_i, so the diet share D_i = a_i f_i B_i / sum_j a_j f_j B_j at any saturation. The FIELD
clearance rate is therefore c_i = D_i / B_i (up to one common factor), and the ratio c_i / c_j is
free of Q/B, of the feeding level and of saturation. It is compared with the model ratio
(a_i f_eff,i) / (a_j f_eff,j). Zooplankton is the anchor pair: no stage access, no bed geometry
(the near-bed krill layer, 20-40 m thick by day, contains the whiting's 5-20 m layer).

Field diets (shares of the four checked prey groups only; renormalised):
  V1 booked    the library's M2 x B (whiting -> plk 0.203, zoo 0.0544, epi 0.069, ben 0.0967)
  V2 223.26    research_notes/"Rovfiskens uppdelning" params_split_223.md 3.1, bed row (diet of the
               whole food: plk 0.280, zoo 0.280, epi 0.137, ben 0.080)
  V3 SMS+H91   SMS key-run stomachs (North Sea 1981-91, tables/sms_whiting_diet_stdout.txt) for the
               planktivore and benthivore shares, Hislop et al. 1991 (1981, via ICES FishMap Fig. 4)
               for the invertebrate share, split zoo / epi / infauna 0.70 / 0.19 / 0.10 as in the
               Skagerrak 1991 stomachs; all weighted to the box whiting length mix by consumption
  V4 Skag91    ICES stomach DB, Skagerrak April 1991, 106 stomachs (tables/whiting_diet_ices_stdout.txt)
Prey biomass today (t/km2): planktivores 3.3, zooplankton 9.72 (inference start; 7.0 as a variant),
mobile epifauna 3.78, benthivores 1.53.
"""

B = {"plk": 3.3, "zoo": 9.72, "epi": 3.78, "ben": 1.53}
H = {"plk": 227.3, "zoo": 143.6, "epi": 119.7, "ben": 143.6}
JB_PLK, JB_BEN = 0.38, 0.15

# box whiting biomass by length band (Fiskgrupper i rutan/predators/tables/pred_length_biomass.csv, pooled),
# consumption weight per tonne ~ W^-0.2 ~ L^-0.6 (band mid-lengths 16, 22.5, 27.5, 32.5, 38 cm)
bio = {"<20": 0.97 + 3.94 + 8.60, "20-24": 16.81, "25-29": 12.50, "30-34": 6.95, ">=35": 3.50 + 1.02 + 0.35 + 0.03}
mid = {"<20": 16.0, "20-24": 22.5, "25-29": 27.5, "30-34": 32.5, ">=35": 38.0}
cw = {k: bio[k] * mid[k] ** -0.6 for k in bio}
s = sum(cw.values())
cw = {k: v / s for k, v in cw.items()}
bs = sum(bio.values())
print("Box whiting: biomass share by band " + ", ".join(f"{k} {bio[k] / bs:.3f}" for k in bio))
print("             consumption weight     " + ", ".join(f"{k} {cw[k]:.3f}" for k in cw))
# SMS by band (all quarters): plk, ben, own, sand, other
SMS = {"<20": (0.081, 0.002, 0.006, 0.107), "20-24": (0.214, 0.002, 0.017, 0.200),
       "25-29": (0.359, 0.034, 0.028, 0.190), "30-34": (0.514, 0.029, 0.041, 0.119),
       ">=35": (0.522, 0.075, 0.039, 0.079)}
# Hislop et al. 1991 (1981 North Sea), fish share by weight read from FishMap Fig. 4 (+/- 0.03)
HISLOP_FISH = {"<20": 0.39, "20-24": 0.69, "25-29": 0.83, "30-34": 0.88, ">=35": 0.87}
plk3 = sum(cw[k] * SMS[k][0] for k in cw)
ben3 = sum(cw[k] * SMS[k][1] for k in cw)
smsfish = sum(cw[k] * sum(SMS[k]) for k in cw)
fish_h = sum(cw[k] * HISLOP_FISH[k] for k in cw)
inv = 1.0 - fish_h
zoo3, epi3 = 0.70 * inv, 0.19 * inv
print(f"V3: SMS planktivores {plk3:.3f}, benthivores {ben3:.3f}, SMS-modelled fish total {smsfish:.3f}; "
      f"Hislop all fish {fish_h:.3f} -> invertebrates {inv:.3f} -> zoo {zoo3:.3f}, epi {epi3:.3f}\n")

FIELD = {
    "V1 booked": {"plk": 0.203 * B["plk"], "zoo": 0.0544 * B["zoo"], "epi": 0.069 * B["epi"], "ben": 0.0967 * B["ben"]},
    "V2 223.26": {"plk": 0.280, "zoo": 0.280, "epi": 0.137, "ben": 0.080},
    "V3 SMS+H91": {"plk": plk3, "zoo": zoo3, "epi": epi3, "ben": ben3},
    "V4 Skag91": {"plk": 0.069, "zoo": 0.629, "epi": 0.168, "ben": 0.002},
}


def feff(fj, fa, jb):
    return jb * fj + (1 - jb) * fa


MODELS = {
    "M0 library": {"plk": 0.021 * feff(0.12, 0.008, JB_PLK), "zoo": 0.0023, "epi": 0.27 * 0.29,
                   "ben": 0.21 * feff(0.049, 0.0, JB_BEN)},
    "M0' plk f=1 (pre-223.32)": {"plk": 0.021, "zoo": 0.0023, "epi": 0.27 * 0.29, "ben": 0.21 * feff(0.049, 0.0, JB_BEN)},
    "M1 proposed (epi a 0.008)": {"plk": 0.021 * feff(0.16, 0.010, JB_PLK), "zoo": 0.0023, "epi": 0.008 * 0.29,
                                  "ben": 0.21 * feff(0.049, 0.0, JB_BEN)},
    "M1 overlap 10 m both 0.0116": {"plk": 0.021 * feff(0.16, 0.010, JB_PLK), "zoo": 0.0023, "epi": 0.0116 * 0.29,
                                    "ben": 0.21 * feff(0.049, 0.0, JB_BEN)},
    "M1 low epi a 0.006": {"plk": 0.021 * feff(0.16, 0.010, JB_PLK), "zoo": 0.0023, "epi": 0.006 * 0.29,
                           "ben": 0.21 * feff(0.049, 0.0, JB_BEN)},
    "M1 high epi a 0.023": {"plk": 0.021 * feff(0.16, 0.010, JB_PLK), "zoo": 0.0023, "epi": 0.023 * 0.29,
                            "ben": 0.21 * feff(0.049, 0.0, JB_BEN)},
}
P = ["plk", "zoo", "epi", "ben"]
print("Field diet shares (four checked pairs, renormalised) and clearance c = D/B relative to zooplankton:")
fc = {}
for n, d in FIELD.items():
    t = sum(d.values())
    sh = {k: d[k] / t for k in P}
    c = {k: sh[k] / B[k] for k in P}
    fc[n] = {k: c[k] / c["zoo"] for k in P}
    print(f"  {n:11s} shares " + " ".join(f"{k} {sh[k]:.3f}" for k in P) + "   c/c_zoo "
          + " ".join(f"{k} {fc[n][k]:.2f}" for k in P))
print("\nModel a x f_eff (km2/t/tick) and relative to zooplankton:")
mc = {}
for n, m in MODELS.items():
    mc[n] = {k: m[k] / m["zoo"] for k in P}
    print(f"  {n:28s} " + " ".join(f"{k} {m[k]:.5f}" for k in P) + "   rel " + " ".join(f"{k} {mc[n][k]:.2f}" for k in P))
print("\nDiscrepancy model / field of the clearance ratio to zooplankton (1 = consistent; >1 model too high):")
for n in MODELS:
    print(f"  {n}")
    for f in FIELD:
        print(f"     vs {f:11s} " + " ".join(f"{k} {mc[n][k] / fc[f][k]:7.2f}" if fc[f][k] > 0 else f"{k}     n/a" for k in ("plk", "epi", "ben")))
print("\nConcurrent-search diet shares a f_eff B / sum (all prey visible), vs field:")
for n, m in MODELS.items():
    for zb in (9.72, 7.0):
        bb = dict(B, zoo=zb)
        tot = sum(m[k] * bb[k] for k in P)
        print(f"  {n:28s} zoo B {zb:5.2f}: " + " ".join(f"{k} {m[k] * bb[k] / tot:.3f}" for k in P))
print("  planktivores 25 % visible (pair visibility_floor 0.25), zoo B 9.72:")
for n, m in MODELS.items():
    w = dict(B, plk=0.25 * B["plk"])
    tot = sum(m[k] * w[k] for k in P)
    print(f"  {n:28s}               " + " ".join(f"{k} {m[k] * w[k] / tot:.3f}" for k in P))

print("\nSingle-pair Holling return at today's density (x = a f h B; share of the intake ceiling x/(1+x));"
      " half-saturation 1/(a f h) in t/km2 of the prey group:")
for n, m in MODELS.items():
    out = []
    for k in P:
        x = m[k] * H[k] * B[k]
        out.append(f"{k} x {x:6.2f} -> {x / (1 + x):.2f} (K {1 / (m[k] * H[k]):.2f})")
    print(f"  {n:28s} " + "; ".join(out))

print("\nTime-allocation reading (check, not a derivation): if the field diet is the outcome of free allocation,"
      " no pair can pay far more per allocated tonne than the others at field density. The epifauna a that puts"
      " its return x/(1+x) equal to r_other:")
for r in (0.44, 0.6, 0.76):
    x = r / (1 - r)
    print(f"  r_other {r:.2f}: x {x:.2f} -> a_epi = x / (h f B) = {x / (H['epi'] * 0.29 * B['epi']):.4f}")
