"""Check (not a derivation): whiting's concurrent-search diet under the layer and overlap conventions
for the planktivore pair, against field diets. No model output is read.

Usage: python3 -I step2_diet_check.py

D_i = a_i f_i B_i / sum_j a_j f_j B_j (one handling budget; all prey visible), as clearance_test.py.
Pairs other than planktivores at the section 232 library values: zooplankton a 0.0023, epifauna
0.008 x f 0.29, benthivores 0.21 x f_eff (J/B 0.15) [0.049, 0]. Planktivore a x f_eff from
tables/step2_whiting_overlap_stdout.txt (window ratio 0.39; 0.45 in brackets).
ZOO_OV: under the overlap convention the zooplankton pair would also change (the pool is the 0-25 m
A17 community; krill 0.12 of it sit in a 20-40 m layer at ~70-100 m by day): shown at x1, x0.5, x0.25
as a sensitivity only (J; that pair is not re-derived here).
Field (shares of the four checked prey groups, renormalised):
  Skag91 Q2   ICES DB Skagerrak April 1991, box rectangles (tables/whiting_diet_ices_stdout.txt)
  Jut81 Q1    ICES DB 1981 Q1, 43F6-43F8 (counts x item weights; tables/step2_ices1981_counts_stdout.txt):
              planktivores 0.17, krill+other zooplankton 0.33, shrimp+crab+mysid 0.45, benthivores 0 (J split)
  Katt91      ICES DB Kattegat 1991 (shallow; sprat/herring nursery)
  NS SMS+H91  V3 of clearance_test.py (North Sea, box length mix)
  box (c)     step2_sms_m2_species.py transfer (c): planktivores 0.11-0.14 of the ration; the rest split as
              V3's invertebrates and benthivores (J)
"""
B = {"plk": 3.3, "zoo": 9.72, "epi": 3.78, "ben": 1.53}
FIX = {"zoo": 0.0023, "epi": 0.008 * 0.29, "ben": 0.21 * (0.15 * 0.049)}
PLK = {"layer (library, [0.16, 0.010])": (0.0211 * 0.066, 0.00236),
       "overlap, static band": (0.00091, 0.00149),
       "overlap + follow 0.3": (0.00098, 0.00161)}
FIELD = {
    "Skag91 Q2 (box rect.)": {"plk": 0.069, "zoo": 0.629, "epi": 0.168, "ben": 0.002},
    "Jut81 Q1 (counts)": {"plk": 0.17, "zoo": 0.33, "epi": 0.45, "ben": 0.0},
    "Katt91": {"plk": 0.472, "zoo": 0.172, "epi": 0.121, "ben": 0.0},
    "NS SMS+H91 (V3)": {"plk": 0.443, "zoo": 0.416, "epi": 0.113, "ben": 0.028},
    "box transfer (c)": {"plk": 0.125, "zoo": 0.416 / 0.557 * 0.85, "epi": 0.113 / 0.557 * 0.85,
                         "ben": 0.028 / 0.557 * 0.85},
}
P = ["plk", "zoo", "epi", "ben"]
print("Field diet shares of the four checked pairs (renormalised):")
for k, d in FIELD.items():
    s = sum(d.values())
    print(f"  {k:24s} " + "  ".join(f"{p} {d[p] / s:.3f}" for p in P))
print("\nModel concurrent-search diet shares (all prey visible), window 0.39 [0.45]:")
for zf in (1.0, 0.5, 0.25):
    for lab, (a39, a45) in PLK.items():
        if lab.startswith("layer") and zf != 1.0:
            continue
        out = []
        for a in (a39, a45):
            x = {"plk": a * B["plk"], "zoo": FIX["zoo"] * zf * B["zoo"], "epi": FIX["epi"] * B["epi"],
                 "ben": FIX["ben"] * B["ben"]}
            s = sum(x.values())
            out.append({p: x[p] / s for p in P})
        print(f"  {lab:32s} zoo a x{zf:4.2f}: " + "  ".join(f"{p} {out[0][p]:.3f} [{out[1][p]:.3f}]" for p in P))
