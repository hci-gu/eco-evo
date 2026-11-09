"""COPY of clearance_test.py with ONE change: the Skagerrak GOV ratios 1991-96 / 2001-25 are read from
tables/then_now_densities_fixedkey_stdout.txt (haul key with the shoot position) instead of the original stdout,
whose 2001-20 reference lost 218 of 980 Skagerrak hauls and added other hauls' catch to 24 more (the 26D4 key
collision). Herring 0.97 -> 0.98, blue whiting 0.94 -> 0.90, Norway pout 4.46 -> 4.65, sprat 0.45 -> 0.51,
benthivores 1.00 -> 1.06, whiting 0.87 -> 0.91, water column 0.30 -> 0.31, water column < 40 cm 0.84 -> 0.88.

Clearance-ratio test of the porpoises' four checked pairs against field diets (no model output read).

Usage: python3 -I clearance_test.py

Method as "../../Vitlingens fiskbyten/scripts/clearance_test.py": under concurrent search (one handling
budget) the mass taken of prey i is proportional to a_i f_i B_i, so the field clearance c_i = D_i / B_i
(up to a common factor) and c_i / c_anchor is free of Q/B, feeding level and saturation. It is compared
with the model's (a_i f_eff,i) / (a_anchor f_eff,anchor). Anchor: planktivores (0.65 of the booked diet,
f = 1 on size, the HERAS-level density is the best-measured box density).

Field diets, shares of the four model pools (renormalised; gobies, sandeels, pearlsides, hagfish,
eelpout and "fish < 5 %" dropped):
  V1 booked      library M2 x B (porpoises -> plk 0.0127, ben 0.0064, wht 0.019, wc 0.0077)
  V2 Borjesson   Borjesson et al. 2003, Kattegat-Skagerrak 1988-96, 112 bycaught, %W via Sveegaard 2011
                 Paper V Table 2: herring 56.2 + sprat 9.0 + Norway pout 4.1 = plk; cod 5.5 = ben;
                 whiting 5.4 = wht; saithe 5.0 = wc; "Gadidae spp." 6.8 split g to cod, 1 - g to whiting
                 (g 0.5; 0.25-0.75; Sveegaard: "presumably mainly cod and whiting")
  V3 Aarefjord   Aarefjord et al. 1995, Kattegat 1985-90, 39 porpoises (%W estimated, same table):
                 herring 42.1 + sprat 1.1 = plk; cod 10.6 = ben; whiting 8.2 = wht; Gadidae 11.1 split g
Prey densities (t/km2):
  today          the manifest's inference start (box): plk 3.33, ben 1.53, wht 0.446, wc 0.617
  1990s          today x the GOV ratio 1991-96 / 2001-25 in the Skagerrak (tables/then_now_densities_stdout.txt),
                 members weighted by today's composition; herring either by GOV (0.97) or by the stock mix
                 0.5 WBSS SSB (x2.90) + 0.5 NSAS SSB (x0.58) = x1.74 (ICES SAG 2025, data/sag/)
Stage access / accessible fraction (effective at the start J/B): plk 1; ben [1, 0.55] at J/B 0.15 -> 0.6175;
wht [1, 0.95] at J/B 0.143 (start mean) -> 0.957; wc (no compartment): window f (tables/wc_window_stdout.txt).
"""

B0 = {"plk": 3.333, "ben": 1.528, "wht": 0.446, "wc": 0.617}
P = ["plk", "ben", "wht", "wc"]
F_BEN = 0.15 * 1.0 + 0.85 * 0.55
F_WHT = 0.143 * 1.0 + 0.857 * 0.95
F_WC = {"< 30 cm": 0.0174, "< 40 cm": 0.0536, "ramp 30-50": 0.101, "< 45 cm": 0.156}
H = 28.6  # every porpoise pair (mass ceiling 0.035 t/t/tick)
E = {"plk": 4000 + 0.5 * 6250, "ben": 3600 + 0.5 * 1800, "wht": 3900 + 0.45 * 2000, "wc": 4180 + 0.45 * 2000}
ASSIM = {"plk": 0.9, "ben": 0.84, "wht": 0.84, "wc": 0.84}
EAT_COST = 90.0  # MJ/t/tick (resting 60 x feeding_cost 1.5)

# Skagerrak GOV ratios 1991-96 / 2001-25 (then_now_densities_stdout.txt, region skag)
R_SP = {"herring": 0.98, "blue whiting": 0.90, "Norway pout": 4.65, "sprat": 0.51}
COMP_PLK = {"herring": 0.79, "blue whiting": 0.105, "Norway pout": 0.075, "sprat": 0.03}
R_BEN, R_WHT, R_WC_ALL, R_WC_LT40 = 1.06, 0.91, 0.31, 0.88


def plk_ratio(r_herring):
    return sum(w * (r_herring if s == "herring" else R_SP[s]) for s, w in COMP_PLK.items())


DENS = {
    "today (box)": dict(B0),
    "1990s, GOV herring": {"plk": B0["plk"] * plk_ratio(0.98), "ben": B0["ben"] * R_BEN,
                           "wht": B0["wht"] * R_WHT, "wc": B0["wc"] * R_WC_ALL},
    "1990s, stock-mix herring": {"plk": B0["plk"] * plk_ratio(1.74), "ben": B0["ben"] * R_BEN,
                                 "wht": B0["wht"] * R_WHT, "wc": B0["wc"] * R_WC_ALL},
}
# for the wc pair the window part matters; the 1990s window biomass changed by R_WC_LT40, the group by R_WC_ALL.
# In the per-whole-tonne test below, the 1990s wc density is the group's; the window share then was higher:
# f_1990s = f_now * R_WC_LT40 / R_WC_ALL (saithe < 40 cm 0.29 of GOV biomass in 1991-96 vs 0.09 in 2001-25).
F_WC_SCALE_1990s = R_WC_LT40 / R_WC_ALL
print(f"planktivore ratio 1990s/today: GOV herring x{plk_ratio(0.98):.2f}, stock-mix herring x{plk_ratio(1.74):.2f}")
print(f"wc window share 1990s / today x{F_WC_SCALE_1990s:.2f} (more small saithe then)")


def booked():
    return {"plk": 0.0127 * B0["plk"], "ben": 0.0064 * B0["ben"], "wht": 0.019 * B0["wht"], "wc": 0.0077 * B0["wc"]}


def borjesson(g):
    return {"plk": 56.2 + 9.0 + 4.1, "ben": 5.5 + g * 6.8, "wht": 5.4 + (1 - g) * 6.8, "wc": 5.0}


def aarefjord(g):
    return {"plk": 42.1 + 1.1, "ben": 10.6 + g * 11.1, "wht": 8.2 + (1 - g) * 11.1, "wc": 0.0}


def norm(d):
    t = sum(d.values())
    return {k: d[k] / t for k in P}


FIELD = {
    "V1 booked": (norm(booked()), "today (box)"),
    "V2 Borjesson g0.5": (norm(borjesson(0.5)), None),
    "V2 Borjesson g0.25": (norm(borjesson(0.25)), None),
    "V2 Borjesson g0.75": (norm(borjesson(0.75)), None),
    "V3 Aarefjord g0.5": (norm(aarefjord(0.5)), None),
}

print("\nField diet shares on the four pools:")
for n, (d, _) in FIELD.items():
    print(f"  {n:20s} " + " ".join(f"{k} {d[k]:.3f}" for k in P))

# model a and f_eff
MODELS = {
    "M0 library": {"plk": (0.0225, 1.0), "ben": (0.12, F_BEN), "wht": (0.2, F_WHT), "wc": (0.2, 1.0)},
    "M1 library a, wc window f 0.054": {"plk": (0.0225, 1.0), "ben": (0.12, F_BEN), "wht": (0.2, F_WHT),
                                        "wc": (0.2, F_WC["< 40 cm"])},
    "M2 one clearance a 0.0225, windows": {"plk": (0.0225, 1.0), "ben": (0.0225, F_BEN), "wht": (0.0225, F_WHT),
                                           "wc": (0.0225, F_WC["< 40 cm"])},
    "M3 = M2 with whiting a 0.045 (field centre)": {"plk": (0.0225, 1.0), "ben": (0.0225, F_BEN),
                                                    "wht": (0.045, F_WHT), "wc": (0.0225, F_WC["< 40 cm"])},
}


def rel(m):
    af = {k: m[k][0] * m[k][1] for k in P}
    return {k: af[k] / af["plk"] for k in P}


print("\nModel a x f_eff relative to planktivores:")
for n, m in MODELS.items():
    r = rel(m)
    print(f"  {n:36s} " + " ".join(f"{k} {m[k][0] * m[k][1]:.5f} ({r[k]:5.2f})" for k in P))

print("\nField clearance c_i / c_plk per tonne of the WHOLE prey group (D_i / B_i over D_plk / B_plk):")
FC = {}
for fn, (d, fixed) in FIELD.items():
    for dn, B in DENS.items():
        if fixed and dn != fixed:
            continue
        c = {k: d[k] / B[k] for k in P}
        FC[(fn, dn)] = {k: c[k] / c["plk"] for k in P}
        print(f"  {fn:20s} @ {dn:26s} " + " ".join(f"{k} {FC[(fn, dn)][k]:6.3f}" for k in P))

print("\nDiscrepancy model / field of the clearance ratio to planktivores (1 = consistent; > 1 model too high):")
for mn, m in MODELS.items():
    r = rel(m)
    print(f"  {mn}")
    for key, fc in FC.items():
        out = []
        for k in ("ben", "wht", "wc"):
            rk = r[k]
            if k == "wc" and key[1].startswith("1990s") and m["wc"][1] < 1.0:
                rk = r[k] * F_WC_SCALE_1990s  # the window held more of the group then
            out.append(f"{k} {rk / fc[k]:7.2f}" if fc[k] > 0 else f"{k}     n/a")
        print(f"     vs {key[0]:20s} @ {key[1]:26s} " + " ".join(out))

print("\nImplied a_i = a_plk x (c_i / c_plk)_field / f_i, with a_plk 0.0225 kept (the a that makes the pair"
      " reproduce that field diet at that density under concurrent search):")
for key, fc in FC.items():
    parts = []
    for k, f in (("ben", F_BEN), ("wht", F_WHT), ("wc", F_WC["< 40 cm"])):
        ff = f * (F_WC_SCALE_1990s if (k == "wc" and key[1].startswith("1990s")) else 1.0)
        parts.append(f"{k} {0.0225 * fc[k] / ff:.4f}")
    print(f"  {key[0]:20s} @ {key[1]:26s} " + " ".join(parts))
print("  (wc a at the other window limits scales as 0.0536 / f: < 30 cm x3.1, ramp x0.53, < 45 cm x0.34)")

print("\nConcurrent-search diet shares a f_eff B / sum at today's box densities (all prey visible):")
for mn, m in MODELS.items():
    enc = {k: m[k][0] * m[k][1] * B0[k] for k in P}
    t = sum(enc.values())
    print(f"  {mn:36s} " + " ".join(f"{k} {enc[k] / t:.3f}" for k in P))
for fn in ("V1 booked", "V2 Borjesson g0.5", "V3 Aarefjord g0.5"):
    d = FIELD[fn][0]
    print(f"  field {fn:30s} " + " ".join(f"{k} {d[k]:.3f}" for k in P))

print("\nBox diet predicted from each field clearance set (c_field x B_today, concurrent search):")
for key, fc in FC.items():
    if key[1] == "today (box)":
        continue
    enc = {k: fc[k] * B0[k] for k in P}
    # wc: the field c was measured on the 1990s window share; today's window holds less of the group
    enc["wc"] = enc["wc"] / F_WC_SCALE_1990s
    t = sum(enc.values())
    print(f"  {key[0]:20s} @ {key[1]:26s} " + " ".join(f"{k} {enc[k] / t:.3f}" for k in P))

print("\nSingle-pair return at today's grid-mean density, all visible (x = a f h B; intake share x/(1+x) of the"
      " ceiling 0.035 t/t/tick; MJ/t/tick = share x 0.035 x E x assim; eat cost 90):")
for mn, m in MODELS.items():
    out = []
    for k in P:
        x = m[k][0] * m[k][1] * H * B0[k]
        mj = x / (1 + x) * (1 / H) * E[k] * ASSIM[k]
        out.append(f"{k} x {x:5.2f} -> {x / (1 + x):.2f}, {mj:5.0f} MJ")
    print(f"  {mn:36s} " + "; ".join(out))
print("  break-even visible density per pair, B = I / (a f (1 - I h)), I = 90 / (E assim):")
for mn, m in MODELS.items():
    out = []
    for k in P:
        i = EAT_COST / (E[k] * ASSIM[k])
        b = i / (m[k][0] * m[k][1] * (1 - i * H)) if i * H < 1 else float("inf")
        out.append(f"{k} {b:7.2f}")
    print(f"  {mn:36s} " + " ".join(out))

print("\nField ration check (scale, not ratio): ration 5.3 %BM/day (Rojano-Donate 2018 FMR on clupeids) ="
      " 0.0133 t/t/tick = 0.38 of the 0.035 ceiling.")
x_need = 0.38 / (1 - 0.38)
print(f"  Under continuous concurrent search (Wisniewska et al. 2016: porpoises forage day and night) the summed"
      f" x = h sum a f B that gives 0.38 is {x_need:.2f}; h sum a f B per model at today's box density:")
for mn, m in MODELS.items():
    s = H * sum(m[k][0] * m[k][1] * B0[k] for k in P)
    print(f"  {mn:36s} sum x {s:6.2f} -> feeding level {s / (1 + s):.2f} ({s / x_need:.1f}x the field-ration x)")

print("\nDiet gate against a re-booked diet (wc share 0.02 [J], the 0.053 freed pro rata to plk/ben/wht):")
bk = norm(booked())
k = (1 - 0.02) / (1 - bk["wc"])
rb = {"plk": bk["plk"] * k, "ben": bk["ben"] * k, "wht": bk["wht"] * k, "wc": 0.02}
print("  re-booked shares " + " ".join(f"{q} {rb[q]:.3f}" for q in P)
      + "; M2 /yr " + " ".join(f"{q} {rb[q] * 234.0 / (B0[q] * 3600):.4f}" for q in P) + " (Q/B 19.5, 12 t)")
for mn, m in MODELS.items():
    enc = {q: m[q][0] * m[q][1] * B0[q] for q in P}
    t = sum(enc.values())
    out = []
    for q in P:
        e = enc[q] / t
        out.append(f"{q} {e:.3f} ({e / bk[q]:.2f} | {e / rb[q]:.2f})")
    print(f"  {mn:44s} " + " ".join(out))
print("  (enc share; ratio to booked | to re-booked; the gate calls OFF outside 1/3-3 when either share >= 0.05)")
