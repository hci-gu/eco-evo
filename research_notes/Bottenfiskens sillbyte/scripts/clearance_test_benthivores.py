"""Clearance-ratio test of benthivorous_fish -> planktivorous_fish against field diets, anchored on the
benthivores' bed pairs; factor decomposition of the pair's a x f; per-species (pout / herring / sprat /
blue whiting) test; concurrent-search and time-allocation diet shares. No model output is read.

Usage: python3 -I clearance_test_benthivores.py <benth_diet_ices.csv> <stage_length_biomass.csv>

Method (as research_notes/"Vitlingens fiskbyten"/scripts/clearance_test.py): with concurrent search and one
handling budget the diet share D_i ~ a_i f_i B_i at any saturation, so the field clearance c_i = D_i / B_i
(up to one common factor) and c_i / c_anchor is free of Q/B, feeding level and saturation. Model
clearance per TOTAL tonne of prey: a x (accessible share): infauna 0.036 x exposed 0.2, epifauna
0.15 x 0.62, Nephrops 0.44 x 0.02 (B_vis/B in the gate), planktivores 0.0062 x f_eff (stage access at
J/B 0.38), whiting 0.0026 x f_eff (J/B 0.15). Anchors: infauna and epifauna (the gate passes both).

Field diets (shares of the five checked prey groups; bed split infauna : epifauna : Nephrops by the booked
0.578 : 0.318 : 0.016 where a source gives one benthos number):
  V1 booked    library M2 x B (budget_gate.py last block)
  V2 223.10    in-area central diet after the no-pool redistribution (infauna 0.544, epifauna 0.314 of which
               Nephrops 0.015, planktivores 0.120, predators/whiting 0.023)
  V3 evid. C   "Fiskgrupper i rutan"/benthivores evidence C pooled consumption-weighted diet: benthos 72 %,
               P 8.4 % (variants 6.5 / 14.7), W 1.9 %
  V4 SMS       SMS 2023 key run stomachs (North Sea 1981-91) for cod and haddock at the box length mixes
               (tables/sms_diet_pred16.txt, sms_diet_pred18.txt); other members from evidence C
  V5 ICES deep ICES stomach DB post-2000: cod Skagerrak >= 80 m and NNS >= 100 m, haddock NNS >= 100 m
               (tables/benth_diet_ices.csv), box length mixes; other members from evidence C
"""
import sys

import numpy as np
import pandas as pd

ICES = pd.read_csv(sys.argv[1])
SL = pd.read_csv(sys.argv[2])

# ---------------------------------------------------------------- model, current library
B = {"inf": 93.61, "epi": 3.78, "neph": 4.58, "plk": 3.33, "wht": 0.45}     # inference start (gate)
JB_PLK, JB_WHT = 0.38, 0.15
SA_PLK, SA_WHT = (0.494, 0.136), (0.376, 0.042)


def feff(sa, jb):
    return jb * sa[0] + (1 - jb) * sa[1]


A_MODEL = {"inf": 0.036 * 0.2, "epi": 0.15 * 0.62, "neph": 0.44 * 0.092 / 4.58,
           "plk": 0.0062 * feff(SA_PLK, JB_PLK), "wht": 0.0026 * feff(SA_WHT, JB_WHT)}
P5 = ["inf", "epi", "neph", "plk", "wht"]

# ---------------------------------------------------------------- box member mixes and consumption weights
BEN_W = {"haddock": 0.386, "cod": 0.314, "long rough dab": 0.126, "dab": 0.045, "plaice": 0.038, "witch": 0.017,
         "starry ray": 0.015, "hagfish": 0.013, "grey gurnard": 0.009, "lemon sole": 0.009, "others": 0.033}
QB = {"haddock": 3.8, "cod": 3.2, "long rough dab": 3.4, "dab": 3.5, "plaice": 3.42, "witch": 3.0, "starry ray": 1.7,
      "hagfish": 4.0, "grey gurnard": 3.2, "lemon sole": 4.32, "others": 3.5}            # evidence C section 5
# evidence C section 1 diets (B, P, W) for the members without stomach data used here
EVC = {"long rough dab": (85, 3, 2), "dab": (95, 0, 0), "plaice": (99.5, 0, 0), "witch": (99.6, 0, 0),
       "starry ray": (44, 10, 12), "hagfish": (20, 0, 0), "grey gurnard": (30, 10, 15), "lemon sole": (100, 0, 0),
       "others": (75, 5, 0)}
CONS = {k: BEN_W[k] * QB[k] for k in BEN_W}
CT = sum(CONS.values())


def len_mix(sp, bands):
    e = SL[(SL.species == sp) & (SL.period == "2001-25")].groupby("L_cm")[["J", "A"]].sum().sum(axis=1)
    out = {}
    for lab, lo, hi in bands:
        m = e[(e.index >= lo) & (e.index < hi)]
        mid = 0.5 * (lo + min(hi, 100))
        out[lab] = m.sum() * mid ** -0.6          # consumption per tonne ~ W^-0.2 ~ L^-0.6
    t = sum(out.values())
    return {k: v / t for k, v in out.items()}


CBANDS = [("<30", 15, 30), ("30-39", 30, 40), ("40-49", 40, 50), ("50-69", 50, 70), (">=70", 70, 200)]
HBANDS = [("<30", 10, 30), ("30-39", 30, 40), ("40-49", 40, 50), ("50-69", 50, 70)]
CMIX, HMIX = len_mix("cod", CBANDS), len_mix("haddock", HBANDS)
print("Box cod consumption weights by band: " + ", ".join(f"{k} {v:.3f}" for k, v in CMIX.items()))
print("Box haddock consumption weights:     " + ", ".join(f"{k} {v:.3f}" for k, v in HMIX.items()))
print("Member consumption shares: " + ", ".join(f"{k} {v / CT:.3f}" for k, v in CONS.items()))

# SMS 'Quarter all' tables (tables/sms_diet_pred16.txt / 18): plk, ben, wht, sand, other; herring, sprat, pout
SMS_COD = {"<30": (0.047, 0.016, 0.008, 0.073, 0.856, 0.007, 0.012, 0.028),
           "30-39": (0.149, 0.049, 0.024, 0.086, 0.691, 0.015, 0.011, 0.122),
           "40-49": (0.198, 0.057, 0.067, 0.048, 0.630, 0.051, 0.009, 0.137),
           "50-69": (0.266, 0.063, 0.103, 0.024, 0.544, 0.111, 0.008, 0.147),
           ">=70": (0.197, 0.098, 0.119, 0.010, 0.575, 0.121, 0.005, 0.072)}
SMS_HAD = {"<30": (0.022, 0, 0, 0.134, 0.844, 0, 0, 0.022), "30-39": (0.071, 0, 0, 0.186, 0.743, 0, 0, 0.071),
           "40-49": (0.118, 0, 0, 0.144, 0.738, 0, 0, 0.118), "50-69": (0.138, 0, 0, 0.151, 0.712, 0, 0, 0.138)}
OTHER_BED = 0.85   # [J] share of SMS 'other food' that is bed prey (crustaceans, polychaetes, echinoderms...)


def sms_member(tab, mix):
    p = sum(mix[b] * tab[b][0] for b in mix)
    w = sum(mix[b] * tab[b][2] for b in mix)
    bed = sum(mix[b] * tab[b][4] for b in mix) * OTHER_BED
    sp = {"herring": sum(mix[b] * tab[b][5] for b in mix), "sprat": sum(mix[b] * tab[b][6] for b in mix),
          "Norway pout": sum(mix[b] * tab[b][7] for b in mix)}
    return bed, p, w, sp


def ices_member(pred, areas, mix):
    """box-length-weighted shares from the ICES CSV, pooled over the areas by mass; a band with < 5 fed
    borrows the all-length row of the same areas."""
    d = ICES[(ICES.pred == pred) & (ICES.area.isin(areas)) & (ICES.period == "2000+")]
    allrow = d[d.length == "all"]
    def row(sel):
        g = sel.g_ident.values
        if g.sum() <= 0:
            return None
        cols = ["plk", "own", "epi", "neph", "inf", "plk_Norway_pout", "plk_herring", "plk_sprat", "plk_blue_whiting"]
        return {c: float((sel[c].values * g).sum() / g.sum()) / 100 for c in cols}, int(sel.fed.sum())
    base, nb = row(allrow)
    acc = {k: 0.0 for k in base}
    for b, w in mix.items():
        r = row(d[d.length == b])
        use = r[0] if (r is not None and r[1] >= 5) else base
        for k in acc:
            acc[k] += w * use[k]
    bed = acc["epi"] + acc["neph"] + acc["inf"]
    sp = {"Norway pout": acc["plk_Norway_pout"], "herring": acc["plk_herring"], "sprat": acc["plk_sprat"],
          "blue whiting": acc["plk_blue_whiting"]}
    return bed, acc["plk"], acc["own"], sp, nb


def group(cod, had):
    """consumption-weighted group (bed, P, W) from cod, haddock and evidence C for the rest."""
    bed = CONS["cod"] * cod[0] + CONS["haddock"] * had[0]
    p = CONS["cod"] * cod[1] + CONS["haddock"] * had[1]
    w = CONS["cod"] * cod[2] + CONS["haddock"] * had[2]
    for k, (b_, p_, w_) in EVC.items():
        bed += CONS[k] * b_ / 100
        p += CONS[k] * p_ / 100
        w += CONS[k] * w_ / 100
    return bed / CT, p / CT, w / CT


BED_SPLIT = {"inf": 0.578 / 0.912, "epi": 0.318 / 0.912, "neph": 0.016 / 0.912}


def from_bpw(bed, p, w):
    d = {k: bed * v for k, v in BED_SPLIT.items()}
    d.update(plk=p, wht=w)
    return d


FIELD = {"V1 booked": {"inf": 0.578, "epi": 0.318, "neph": 0.016, "plk": 0.075, "wht": 0.013},
         "V2 223.10": {"inf": 0.544, "epi": 0.299, "neph": 0.015, "plk": 0.120, "wht": 0.023},
         "V3 evidC": from_bpw(0.72, 0.084, 0.019),
         "V3 evidC lowP": from_bpw(0.72, 0.065, 0.019),
         "V3 evidC highP": from_bpw(0.72, 0.147, 0.019)}
SPMIX = {}
c_sms, h_sms = sms_member(SMS_COD, CMIX), sms_member(SMS_HAD, HMIX)
FIELD["V4 SMS"] = from_bpw(*group(c_sms[:3], h_sms[:3]))
tot_sp = {k: CONS["cod"] * c_sms[3][k] + CONS["haddock"] * h_sms[3][k] for k in c_sms[3]}
SPMIX["V4 SMS"] = {k: v / sum(tot_sp.values()) for k, v in tot_sp.items()}
print(f"\nSMS cod at box mix: bed {c_sms[0]:.3f} P {c_sms[1]:.3f} whiting {c_sms[2]:.3f} (P: "
      + ", ".join(f"{k} {v:.3f}" for k, v in c_sms[3].items()) + ")")
print(f"SMS haddock at box mix: bed {h_sms[0]:.3f} P {h_sms[1]:.3f} (pout {h_sms[3]['Norway pout']:.3f})")
for lab, areas in (("V5 ICES Skag>=80m", ["Skag>=80m"]), ("V5 ICES NNS>=100m", ["NNS>=100m"]),
                   ("V5 ICES both", ["Skag>=80m", "NNS>=100m"])):
    c = ices_member("cod", areas, CMIX)
    h = ices_member("haddock", ["NNS>=100m"], HMIX)
    FIELD[lab] = from_bpw(*group(c[:3], h[:3]))
    s = c[3]
    SPMIX[lab] = {k: v / max(sum(s.values()), 1e-9) for k, v in s.items()}
    print(f"{lab}: cod (fed {c[4]}) bed {c[0]:.3f} P {c[1]:.3f} whiting {c[2]:.3f}; P by species "
          + ", ".join(f"{k} {v:.3f}" for k, v in s.items()) + f"; haddock P {h[1]:.3f}")

print("\nField shares of the five checked pairs (renormalised) and clearance relative to infauna / epifauna:")
fc = {}
for n, d in FIELD.items():
    t = sum(d.values())
    sh = {k: d[k] / t for k in P5}
    c = {k: sh[k] / B[k] for k in P5}
    fc[n] = c
    print(f"  {n:18s} " + " ".join(f"{k} {sh[k]:.3f}" for k in P5)
          + f"   c_plk/c_inf {c['plk'] / c['inf']:6.2f}  c_plk/c_epi {c['plk'] / c['epi']:.3f}  c_epi/c_inf {c['epi'] / c['inf']:.2f}")
mc = A_MODEL
print("\nModel a x accessible share (km2/t/tick, per total prey tonne): " + ", ".join(f"{k} {mc[k]:.5f}" for k in P5))
print(f"  model c_plk/c_inf {mc['plk'] / mc['inf']:.3f}, c_plk/c_epi {mc['plk'] / mc['epi']:.4f}, c_epi/c_inf {mc['epi'] / mc['inf']:.2f}")
print("\nModel / field of the planktivore clearance ratio (1 = consistent; < 1 model too low):")
for n, c in fc.items():
    print(f"  vs {n:18s} anchor infauna {mc['plk'] / mc['inf'] / (c['plk'] / c['inf']):6.3f} (x{(c['plk'] / c['inf']) / (mc['plk'] / mc['inf']):5.1f} needed)"
          f"   anchor epifauna {mc['plk'] / mc['epi'] / (c['plk'] / c['epi']):6.3f} (x{(c['plk'] / c['epi']) / (mc['plk'] / mc['epi']):5.1f})")

# ---------------------------------------------------------------- factor decomposition [D]
print("\n=== Factor decomposition of a x f_eff (planktivore pair) ===")
A_ENC = 0.019            # per capable tonne (220, Kd 0.25), includes the vertical overlap (pool annual ~0.13)
PHI = 0.32               # 223.26 gape-scored capable share (group_params.py: 0.311)
F_POOL = feff(SA_PLK, JB_PLK)
cur = 0.0062 * F_POOL
print(f"current: a 0.0062 (= {A_ENC} x phi {PHI}) x f_eff {F_POOL:.3f} = {cur:.5f}")
# (1) gape counted twice: phi's P scores are gape fractions (cod 0.3/0.7/0.9: 'cod 40 cm swallows 96 % of
# pout, 45 % of herring' -> 0.7), and the stage access is a second prey-size window. Replace phi by the
# membership share without gape: cod >= 20 cm (0.314 x 0.982) at 1, haddock >= 20 cm with its own window
# relative to cod's (haddock f_eff 0.077 / cod 0.272; window_by_species_stdout.txt) x a non-gape capability
# c_h, rays + gurnard ~0.024 x 0.5 [J].
for ch, lab in ((1.0, "c_h 1 (SMS: at equal length haddock P share ~0.5 of cod's, its window ~0.3-0.5)"),
                (0.3, "c_h 0.3 (low: 2025 Q1 haddock ate no planktivores)")):
    M = 0.314 * 0.982 + 0.386 * 0.796 * ch * 0.077 / 0.272 + 0.024 * 0.5
    print(f"  (1) membership share {M:.3f} vs phi {PHI} -> x{M / PHI:.2f}   [{lab}]")
M_C = 0.314 * 0.982 + 0.386 * 0.796 * 0.65 * 0.077 / 0.272 + 0.024 * 0.5
k1 = M_C / PHI
# (2) overlap: GOV-revised daytime near-bed shares (window_by_species_stdout.txt): pool sum x1.87
k2 = 0.0748 / 0.0399
# (3) window species weighting by encounter instead of pool biomass: f_eff ENC 0.320 (current F) / 0.297
#     (revised F) vs POOL 0.272
k3 = 0.297 / 0.272
# (4) window ratio: 0.30 (current) ~ length-dependent all-P p95 (f_eff 0.298 ENC'); herring-specific p95
#     0.421 ENC' -> x1.42 high variant
k4c, k4h = 1.0, 0.421 / 0.297
# (5) blue whiting absent from the window table (11 % of the pool, 16 % of encounter, above the window)
k5 = 1 - 0.16 * 0.9
corr = cur * k1 * k2 * k3 * k4c * k5
print(f"  (1) gape double count, central c_h 0.65            x{k1:.2f}")
print(f"  (2) GOV near-bed shares (herring 0.34, sprat 0.5)  x{k2:.2f}")
print(f"  (3) window weighted by encounter, not biomass      x{k3:.2f}")
print(f"  (4) window ratio 0.30 = length-dependent p95       x{k4c:.2f} (herring-specific p95: x{k4h:.2f})")
print(f"  (5) blue whiting outside the window                x{k5:.2f}")
print(f"  corrected a x f_eff = {corr:.5f} (x{corr / cur:.2f}); herring-ratio variant {corr * k4h:.5f} (x{corr * k4h / cur:.2f})")
for n in ("V1 booked", "V3 evidC", "V4 SMS", "V5 ICES both"):
    c = fc[n]
    need = (c["plk"] / c["inf"]) / (mc["plk"] / mc["inf"])
    print(f"  remaining gap vs {n:14s}: x{need / (corr / cur):.1f} (herring-ratio variant x{need / (corr * k4h / cur):.1f})")

# ---------------------------------------------------------------- per-species test
print("\n=== Per-species clearance relative to infauna (field vs model) ===")
BSP = {"herring": 2.60, "Norway pout": 0.25, "blue whiting": 0.35, "sprat": 0.10}   # params_planktivorous_fish.md
FB_CUR = {"herring": 0.0279, "Norway pout": 0.1440, "blue whiting": 0.0607, "sprat": 0.0184}
FB_REV = {"herring": 0.0621, "Norway pout": 0.1440, "blue whiting": 0.1148, "sprat": 0.0905}
# per-species window at the species' own J share, cod ratio 0.30 (window_by_species_stdout.txt)
FW = {"herring": 0.54 * 0.463 + 0.46 * 0.102, "Norway pout": 0.73 * 0.716 + 0.27 * 0.329,
      "sprat": 0.36 * 0.845 + 0.64 * 0.643, "blue whiting": 0.02}
S_POOL = {"herring": 0.788, "Norway pout": 0.076, "blue whiting": 0.106, "sprat": 0.030}
SUMC = sum(S_POOL[s] * FB_CUR[s] for s in S_POOL)
SUMR = sum(S_POOL[s] * FB_REV[s] for s in S_POOL)
A_NEW = A_ENC * k2 * M_C       # proposed a per whole tonne (membership share, revised overlap)
K5 = k5
print(f"proposed a = {A_ENC} x overlap x{k2:.2f} x membership {M_C:.3f} = {A_NEW:.4f}; per-species model current -> proposed")
for lab, mix in list(SPMIX.items()) + [("Skag 2023-25 (evid. C)", {"Norway pout": 0.83, "herring": 0.03, "sprat": 0.13,
                                                                    "blue whiting": 0.0})]:
    t = sum(mix.values())
    out = []
    base = FIELD.get(lab, FIELD["V3 evidC"])
    tt = sum(base.values())
    cinf = base["inf"] / tt / B["inf"]
    for s in ("Norway pout", "herring", "sprat", "blue whiting"):
        dj = base["plk"] / tt * mix.get(s, 0) / t
        cf = dj / BSP[s] / cinf
        cm = 0.0062 * FB_CUR[s] / SUMC * FW[s] / mc["inf"]
        cr = A_NEW * FB_REV[s] / SUMR * FW[s] * K5 / mc["inf"]
        out.append(f"{s} field {cf:6.2f} model {cm:5.2f} -> {cr:5.2f} ratio {cm / cf if cf > 0 else float('inf'):6.3f}"
                   f" -> {cr / cf if cf > 0 else float('inf'):6.3f}")
    print(f"  {lab:22s} " + " | ".join(out))
print("  (model per species: a x Fbar_j / sum s Fbar x window_j at the species' J share (proposed: x0.86 for the"
      " blue whiting share); field: the variant's"
      " P share x species split / species B; pout B 0.25 t/km2)")
for bp in (0.25, 0.6, 1.3):
    s = dict(S_POOL)
    print(f"  pout B {bp:.2f} t/km2: field pout clearance x{0.25 / bp:.2f} of the 0.25 case")

# ---------------------------------------------------------------- concurrent-search and time-allocation readings
print("\n=== Concurrent-search diet shares a f B / sum (all prey visible) ===")
H = {"inf": 112.0, "epi": 258.0, "neph": 286.0, "plk": 238.2, "wht": 177.0}
MODELS = {"current": dict(A_MODEL), "corrected (x%.2f)" % (corr / cur): dict(A_MODEL, plk=corr),
          "corrected, herring ratio": dict(A_MODEL, plk=corr * k4h)}
for n, m in MODELS.items():
    for vis in (1.0, 0.25):
        bb = dict(B, plk=B["plk"] * vis)
        tot = sum(m[k] * bb[k] for k in P5)
        print(f"  {n:26s} plk visible {vis:4.2f}: " + " ".join(f"{k} {m[k] * bb[k] / tot:.3f}" for k in P5))
for n in ("V1 booked", "V3 evidC", "V4 SMS", "V5 ICES both"):
    t = sum(FIELD[n].values())
    print(f"  field {n:20s}            " + " ".join(f"{k} {FIELD[n][k] / t:.3f}" for k in P5))
print("\nSingle-pair return x/(1+x), x = a f h B (share of the pair's own ceiling); ceilings MJ/t/tick: "
      "infauna 23.2, planktivores ~28 (E/h)")
for n, m in MODELS.items():
    out = []
    for k in ("inf", "epi", "plk"):
        for vis in ((1.0, 0.25) if k == "plk" else (1.0,)):
            x = m[k] * H[k] * B[k] * vis
            out.append(f"{k}{'' if vis == 1 else '@0.25'} x {x:6.2f} -> {x / (1 + x):.2f}")
    print(f"  {n:26s} " + "; ".join(out))
print("Time-allocation check (not a derivation): for free allocation to put any time on fish, the fish pair must pay"
      " about what the bed pays per allocated tick. Bed: 0.99 x 23.2 = 23.0 MJ/t/tick; planktivore ceiling ~28.")
for r in (0.6, 0.82):
    x = r / (1 - r)
    print(f"  return {r:.2f} of the planktivore ceiling: x {x:.2f} -> a f = {x / (H['plk'] * B['plk']):.5f} (all visible),"
          f" {x / (H['plk'] * B['plk'] * 0.25):.5f} (25 % visible); current {cur:.5f}")

# ---------------------------------------------------------------- pout biomass convention, model AND field side
print("\n=== Pout biomass scenarios (pool composition; q = 1 GOV/acoustic 0.25 vs Walker-q corrected) ===")
print("Model side: s_pout rises, so the pool sum s_j Fbar_j and the encounter-weighted window rise with it;"
      " field side: the pool B rises (c_plk = D/B falls). Proposed a and window otherwise.")
B_OTH = {"herring": 2.60, "blue whiting": 0.35, "sprat": 0.10}


def fw_mix(s):
    w = {k: s[k] * FB_REV[k] for k in s}
    return sum(w[k] * FW[k] for k in s) / sum(w.values())


S0 = dict(S_POOL)
FW0 = fw_mix(S0)
F_PROP = 0.297 * K5     # proposed f_eff (ENC', blue whiting share removed)
for bp in (0.25, 0.6, 1.3):
    bpool = sum(B_OTH.values()) + bp
    s = {k: v / bpool for k, v in B_OTH.items()}
    s["Norway pout"] = bp / bpool
    sumr = sum(s[k] * FB_REV[k] for k in s)
    a_b = A_ENC * sumr / SUMC * M_C
    f_b = F_PROP * fw_mix(s) / FW0
    af = a_b * f_b
    m = dict(A_MODEL, plk=af)
    bb = dict(B, plk=bpool)
    tot = sum(m[k] * bb[k] for k in P5)
    out = []
    for n in ("V1 booked", "V3 evidC", "V4 SMS", "V5 ICES both"):
        t = sum(FIELD[n].values())
        cfield = (FIELD[n]["plk"] / t / bpool) / (FIELD[n]["inf"] / t / B["inf"])
        out.append(f"{n} x{cfield / (af / mc['inf']):.1f}")
    print(f"  pout {bp:.2f} t/km2 (pool {bpool:.2f}, s_pout {s['Norway pout']:.3f}): a {a_b:.4f} x f_eff {f_b:.3f} = {af:.5f}"
          f" (x{af / cur:.2f} vs current); concurrent plk share {m['plk'] * bb['plk'] / tot:.3f}"
          f" (booked share at this pool B: M2 x B = {0.145 * bpool:.2f} t/km2/yr -> see note); remaining gap "
          + ", ".join(out))
print("  Note: with a larger pout B the booked M2 0.145 on the pool would be re-derived on the new pool B; the"
      " gap above keeps the field DIET shares fixed, so it is the clearance gap, not the M2 gap.")
