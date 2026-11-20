"""Benthivores: the Norway pout term against the bed terms (no model output read).

Usage:
  python3 -I pout_vs_bed.py <benth_encounter.py> <length_biomass_longrun.csv> <depth.npz> <fg_library.yaml>

  <benth_encounter.py>: research_notes/"Attackrater ur motesteori" (o-umlaut)/scripts/benth_encounter.py.
      It is executed with runpy (stdout discarded, n_mc 300, Kd top 0.25 = the library's section 220
      value) and its own functions and tables are reused: the light series at the grid's bottom, the
      reactive-distance light factor, the Gerritsen-Strickler relative speed, the per-1-cm-class
      numbers per tonne, and the per-member benthic a_enc.
  <length_biomass_longrun.csv>, <depth.npz>: that script's own two inputs.
  <fg_library.yaml>: the live library; the benthivore pair values (a, h, accessible_fraction,
      stage_access, predation_mortality) are read from it and printed, so the numbers below are tied
      to the library they were computed on.

Sections:
  A. The gate's lambda = a x B_vis per pair (B and B_vis as budget_gate.py prints them) and the
     factor the planktivore term is short.
  B. The pout term rebuilt per species and per member (cod >= 20 cm, haddock >= 20 cm), F day/night,
     species windows, light; the species-resolved planktivore lambda vs the engine's pooled one.
  C. One-at-a-time sensitivity of the pout term; vertical-geometry and school variants.
  D. Member-resolved diet (each member saturated on its own gut) vs the pooled gate.
  E. Bed side: the lambda band from the exposure note's MC 5-95 %, lab anchors, double-count audit
     numbers.
  F. Norway pout biomass conventions: pool composition, model lambda, implied pout M2.
  G. Scenarios: gate shares and single-pair saturation; Charnov inclusion thresholds.
"""
import contextlib
import io
import math
import runpy
import sys

import numpy as np
import yaml

ENC, LBCSV, DEPTH, LIB = sys.argv[1:5]

# ---------------------------------------------------------------------------------------------
# run the encounter script quietly and reuse its namespace
# ---------------------------------------------------------------------------------------------
_argv = sys.argv
sys.argv = [ENC, LBCSV, DEPTH, "300", "0.25"]
with contextlib.redirect_stdout(io.StringIO()):
    NS = runpy.run_path(ENC, run_name="benth_encounter")
sys.argv = _argv

CL = NS["CL"]
light_factor = NS["light_factor"]
vrel = NS["vrel"]
day_prey = NS["day_prey"]
TICK = NS["TICK"]
U_PREY = NS["U_PREY"]
per_sp = NS["per_sp"]          # benthic a_enc per species (central), weights in the pair
GROUP = NS["GROUP"]

lib = yaml.safe_load(open(LIB, encoding="utf-8"))
inter = lib.get("interaction_definitions") or {}
if not inter:
    for v in lib.values():
        if isinstance(v, dict) and "benthivorous_fish_preys_on_planktivorous_fish" in v:
            inter = v
            break


def pair(prey):
    return inter[f"benthivorous_fish_preys_on_{prey}"]


print("Library values read (benthivorous_fish_preys_on_*):")
for prey in ("benthic_infauna", "mobile_epifauna", "norway_lobster", "planktivorous_fish", "whiting"):
    p = pair(prey)
    print(f"  {prey:<20} a {p.get('max_intake_rate')}  h {p.get('handling_time')}  "
          f"f {p.get('accessible_fraction', '-')}  stage_access {p.get('stage_access', '-')}  "
          f"M2 {p.get('predation_mortality')}")

# =============================================================================================
# A. gate lambdas  [D from library + budget_gate.py densities]
# =============================================================================================
# B and B_vis from budget_gate.py (2026-11-06 run; tables/budget_gate_stdout.txt)
GATE = {  # prey: (B, B_vis)
    "benthic_infauna": (93.61, 18.722),
    "mobile_epifauna": (3.78, 2.342),
    "norway_lobster": (4.58, 0.092),
    "planktivorous_fish": (3.33, 0.862),
    "whiting": (0.45, 0.038),
}
BOOKED = {"benthic_infauna": 0.578, "mobile_epifauna": 0.318, "norway_lobster": 0.016,
          "planktivorous_fish": 0.075, "whiting": 0.013}
A = {k: float(pair(k)["max_intake_rate"]) for k in GATE}
H = {k: float(pair(k)["handling_time"]) for k in GATE}
LAM = {k: A[k] * GATE[k][1] for k in GATE}
print()
print("=" * 100)
print("A. Gate lambda = a x B_vis (per tick) and shares")
print("=" * 100)
tot = sum(LAM.values())
for k, v in LAM.items():
    print(f"  {k:<20} a {A[k]:.4f} B_vis {GATE[k][1]:7.3f} lambda {v:.5f} share {v / tot:.3f} "
          f"booked {BOOKED[k]:.3f}")
lam_bed = sum(LAM[k] for k in ("benthic_infauna", "mobile_epifauna", "norway_lobster"))
X_bed = sum(H[k] * LAM[k] for k in ("benthic_infauna", "mobile_epifauna", "norway_lobster"))
print(f"  lambda_bed {lam_bed:.4f}; shared denominator X_bed = sum h lambda = {X_bed:.1f}")


def plk_needed(share, lam_rest):
    return share * lam_rest / (1 - share)


lam_rest = tot - LAM["planktivorous_fish"]
for s in (0.075, 0.102, 0.123):
    need = plk_needed(s, lam_rest)
    print(f"  planktivore share {s:.3f} needs lambda_plk {need:.4f} = x{need / LAM['planktivorous_fish']:.1f} "
          f"the current {LAM['planktivorous_fish']:.5f}")

# =============================================================================================
# B. pout term per species and member
# =============================================================================================
POOL_S = {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030}  # [F] pool shares
B_POOL = 3.33
# band shares (day, night): herring reconciled 0.16 (Bottenfiskens sillbyte reconcile_summary.md),
# the others benthivores_encounter.md section 2.3 (J)
FBAND = {"herring": (0.16, 0.02), "blue whiting": (0.3, 0.15), "Norway pout": (0.7, 0.4),
         "sprat": (0.1, 0.01)}
# species window shares at cod / haddock ratio 0.3 (f_J, f_A, J share) [D: Bottenfiskens sillbyte
# tables/window_by_species_stdout.txt]; blue whiting above the cod window (p10 17.5 cm) except
# cod >= ~60 cm: 0.05 J for cod, 0 for haddock
WIN_COD = {"herring": (0.463, 0.102, 0.54), "Norway pout": (0.716, 0.329, 0.73),
           "sprat": (0.845, 0.643, 0.36), "blue whiting": (0.05, 0.05, 0.0)}
WIN_HAD = {"herring": (0.136, 0.003, 0.54), "Norway pout": (0.468, 0.038, 0.73),
           "sprat": (0.698, 0.319, 0.36), "blue whiting": (0.0, 0.0, 0.0)}
WIN_COD_R04 = {"herring": (0.722, 0.341, 0.54), "Norway pout": (0.903, 0.613, 0.73),
               "sprat": (0.974, 0.858, 0.36), "blue whiting": (0.10, 0.10, 0.0)}


def win(tab, sp):
    fj, fa, j = tab[sp]
    return j * fj + (1 - j) * fa


def member_base(sp, lmin_cm=20.0, rs=2.0, v=0.4, p=0.3, hb=10.0):
    """km2 per tonne (of the member's >= lmin share) per tick per unit of g x F, at R_sat."""
    d = CL[sp]
    d = d[d.L_cm >= lmin_cm]
    s = d.s / d.s.sum()
    n = s * 1e6 / d.w_g
    L = d.L.values
    vr = np.array([vrel(x, U_PREY) for x in v * L])
    vol = math.pi * (rs * L) ** 2 * vr * p * TICK
    return float((n * vol).sum()) / hb / 1e6


def gF(fday, fnight, rnv=0.6, rs=2.0, ke=0.02, prof="central"):
    g = light_factor(rnv / rs, ke, prof)
    return float((g * np.where(day_prey, fday, fnight)).mean())


# group biomass shares of cod / haddock >= 20 cm [D from the length table, as Bottenfiskens sillbyte]
def share_ge(sp, lmin=20.0):
    d = CL[sp]
    return float(d[d.L_cm >= lmin].s.sum())


W_COD = GROUP["cod"] / 100 * share_ge("cod")
W_HAD = GROUP["haddock"] / 100 * share_ge("haddock")
C_H = 0.65  # haddock non-gape capability on fish prey, central (0.3-1) [D/J, Bottenfiskens sillbyte]

print()
print("=" * 100)
print("B. Species-resolved planktivore term (per WHOLE benthivore tonne)")
print("=" * 100)
print(f"  group shares >= 20 cm: cod {W_COD:.3f}, haddock {W_HAD:.3f}; haddock capability c_h {C_H}")
base_cod, base_had = member_base("cod"), member_base("haddock")
print(f"  same-layer base at R_sat (F = 1, g = 1): cod {base_cod:.3f}, haddock {base_had:.3f} km2/t/tick")


def lam_species(sp, B_sp, fband=None, base_c=None, base_h=None, c_h=C_H, wc=WIN_COD, wh=WIN_HAD,
                gf=None):
    fd, fn = fband or FBAND[sp]
    g = gf if gf is not None else gF(fd, fn)
    bc = base_cod if base_c is None else base_c
    bh = base_had if base_h is None else base_h
    a_c = W_COD * bc * g * win(wc, sp)
    a_h = W_HAD * c_h * bh * g * win(wh, sp)
    return (a_c + a_h) * B_sp, a_c * B_sp, a_h * B_sp


rows = {}
for sp, s in POOL_S.items():
    lt, lc, lh = lam_species(sp, s * B_POOL)
    rows[sp] = lt
    print(f"  {sp:<13} B {s * B_POOL:5.3f}  gF {gF(*FBAND[sp]):.4f}  window cod {win(WIN_COD, sp):.3f} "
          f"haddock {win(WIN_HAD, sp):.3f}  lambda {lt:.5f} (cod {lc:.5f}, haddock {lh:.5f})")
lam_res = sum(rows.values())
print(f"  species-resolved lambda_plk {lam_res:.5f} vs engine {LAM['planktivorous_fish']:.5f} "
      f"(x{lam_res / LAM['planktivorous_fish']:.2f}); pout share of it {rows['Norway pout'] / lam_res:.2f}")
LAM_POUT0 = rows["Norway pout"]
for s in (0.075, 0.102, 0.123):
    need = plk_needed(s, lam_rest) * 0.9  # 83-99 % of the field take is pout: 0.9 central
    print(f"  field share {s:.3f}, 90 % pout: lambda_pout needed {need:.4f} = x{need / LAM_POUT0:.1f} "
          f"the rebuilt pout term")

# =============================================================================================
# C. sensitivity of the pout term
# =============================================================================================
print()
print("=" * 100)
print("C. Pout term, one factor at a time (ratio to central)")
print("=" * 100)
Bp = POOL_S["Norway pout"] * B_POOL
c0 = LAM_POUT0


def rel(**kw):
    return lam_species("Norway pout", Bp, **kw)[0] / c0


tests = [
    ("R_sat 1.5 L", dict(base_c=member_base("cod", rs=1.5), base_h=member_base("haddock", rs=1.5),
                         gf=gF(0.7, 0.4, rs=1.5))),
    ("R_sat 3 L", dict(base_c=member_base("cod", rs=3.0), base_h=member_base("haddock", rs=3.0),
                       gf=gF(0.7, 0.4, rs=3.0))),
    ("R_nv 0.3 L", dict(gf=gF(0.7, 0.4, rnv=0.3))),
    ("R_nv 1.0 L", dict(gf=gF(0.7, 0.4, rnv=1.0))),
    ("Ke 0.005 lx", dict(gf=gF(0.7, 0.4, ke=0.005))),
    ("Ke 0.1 lx", dict(gf=gF(0.7, 0.4, ke=0.1))),
    ("light low K (clear)", dict(gf=gF(0.7, 0.4, prof="lowK"))),
    ("light high K (turbid)", dict(gf=gF(0.7, 0.4, prof="highK"))),
    ("v 0.2 BL/s", dict(base_c=member_base("cod", v=0.2), base_h=member_base("haddock", v=0.2))),
    ("v 0.8 BL/s", dict(base_c=member_base("cod", v=0.8), base_h=member_base("haddock", v=0.8))),
    ("p 0.1", dict(base_c=member_base("cod", p=0.1), base_h=member_base("haddock", p=0.1))),
    ("p 0.5 (Juanes et al. 2002: ~half of piscivore attacks)",
     dict(base_c=member_base("cod", p=0.5), base_h=member_base("haddock", p=0.5))),
    ("p 0.6", dict(base_c=member_base("cod", p=0.6), base_h=member_base("haddock", p=0.6))),
    ("H 5 m", dict(base_c=member_base("cod", hb=5), base_h=member_base("haddock", hb=5))),
    ("H 20 m", dict(base_c=member_base("cod", hb=20), base_h=member_base("haddock", hb=20))),
    ("F pout 0.4 / 0.1", dict(fband=(0.4, 0.1))),
    ("F pout 0.9 / 0.7", dict(fband=(0.9, 0.7))),
    ("c_h 0.3", dict(c_h=0.3)),
    ("c_h 1.0", dict(c_h=1.0)),
    ("window ratio 0.4 (herring-specific p95 band)", dict(wc=WIN_COD_R04)),
]
for lab, kw in tests:
    print(f"  {lab:<52} x{rel(**kw):.2f}")
# all favourable at once (corner, not a plausible combination)
corner = lam_species("Norway pout", Bp, base_c=member_base("cod", rs=3.0, v=0.8, p=0.6, hb=5),
                     base_h=member_base("haddock", rs=3.0, v=0.8, p=0.6, hb=5),
                     gf=gF(0.9, 0.7, rs=3.0, rnv=1.0, ke=0.005, prof="lowK"), c_h=1.0,
                     wc=WIN_COD_R04)[0] / c0
print(f"  all-high corner (R_sat 3, R_nv 1, Ke .005, low K, v .8, p .6, H 5, F .9/.7, c_h 1, ratio .4) x{corner:.0f}")
# MC: log-uniform on the same ranges, independent
rng = np.random.default_rng(20261106)
n = 400
mc = np.empty(n)
for i in range(n):
    rs = math.exp(rng.uniform(math.log(1.5), math.log(3.0)))
    rnv = math.exp(rng.uniform(math.log(0.3), math.log(1.0)))
    ke = math.exp(rng.uniform(math.log(0.005), math.log(0.1)))
    prof = rng.choice(["lowK", "central", "highK"])
    v = math.exp(rng.uniform(math.log(0.2), math.log(0.8)))
    p = math.exp(rng.uniform(math.log(0.1), math.log(0.6)))
    hb = math.exp(rng.uniform(math.log(5), math.log(20)))
    fd = math.exp(rng.uniform(math.log(0.4), math.log(0.9)))
    fn = math.exp(rng.uniform(math.log(0.1), math.log(0.7)))
    ch = math.exp(rng.uniform(math.log(0.3), math.log(1.0)))
    # base scales as rs^2 v p / hb (v_rel ~ v for v >> u)
    sc = rs ** 2 / 4.0 * v / 0.4 * p / 0.3 * 10.0 / hb
    mc[i] = lam_species("Norway pout", Bp, base_c=base_cod * sc, base_h=base_had * sc,
                        gf=gF(fd, fn, rnv=rnv, rs=rs, ke=ke, prof=prof), c_h=ch)[0] / c0
print(f"  MC (n {n}, log-uniform, independent) ratio to central: 5/50/95 % "
      f"{np.percentile(mc, 5):.2f} {np.percentile(mc, 50):.2f} {np.percentile(mc, 95):.2f}")

# vertical geometry: predator inside 0-2 m of the bed, pout 5-10 m up
print()
print("  Vertical geometry variant (structural, J): cod/haddock search within ~2 m of the bed")
print("  (Stensholt & Stensholt 2004: 19-34 % of tag depths within 2 m of the daily maximum depth),")
print("  pout layer 5-10 m above the bed (Albert 1994; Aglen 1996: best trawl/acoustic fit at 4 m).")
for R in (0.4, 0.8, 1.4):
    # vertical reach of a predator at height z_p in [0, 2] m with reaction radius R: [z_p - R, z_p + R]
    # pout uniform over [3, 10] m (day) -> overlap of [0, 2 + R] with [3, 10]
    reach_top = 2.0 + R
    ov = max(0.0, min(reach_top, 10.0) - 3.0) / 7.0
    print(f"    R {R:.1f} m: share of the pout layer within reach {ov:.2f} (band model assumes F_day 0.7 in 10 m)")

# school dilution: per-encounter yield vs the ration
print()
print("  School variant (structural): a 15 cm pout weighs ~25 g [J, L-W]; a 2 kg cod's ration at the")
r_cod = 2000 * 0.0089  # ration t/t/tick from budget_gate x body mass
print(f"  group ration 0.0089 t/t/tick is {r_cod:.0f} g per 6 h = {r_cod / 25:.1f} pout per tick, so a school")
print("  encounter yields at most ~1 prey; with the school's own radius r_s the encounter rate falls by")
for rs_, nn in ((3, 100), (5, 1000)):
    for R in (0.5, 1.0):
        print(f"    r_s {rs_} m, n {nn}, R {R} m: (R + r_s)^2 / (n R^2) = {(R + rs_) ** 2 / (nn * R * R):.3f}")
print("  i.e. schooling lowers, never raises, a gut-limited cod's pout encounter (convention: left to the")
print("  prey's hide action / visibility floor, Attackrater ur motesteori addendum_validation_behaviour.md).")

# =============================================================================================
# D. member-resolved diet vs pooled gate
# =============================================================================================
print()
print("=" * 100)
print("D. Member-resolved diet (each member saturated on its own gut) vs the pooled gate")
print("=" * 100)
# benthic a_enc per member (central) from the encounter script; e by mode on infauna (A-weighted
# takeable / p, exposure_e_items.csv: gadoid 0.216, flatfish 0.131 x width 0.80), epifauna e per member
# (exposure_e_central_stdout.txt)
E_INF = {"gadoid": 0.216, "flat": 0.131 * 0.80, "lrd": 0.131 * 0.80, "gurnard": 0.216, "ray": 0.216}
E_EPI = {"haddock": 0.48, "cod": 0.72, "long rough dab": 0.35 * 0.92, "dab": 0.35 * 0.92,
         "plaice": 0.35 * 0.92, "witch": 0.35 * 0.92, "lemon sole": 0.35 * 0.92, "starry ray": 0.62,
         "grey gurnard": 0.56}
SP_MODE = NS["SP_MODE"]
aenc = {sp: v[0] / v[4] for sp, v in per_sp.items()}
wts = {sp: v[4] for sp, v in per_sp.items()}
inf_raw = {sp: aenc[sp] * E_INF[SP_MODE[sp]] for sp in aenc}
epi_raw = {sp: aenc[sp] * E_EPI[sp] for sp in aenc}
k_inf = A["benthic_infauna"] / sum(wts[s] * inf_raw[s] for s in aenc)
k_epi = A["mobile_epifauna"] / sum(wts[s] * epi_raw[s] for s in aenc)
# consumption shares (Bottenfiskens sillbyte clearance_test stdout), renormalised over the 9 members
CONS = {"haddock": 0.419, "cod": 0.287, "long rough dab": 0.122, "dab": 0.045, "plaice": 0.037,
        "witch": 0.015, "starry ray": 0.007, "grey gurnard": 0.008, "lemon sole": 0.011}
cs = sum(CONS.values())
lam_p_cod = lam_species("Norway pout", Bp)[1]
plk_cod = sum(lam_species(sp, s * B_POOL)[1] for sp, s in POOL_S.items()) / W_COD   # per cod tonne
plk_had = sum(lam_species(sp, s * B_POOL)[2] for sp, s in POOL_S.items()) / W_HAD   # per haddock tonne


def member_diet(scale_plk=1.0, scale_bed=1.0):
    out = 0.0
    pooled_p = pooled_t = 0.0
    for sp in aenc:
        lb = scale_bed * (k_inf * inf_raw[sp] * GATE["benthic_infauna"][1]
                          + k_epi * epi_raw[sp] * GATE["mobile_epifauna"][1]
                          + (LAM["norway_lobster"] if sp in ("cod", "haddock") else 0.0))
        lp = scale_plk * ({"cod": plk_cod * share_ge("cod"), "haddock": plk_had * share_ge("haddock")}
                          .get(sp, 0.0))
        out += CONS[sp] / cs * lp / (lp + lb)
        w = GROUP[sp]
        pooled_p += w * lp
        pooled_t += w * (lp + lb)
    return out, pooled_p / pooled_t


md, pooled = member_diet()
print(f"  infauna a_eff per member (km2/t/tick): " + ", ".join(f"{s} {k_inf * inf_raw[s]:.3f}" for s in aenc))
print(f"  planktivore lambda per cod tonne {plk_cod * share_ge('cod'):.4f}, per haddock tonne "
      f"{plk_had * share_ge('haddock'):.4f}")
print(f"  planktivore share: member-resolved (consumption-weighted) {md:.4f}; pooled (gate form) {pooled:.4f}; "
      f"ratio x{md / pooled:.2f}")
cod_lb = (k_inf * inf_raw['cod'] * GATE['benthic_infauna'][1] + k_epi * epi_raw['cod'] * GATE['mobile_epifauna'][1]
          + LAM['norway_lobster'])
cod_lp = plk_cod * share_ge('cod')
print(f"  cod alone: planktivore share {cod_lp / (cod_lp + cod_lb):.3f} (field, box-length weighted: 0.19 Skagerrak "
      f">= 80 m, 0.54 NNS >= 100 m, SMS box mix 0.176)")

# =============================================================================================
# E. bed side band and lab anchors
# =============================================================================================
print()
print("=" * 100)
print("E. Bed side: lambda band and lab anchors")
print("=" * 100)
for lab, ai, ae in (("MC 5 %", 0.016, 0.05), ("central", A["benthic_infauna"], A["mobile_epifauna"]),
                    ("MC 95 %", 0.092, 0.53)):
    lb = ai * GATE["benthic_infauna"][1] + ae * GATE["mobile_epifauna"][1] + LAM["norway_lobster"]
    print(f"  {lab:<8} a_inf {ai:.3f} a_epi {ae:.3f}: lambda_bed {lb:.3f} (x{lam_bed / lb:.2f} vs central)")
# Persson et al. 2012 (MEPS 456:245) Type II a for ~18 cm (55 g) juvenile cod on sand, cm2/s
for prey, a_cm2s in (("Crangon (buries)", 1.8), ("Palaemon", 21.0), ("Neomysis", 57.0)):
    a_fish = a_cm2s * 1e-4 * TICK                # m2 per fish per tick
    a_t = a_fish / 55e-6 / 1e6                   # km2 per t per tick (1 m2/g = 1 km2/t)
    # per-tonne 2-D a ~ L^-1 (area/fish ~ L^2, mass ~ L^3); box gadoid biomass-weighted L ~ 40 cm (J)
    print(f"  Persson 2012 {prey:<17} {a_t:6.3f} km2/t/tick at 18 cm; scaled to ~40 cm (x18/40): {a_t * 18 / 40:.3f}")
print(f"  library epifauna: a {A['mobile_epifauna']} per accessible tonne; x f 0.62 = "
      f"{A['mobile_epifauna'] * 0.62:.3f} per total tonne")
print(f"  library infauna: a {A['benthic_infauna']} per exposed tonne; x 0.2 = {A['benthic_infauna'] * 0.2:.4f} per total tonne")

# =============================================================================================
# F. pout biomass conventions
# =============================================================================================
print()
print("=" * 100)
print("F. Norway pout biomass: conventions and consequences")
print("=" * 100)
OTHER = B_POOL * (1 - POOL_S["Norway pout"])
BEN_FLOW = 0.145 * B_POOL          # booked benthivore -> planktivore flow, t/km2/yr [F library]
WHT_FLOW = 0.07 * B_POOL           # booked whiting -> planktivore flow (232.1) [F library]
POUT_FR_BEN = (0.83, 0.99)         # pout share of the benthivores' planktivore take in-area [F/D]
POUT_FR_WHT = 0.61                 # SMS whiting planktivore take that is pout [S, Bottenfiskens sillbyte]
CONV = [("raw GOV q = 1 / acoustic (library)", 0.25),
        ("Walker q, Q1 (0.58)", 0.58),
        ("Walker q, Q1/Q3 mean", 0.0069 * (83.47 + 390.90) / 2),
        ("Walker q, Q3 (0-group pulse)", 2.70),
        ("stock TSB 2024-25 / 1-2e5 km2", 0.45),
        ("stock TSB 2016-24 / 1-2e5 km2", 2.0)]
SMS_M2_BEN = 0.29   # SMS 2016-22 pout M2 by benthivore-group predators [D, evidence_pout_bluewhiting.md 2.2]
SMS_M2_TOT = 1.89
for lab, b in CONV:
    m2b = (BEN_FLOW * POUT_FR_BEN[0] / b, BEN_FLOW * POUT_FR_BEN[1] / b)
    m2w = WHT_FLOW * POUT_FR_WHT / b
    print(f"  {lab:<38} pout {b:5.2f} t/km2  pool {OTHER + b:5.2f}  pout share {b / (OTHER + b):.3f}  "
          f"M2 benthivores {m2b[0]:.2f}-{m2b[1]:.2f} (SMS {SMS_M2_BEN})  + whiting {m2w:.2f}  "
          f"(SMS total {SMS_M2_TOT})")
b_match = (BEN_FLOW * POUT_FR_BEN[0] / SMS_M2_BEN, BEN_FLOW * POUT_FR_BEN[1] / SMS_M2_BEN)
print(f"  pout B at which the booked benthivore take equals SMS's benthivore M2 0.29: {b_match[0]:.2f}-{b_match[1]:.2f} t/km2")
print("  (caveat: SMS M2 is stock-wide; the box's cod:pout ratio need not equal the stock's)")

# =============================================================================================
# G. scenarios
# =============================================================================================
print()
print("=" * 100)
print("G. Scenarios: gate shares (a x B_vis), single-pair saturation, Charnov thresholds")
print("=" * 100)
# pooled-window covariance: engine lambda_plk vs species-resolved (section B)
F_RES = lam_res / LAM["planktivorous_fish"]
print(f"  species-resolution factor (engine pooled a x f_eff vs species sum) x{F_RES:.2f}")


def lam_plk_at(pout_b, scale=1.0):
    """species-resolved planktivore lambda with pout at pout_b, others as now."""
    t = 0.0
    for sp, s in POOL_S.items():
        b = pout_b if sp == "Norway pout" else s * B_POOL
        t += lam_species(sp, b)[0]
    return t * scale


def shares(lp, bed_scale=1.0):
    lam = dict(LAM)
    for k in ("benthic_infauna", "mobile_epifauna"):
        lam[k] = LAM[k] * bed_scale
    lam["planktivorous_fish"] = lp
    t = sum(lam.values())
    return {k: v / t for k, v in lam.items()}, lam


def show(lab, lp, bed_scale=1.0, plk_bvis=None):
    sh, lam = shares(lp, bed_scale)
    sat = {k: H[k] * lam[k] / (1 + H[k] * lam[k]) for k in lam}
    print(f"  {lab:<54} " + " ".join(f"{k[:6]} {sh[k]:.3f}" for k in sh)
          + f" | sat inf {sat['benthic_infauna']:.2f} epi {sat['mobile_epifauna']:.2f} "
            f"neph {sat['norway_lobster']:.2f} plk {sat['planktivorous_fish']:.2f}")
    return sh


print("  shares: infauna, epifauna, Nephrops, planktivores, whiting | single-pair sat = h lam/(1 + h lam)")
show("current library", LAM["planktivorous_fish"])
show("species-resolved, pout 0.25", lam_plk_at(0.25))
SC = {}
for lab, b in (("0.58", 0.58), ("1.64", 0.0069 * (83.47 + 390.90) / 2), ("2.70", 2.70)):
    SC[lab] = show(f"species-resolved, pout {lab}", lam_plk_at(b))
show("species-resolved, pout 1.64, p 0.5 (Juanes)", lam_plk_at(1.64) * 0.5 / 0.3)
show("species-resolved, pout 1.64, bed at MC 5 %", lam_plk_at(1.64),
     bed_scale=(0.016 * 18.722 + 0.05 * 2.342) / (LAM["benthic_infauna"] + LAM["mobile_epifauna"]))
show("species-resolved, pout 2.70, bed at MC 5 %", lam_plk_at(2.70),
     bed_scale=(0.016 * 18.722 + 0.05 * 2.342) / (LAM["benthic_infauna"] + LAM["mobile_epifauna"]))
print("  field: booked inf 0.578 epi 0.318 neph 0.016 plk 0.075 wht 0.013; in-area plk 0.08-0.17")

# engine a that reproduces the species-resolved lambda at the current stage_access
sa = pair("planktivorous_fish")["stage_access"]
js = (GATE["planktivorous_fish"][1] / GATE["planktivorous_fish"][0] - sa[1]) / (sa[0] - sa[1])
print(f"\n  stage_access {sa}, implied J/B {js:.3f}")
for lab, b in (("0.25", 0.25), ("0.58", 0.58), ("1.64", 1.64), ("2.70", 2.70)):
    pool_b = OTHER + b
    # J/B of the pool rises with pout (pout J share 0.73 vs pool 0.38); keep stage_access, recompute J/B
    js_new = (js * B_POOL - 0.73 * POOL_S["Norway pout"] * B_POOL + 0.73 * b) / pool_b
    fe = js_new * sa[0] + (1 - js_new) * sa[1]
    a_eq = lam_plk_at(b) / (fe * pool_b)
    print(f"  pout {lab}: pool {pool_b:.2f}, J/B {js_new:.3f}, f_eff {fe:.3f}, B_vis {fe * pool_b:.3f}; "
          f"engine a equivalent {a_eq:.4f} (library {A['planktivorous_fish']})")

# Charnov inclusion (energy kept per gut time = (assim - sda) E/h, h = E/ceiling_j)
print()
E = {"benthic_infauna": 2600, "mobile_epifauna": 3750, "norway_lobster": 3600, "planktivorous_fish": 4000 + 0.5 * 6250}
ASS = {"benthic_infauna": 0.7, "mobile_epifauna": 0.8, "norway_lobster": 0.8, "planktivorous_fish": 0.8}
SDA = 0.10
val = {k: (ASS[k] - SDA) * E[k] / H[k] for k in E}
print("  energy kept per gut time (assim - sda) E / h, MJ/t/tick: " + ", ".join(f"{k} {v:.1f}" for k, v in val.items()))
best = max(val, key=val.get)
print(f"  top-ranked prey per gut time: {best} -> always attacked by a rate maximiser (zero-one rule)")
for k in ("benthic_infauna", "mobile_epifauna", "norway_lobster"):
    pass
for k in ("mobile_epifauna", "norway_lobster"):
    r = val[k] / val["benthic_infauna"]
    xstar = r / (1 - r)
    print(f"  {k}: included only while the infauna x = h lam < {xstar:.1f} "
          f"(now {H['benthic_infauna'] * LAM['benthic_infauna']:.0f}; ignores the planktivore term)")
print("  (planktivore E as eaten = lean 4000 + 0.5 x reserve 6250 = 7125: library, prey_includes_reserve)")
