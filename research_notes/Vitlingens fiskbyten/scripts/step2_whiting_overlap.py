"""whiting -> planktivorous_fish under the LAYER and the OVERLAP (band) conventions, per member.

Usage: python3 -I step2_whiting_overlap.py <depth.npz> <stage_length_biomass.csv> <pred_length_biomass.csv> [n_mc]
  <depth.npz>: maps/depth.npz (cell depths weight the light at the bed).
  <stage_length_biomass.csv>: ../Arsungarnas andel (A-ring)/tables/ (box GOV stage lengths).
  <pred_length_biomass.csv>: ../Fiskgrupper i rutan/predators/tables/ (box whiting lengths).

No model output is read. Two conventions for the same adult whiting (section 220 central factors as
in scripts/planktivore_access.py: L 0.2516 m, 7330 fish per t, v 0.8 BL/s, p 0.2, scan 0.5):

LAYER (library, 223.26): the foraging whiting is inside each prey's layer:
    a_j = 0.5 a3d(R_day, H_day,j) + 0.5 a3d(R_night, H_night,j)
    herring/sprat: R 0.5 m in a 70 m day layer, 0.2 m in a 40 m night layer; pout R 0.2 m in 15 m;
    blue whiting R 0.2 m in 50 m. Prey density met = B / H_j.
OVERLAP (band; the convention the benthivores' fish pairs already use, Attackrater (o-umlaut)
benthivores_encounter.md 2.3): the whiting searches its own band of thickness H_W above the bed;
prey density met = B F_j(t) / H_W, F_j = share of member j's biomass inside the band (day / night,
switch at sun elevation 0). The reactive distance follows the light AT THE BED:
    R(t)^2 = R_sat^2 [q + (1 - q) E_b/(E_b + Ke)],  q = (R_nv / R_sat)^2,
E_b from the surface illuminance (cloud factor 0.5) through three Kd profiles to each cell's depth
(same construction as benth_encounter.py). The quantity that differs between the conventions is
the prey concentration per metre at the whiting's height, c_j = F_j / H_W vs 1 / H_j, and R.

Inputs (central, low-high) and tags:
  c_j day / night per metre of band (F_j over a 10 m band; benthivores_encounter.md 2.3 J values,
  herring: the benthivore note's F 0.15 kept as central [J]; tables/step2_heras_bottom_share_stdout.txt gives
  F(bottom 10 m, day) 0.66 / 0.10 / 0.04 / 0.06 over <80 / 80-110 / 110-130 / 130-150 m beds (unmeasured >= 150 m,
  47 % of cells): equal-cell weighting ~0.10, weighting by herring backscatter ~0.31, so 0.15 sits between):
    herring  day 0.015 (0.004-0.034)  night 0.002 (0.0005-0.005)   [D/J]
    sprat    day 0.010 (0.002-0.03)   night 0.001 (0.0005-0.005)   [J]
    pout     day 0.07 (0.03-0.09)     night 0.04 (0.01-0.07)       [J; Albert 1994 5-10 m above bed]
    blue wh. day 0.03 (0.01-0.06)     night 0.015 (0.005-0.04)     [J]
  R_sat 0.5 m (0.38-0.75 = 1.5-3 L) [J, the layer's day R]; R_nv 0.2 m (0.1-0.3) [J, the layer's
  night R]; Ke 0.02 lx (0.005-0.1) [J, gadoid vision note]; Kd profile low / central / high [D/J].
  FOLLOW variant: a share phi of the whiting rises into the clupeids' night layer (layer geometry,
  R_nv) at night: phi 0.3 (0-0.6) [J; Onsrud et al. 2004, Pedersen 2000: pelagic whiting feed at night].
Night/day ratio for the engine: mean a with sun > -5 deg (day) vs below (night), as in the library.
Then: the stage access [f_J, f_A] (window 0.39 x length over whiting >= 20 cm, species weighted by
encounter = variant A+B of planktivore_access.py) with each convention's member a, and the implied
species mix of the planktivore take.
"""
import math
import sys

import numpy as np
import pandas as pd

DEPTH = np.load(sys.argv[1])["depth"].ravel()
SL = pd.read_csv(sys.argv[2])
PL = pd.read_csv(sys.argv[3])
NMC = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
RNG = np.random.default_rng(20261031)
T = 21600.0
LAT = 58.15

# ------------------------------------------------------------------ predator (adult whiting, 220 central)
L, NT = 0.2516, 7330.39
V = 0.8 * L
P_CAP, SCAN = 0.2, 0.5
ADULT_SHARE = 0.753
U = {"herring": 0.2, "sprat": 0.2, "Norway pout": 0.1, "blue whiting": 0.1}
POOL = {"herring": 0.79, "blue whiting": 0.105, "Norway pout": 0.075, "sprat": 0.03}


def area(R, H):
    h = H / 2.0
    return math.pi * R * R if R <= h else 2.0 * (R * R * math.asin(h / R) + h * math.sqrt(R * R - h * h))


def vrel(v, u):
    return (u * u + 3 * v * v) / (3 * v) if v >= u else (v * v + 3 * u * u) / (3 * u)


def a3d(R, u, H):
    return SCAN * area(R, H) * vrel(V, u) * T * NT * P_CAP / H / 1e6


LAYER = {
    "herring": 0.5 * a3d(min(0.5, 2 * L), 0.2, 70.0) + 0.5 * a3d(0.2, 0.2, 40.0),
    "sprat": 0.5 * a3d(min(0.5, 2 * L), 0.2, 70.0) + 0.5 * a3d(0.2, 0.2, 40.0),
    "Norway pout": a3d(0.2, 0.1, 15.0),
    "blue whiting": a3d(0.2, 0.1, 50.0),
}
LAYER_C = {"herring": (1 / 70, 1 / 40), "sprat": (1 / 70, 1 / 40), "Norway pout": (1 / 15, 1 / 15),
           "blue whiting": (1 / 50, 1 / 50)}
LAYER_R = {"herring": (0.5, 0.2), "sprat": (0.5, 0.2), "Norway pout": (0.2, 0.2), "blue whiting": (0.2, 0.2)}

# ------------------------------------------------------------------ light at the bed
K_PROF = {"lowK": (0.20, 0.08, 0.04), "central": (0.30, 0.12, 0.06), "highK": (0.45, 0.15, 0.08)}


def surface_lux(h):
    pts = [(-18, 6e-4), (-12, 8e-3), (-6, 3.0), (0, 500.0), (5, 1.3e5 * math.sin(math.radians(5)))]
    if h >= 5:
        return 1.3e5 * math.sin(math.radians(h))
    if h <= -18:
        return 6e-4
    for (h0, e0), (h1, e1) in zip(pts[:-1], pts[1:]):
        if h0 <= h <= h1:
            f = (h - h0) / (h1 - h0)
            return math.exp(math.log(e0) + f * (math.log(e1) - math.log(e0)))
    return 6e-4


def transmission(z, prof):
    k1, k2, k3 = K_PROF[prof]
    t = math.exp(-k1 * min(z, 20.0))
    if z > 20:
        t *= math.exp(-k2 * (min(z, 50.0) - 20.0))
    if z > 50:
        t *= math.exp(-k3 * (z - 50.0))
    return t


def elevations():
    out = []
    for day in range(1, 366, 2):
        dec = math.radians(23.44) * math.sin(2 * math.pi * (284 + day) / 365.0)
        for hr in np.arange(0.25, 24, 0.5):
            ha = math.radians(15 * (hr - 12))
            p = math.radians(LAT)
            s = math.sin(p) * math.sin(dec) + math.cos(p) * math.cos(dec) * math.cos(ha)
            out.append(math.degrees(math.asin(max(-1, min(1, s)))))
    return np.array(out)


ELEV = elevations()
E0 = np.array([surface_lux(h) for h in ELEV]) * 0.5
ZB = np.percentile(DEPTH[np.isfinite(DEPTH)], np.linspace(1, 99, 50))
TR = {p: np.array([transmission(z, p) for z in ZB]) for p in K_PROF}
DAY_PREY = ELEV > 0.0
DAY_ENG = ELEV > -5.0


def light_g(q, ke, prof):
    Eb = np.outer(E0, TR[prof])
    return (q + (1 - q) * Eb / (Eb + ke)).mean(axis=1)


C_BAND = {  # (day (lo, c, hi), night (lo, c, hi)) per metre of band
    "herring": ((0.004, 0.015, 0.034), (0.0005, 0.002, 0.005)),
    "sprat": ((0.002, 0.010, 0.03), (0.0005, 0.001, 0.005)),
    "Norway pout": ((0.03, 0.07, 0.09), (0.01, 0.04, 0.07)),
    "blue whiting": ((0.01, 0.03, 0.06), (0.005, 0.015, 0.04)),
}
R_SAT, R_NV, KE, PHI = (0.38, 0.5, 0.75), (0.1, 0.2, 0.3), (0.005, 0.02, 0.1), (0.0, 0.3, 0.6)


def overlap_series(sp, rsat, rnv, ke, prof, cday, cnight, phi):
    """a(t) per unit prey biomass of member sp (km2/t/tick) along the year's elevation series."""
    g = light_g(min(rnv / rsat, 1.0) ** 2, ke, prof)
    c = np.where(DAY_PREY, cday, cnight)
    base = SCAN * math.pi * rsat ** 2 * vrel(V, U[sp]) * T * NT * P_CAP / 1e6
    band = base * g * c
    if sp in ("herring", "sprat") and phi > 0:
        night_layer = SCAN * math.pi * rnv ** 2 * vrel(V, U[sp]) * T * NT * P_CAP / 1e6 / 40.0
        band = np.where(DAY_PREY, band, (1 - phi) * band + phi * night_layer)
    return band


def layer_series(sp):
    """the layer variant along the same series (day = sun > 0), for its engine night/day ratio."""
    (cd, cn), (rd, rn) = LAYER_C[sp], LAYER_R[sp]
    base = SCAN * math.pi * vrel(V, U[sp]) * T * NT * P_CAP / 1e6
    return np.where(DAY_PREY, base * rd ** 2 * cd, base * rn ** 2 * cn)


print("=" * 104)
print("1. Prey concentration per metre at the whiting's height: overlap c_j = F_j / H_W vs layer 1 / H_j")
print("=" * 104)
for sp in POOL:
    (dl, dc, dh), (nl, nc, nh) = C_BAND[sp]
    ld, ln = LAYER_C[sp]
    print(f"  {sp:12s} day {dc:.4f} vs {ld:.4f} (x{dc / ld:5.2f}; {dl / ld:.2f}-{dh / ld:.2f})   "
          f"night {nc:.4f} vs {ln:.4f} (x{nc / ln:5.2f}; {nl / ln:.2f}-{nh / ln:.2f})")
g_c = light_g((0.2 / 0.5) ** 2, 0.02, "central")
print(f"  bed light: mean R^2/R_sat^2 by day (sun > 0) {g_c[DAY_PREY].mean():.3f}, night {g_c[~DAY_PREY].mean():.3f} "
      f"(q = {(0.2 / 0.5) ** 2:.2f}); day fraction of the year {DAY_PREY.mean():.3f}")
for prof in K_PROF:
    g = light_g(0.16, 0.02, prof)
    print(f"    Kd {prof:7s}: day {g[DAY_PREY].mean():.3f}, night {g[~DAY_PREY].mean():.3f}")

print()
print("=" * 104)
print("2. Member a per unit member biomass (km2/t/tick, annual mean), adult whiting")
print("=" * 104)
res = {}
hdr = (f"  {'member':12s} {'layer':>8s} {'overlap':>8s} {'x':>5s} {'MC 5-50-95 %':>24s} {'follow':>8s} {'x':>5s}"
       f"   night/day (thr -5): layer / overlap / follow")
print(hdr)
mc_store = {}
for sp in POOL:
    (dl, dc, dh), (nl, nc, nh) = C_BAND[sp]
    s_ov = overlap_series(sp, R_SAT[1], R_NV[1], KE[1], "central", dc, nc, 0.0)
    s_fo = overlap_series(sp, R_SAT[1], R_NV[1], KE[1], "central", dc, nc, PHI[1])
    s_la = layer_series(sp)
    a_la = LAYER[sp]
    a_ov, a_fo = s_ov.mean(), s_fo.mean()

    def ratio(s):
        return s[~DAY_ENG].mean() / s[DAY_ENG].mean()

    def lu(lo, hi, n):
        return np.exp(RNG.uniform(np.log(lo), np.log(hi), n))

    n = NMC
    rs, rn, ke = lu(R_SAT[0], R_SAT[2], n), lu(R_NV[0], R_NV[2], n), lu(KE[0], KE[2], n)
    cdv, cnv = lu(dl, dh, n), lu(nl, nh, n)
    ph = RNG.uniform(PHI[0], PHI[2], n)
    profs = RNG.choice(list(K_PROF), n)
    mc = np.empty(n)
    mcf = np.empty(n)
    for i in range(n):
        mc[i] = overlap_series(sp, rs[i], rn[i], ke[i], profs[i], cdv[i], cnv[i], 0.0).mean()
        mcf[i] = overlap_series(sp, rs[i], rn[i], ke[i], profs[i], cdv[i], cnv[i], ph[i]).mean()
    mc_store[sp] = (mc, mcf)
    q = np.percentile(mc, [5, 50, 95])
    res[sp] = dict(layer=a_la, ov=a_ov, fo=a_fo, r_la=ratio(s_la), r_ov=ratio(s_ov), r_fo=ratio(s_fo))
    print(f"  {sp:12s} {a_la:8.4f} {a_ov:8.4f} {a_ov / a_la:5.2f} {q[0]:8.4f}{q[1]:8.4f}{q[2]:8.4f} "
          f"{a_fo:8.4f} {a_fo / a_la:5.2f}   {ratio(s_la):.2f} / {ratio(s_ov):.2f} / {ratio(s_fo):.2f}")
print(f"  (layer herring/sprat check: {LAYER['herring']:.4f} = planktivore_access.py 0.0304)")

print()
print("=" * 104)
print("3. Pool a (pool weights) per adult tonne and per whole whiting tonne (x adult share 0.753)")
print("=" * 104)
for key, lab in (("layer", "layer (library)"), ("ov", "overlap, static band"), ("fo", "overlap + follow phi 0.3")):
    a_pool = sum(POOL[s] * res[s][key] for s in POOL)
    contrib = ", ".join(f"{s} {POOL[s] * res[s][key] / a_pool:.2f}" for s in POOL)
    # engine night/day ratio of the pool a
    if key == "layer":
        ser = sum(POOL[s] * layer_series(s) for s in POOL)
    else:
        ser = sum(POOL[s] * overlap_series(s, R_SAT[1], R_NV[1], KE[1], "central", C_BAND[s][0][1], C_BAND[s][1][1],
                                           PHI[1] if key == "fo" else 0.0) for s in POOL)
    rr = ser[~DAY_ENG].mean() / ser[DAY_ENG].mean()
    r0 = ser[~DAY_PREY].mean() / ser[DAY_PREY].mean()
    print(f"  {lab:28s} adult {a_pool:.4f}  whole group {ADULT_SHARE * a_pool:.4f}  night/day {rr:.2f} (thr -5) / {r0:.2f} (thr 0)   "
          f"share of a: {contrib}")
mc_pool = sum(POOL[s] * mc_store[s][0] for s in POOL) * ADULT_SHARE
mc_poolf = sum(POOL[s] * mc_store[s][1] for s in POOL) * ADULT_SHARE
print(f"  MC (independent draws per member) whole group, static band: 5/50/95 % "
      f"{np.percentile(mc_pool, 5):.4f} {np.percentile(mc_pool, 50):.4f} {np.percentile(mc_pool, 95):.4f}; "
      f"follow: {np.percentile(mc_poolf, 5):.4f} {np.percentile(mc_poolf, 50):.4f} {np.percentile(mc_poolf, 95):.4f}")

# ------------------------------------------------------------------ stage access with each convention
POOL_W = {1: {"herring": 0.88, "blue whiting": 0.0, "Norway pout": 0.11, "sprat": 0.01},
          2: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030},
          3: {"herring": 0.78, "blue whiting": 0.15, "Norway pout": 0.04, "sprat": 0.03},
          4: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030}}
PERIOD = {1: "2001-25", 2: "1991-96", 3: "2001-25", 4: "1991-96"}
JB = 0.38
wl = PL[PL.common == "whiting"]
mids = wl.L5.values + 2.5
wts = wl.pooled.values / wl.pooled.sum()
m20 = mids >= 20
Ls, ws = mids[m20], wts[m20] / wts[m20].sum()


def share_inside(sp, q, stage, ratio):
    if sp == "blue whiting":
        return 0.05 if stage == "J" else 0.0
    d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
    if d.empty or d[stage].sum() <= 0:
        return np.nan
    Lp = d.L_cm.values + 0.5
    vv = d[stage].values / d[stage].sum()
    return float(sum(w * vv[Lp <= ratio * Lw].sum() for Lw, w in zip(Ls, ws)))


def access(aunit, ratio):
    out = {}
    for stage in ("J", "A"):
        fq = []
        for q in (1, 2, 3, 4):
            num = den = 0.0
            for sp, w in POOL_W[q].items():
                if w <= 0:
                    continue
                f = share_inside(sp, q, stage, ratio)
                if np.isnan(f):
                    continue
                num += w * aunit[sp] * f
                den += w * aunit[sp]
            fq.append(num / den)
        out[stage] = float(np.mean(fq))
    return out


def take_mix(aunit, ratio):
    tot = {s: 0.0 for s in POOL}
    for q in (1, 2, 3, 4):
        for sp, w in POOL_W[q].items():
            if w <= 0:
                continue
            d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
            jb = (d.J.sum() / (d.J.sum() + d.A.sum())) if not d.empty else 0.5
            fJ, fA = share_inside(sp, q, "J", ratio), share_inside(sp, q, "A", ratio)
            fJ = 0.0 if np.isnan(fJ) else fJ
            fA = 0.0 if np.isnan(fA) else fA
            tot[sp] += w * aunit[sp] * (jb * fJ + (1 - jb) * fA)
    s = sum(tot.values())
    return {k: v / s for k, v in tot.items()}, s


print()
print("=" * 104)
print("4. Stage access (window over whiting >= 20 cm, encounter-weighted members) and the implied take mix")
print("=" * 104)
print("   field, planktivore-pool prey in whiting stomachs: SMS North Sea pout 0.61 / sprat 0.26 / herring 0.12;")
print("   Skagerrak Apr 1991 (in the box's rectangles) pout only; Kattegat 1991 sprat 0.46 / herring 0.43 / pout 0.10")
for key, lab in (("layer", "layer"), ("ov", "overlap"), ("fo", "overlap+follow")):
    au = {s: res[s][key] for s in POOL}
    a_pool = ADULT_SHARE * sum(POOL[s] * au[s] for s in POOL)
    for ratio in (0.39, 0.45):
        acc = access(au, ratio)
        eff = JB * acc["J"] + (1 - JB) * acc["A"]
        mix, _ = take_mix(au, ratio)
        print(f"  {lab:15s} ratio {ratio}: a {a_pool:.4f}  stage_access [{acc['J']:.3f}, {acc['A']:.4f}] eff {eff:.4f}"
              f"  a x eff {a_pool * eff:.5f}   take: " + ", ".join(f"{k} {v:.2f}" for k, v in mix.items()))
