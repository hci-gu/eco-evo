"""Holling attack rates of benthivorous_fish from encounter theory (no model output used).

Usage: python3 -I benth_encounter.py <length_biomass_longrun.csv> <depth.npz> [n_mc] [kd_top]

  [kd_top]: Kd (1/m) of the top 20 m in the central light profile (default 0.30, the Aas 2014
      Secchi regression; addendum_vision_light.md: Arst 1993 gives ~0.20 offshore in summer).

  <length_biomass_longrun.csv>: ../../Fiskgrupper i rutan/benthivores/tables/ (NS-IBTS GOV Q1+Q3,
      2001-25, 1-cm classes, biomass share and class weight w_g per species).
  <depth.npz>: the repo's maps/depth.npz (mean depth per 1-km cell), used only to weight light
      at the bottom by cell depth.

Engine definition (lib/environments/ecosystem_env/interactions.holling_a_eff, predation.py):
intake per t predator per 6-h tick = min(eat, phi) x hunger x a B / (1 + a h B + w B_pred).
So a [km2 t-1 tick-1] = area effectively searched per tonne of FORAGING predator (of the share
phi that can take the prey) per tick.  1 m2 per g = 1 km2 per t.

Benthic pair (2-D):   area per fish per tick = v * w * p * 21600 s
    v = search speed (BL/s x L), w = search-path width (x L), p = capture per encounter
    (biomass-weighted over accessible infauna 0.92 / epifauna 0.08 of the accessible stock).
Fish pairs (3-D -> 2-D): area per fish per tick = pi R^2 * v_rel * p * 21600 * sum_j s_j F_j/H
    R = reactive distance (light dependent), v_rel = Gerritsen & Strickler (1977) mean relative
    speed, s_j = share of prey species j in the pool, F_j = share of j's biomass inside the
    predator's near-bottom search band of thickness H (day / night).
Per tonne: sum over 1-cm classes of share_L * 1e6 / W_L * area(L) / 1e6 (km2/t), every length
scaled quantity proportional to L.  Weights across species and stages: group biomass share x
stage share x the pair's predator_share score (group_params.py SC), renormalised, so a is per
tonne of the share inside phi.

Ranges: every factor has low / central / high (sources and judgements in
benthivores_encounter.md).  central = product of centrals; the range is the 5-95 % of a
Monte Carlo with each factor log-uniform between its low and high (independent), and the
all-low / all-high corners are printed too.
"""
import math
import sys

import numpy as np
import pandas as pd

LB = pd.read_csv(sys.argv[1])
DEPTH = np.load(sys.argv[2])["depth"].ravel()
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
KD_TOP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
RNG = np.random.default_rng(20261025)
TICK = 21600.0

# group biomass shares (% of group, long-run 2001-25, tables/long_run_species_by_group.csv)
GROUP = {"haddock": 38.62, "cod": 31.38, "long rough dab": 12.63, "dab": 4.49, "plaice": 3.80,
         "witch": 1.67, "starry ray": 1.47, "grey gurnard": 0.89, "lemon sole": 0.87}
# not in the length table (hagfish 1.27, weever 0.78, others 2.5 %): left out, weights renormalised
CUTS = {"cod": [30, 50], "haddock": [40], "long rough dab": [20], "dab": [20], "plaice": [25],
        "witch": [30], "lemon sole": [25], "grey gurnard": [25], "starry ray": [30]}
# predator_share stage scores, central (Fiskgrupper i rutan/benthivores/scripts/group_params.py)
SCORE = {
    "B": {"cod": [1, .72, .64], "haddock": [1, 1]},  # all others 1
    "P": {"cod": [.3, .7, .9], "haddock": [.1, .25], "long rough dab": [.05, .05],
          "starry ray": [.4, .4], "grey gurnard": [.15, .15]},  # others 0 (flatfish) / 0.05
    "W": {"cod": [0, .15, .4], "haddock": [0, 0], "starry ray": [.3, .3],
          "grey gurnard": [.3, .3]},
}
FLAT = {"dab", "plaice", "witch", "lemon sole"}


def score(pair, sp, k):
    s = SCORE[pair].get(sp)
    if s is not None:
        return s[k]
    if pair == "B":
        return 1.0
    if pair == "P":
        return 0.0 if sp in FLAT else 0.05
    return 0.0


def classes(sp):
    d = LB[(LB.species == sp) & (LB.share_2001_25 > 0)].copy()
    d["s"] = d.share_2001_25 / d.share_2001_25.sum()
    d["L"] = (d.L_cm + 0.5) / 100.0
    edges = [0] + CUTS[sp] + [1000]
    d["stage"] = np.searchsorted(np.array(edges[1:-1]), d.L_cm.values, side="right")
    d["n_per_t"] = d.s * 1e6 / d.w_g
    return d


CL = {sp: classes(sp) for sp in GROUP}


def pair_weights(pair):
    """weight of each (species, class) row: group share x class share x score, renormalised."""
    rows = []
    for sp, d in CL.items():
        for st, dd in d.groupby("stage"):
            sc = score(pair, sp, st)
            if sc > 0:
                rows.append((sp, st, GROUP[sp] * dd.s.sum() * sc))
    tot = sum(r[2] for r in rows)
    return {(sp, st): w / tot for sp, st, w in rows}


def per_tonne_moment(sp, stage, power):
    """sum over the stage's classes of (class share within stage) x n/t x L^power."""
    d = CL[sp][CL[sp].stage == stage]
    s = d.s / d.s.sum()
    return float((s * 1e6 / d.w_g * d.L ** power).sum())


def logu(lo, hi, n):
    return np.exp(RNG.uniform(math.log(lo), math.log(hi), n))


def draw(spec, n):
    """spec = (low, central, high) -> (central, MC samples, low, high)."""
    lo, c, hi = spec
    return c, logu(lo, hi, n), lo, hi


def summarise(name, central, mc, lo, hi, h=None, B=None, extra=""):
    p5, p50, p95 = np.percentile(mc, [5, 50, 95])
    line = (f"{name:<38} central {central:9.4f}  MC 5/50/95 % {p5:9.4f} {p50:9.4f} {p95:9.4f}"
            f"  corners {lo:9.5f} - {hi:8.3f}")
    if h:
        line += f"\n{'':<38} 1/(a h) [t/km2]: central {1 / (central * h):8.3f}  (5-95 %: " \
                f"{1 / (p95 * h):7.3f} - {1 / (p5 * h):8.3f})"
        if B:
            for lab, b in B:
                x = central * h * b
                xl, xh = p5 * h * b, p95 * h * b
                line += f"\n{'':<38} at {lab} = {b:6.2f} t/km2: x = a h B {x:7.2f} -> " \
                        f"{x / (1 + x):.3f} of ceiling (5-95 %: {xl / (1 + xl):.3f} - {xh / (1 + xh):.3f})"
    print(line + extra)
    return p5, p50, p95


# ----------------------------------------------------------------------------------------------
# 1. Benthic pair
# ----------------------------------------------------------------------------------------------
# feeding modes: v (BL/s), w (x L), p_infauna, p_epifauna  -- (low, central, high)
MODE_B = {
    "gadoid": dict(v=(0.2, 0.4, 0.8), w=(0.07, 0.15, 0.4), pi=(0.25, 0.5, 0.9), pe=(0.1, 0.3, 0.7)),
    "flat": dict(v=(0.04, 0.12, 0.3), w=(0.25, 0.6, 1.2), pi=(0.4, 0.7, 1.0), pe=(0.15, 0.4, 0.8)),
    "lrd": dict(v=(0.04, 0.12, 0.3), w=(0.25, 0.6, 1.2), pi=(0.25, 0.5, 0.9), pe=(0.15, 0.4, 0.8)),
    "gurnard": dict(v=(0.1, 0.2, 0.4), w=(0.15, 0.3, 0.6), pi=(0.25, 0.5, 0.9), pe=(0.1, 0.3, 0.7)),
    "ray": dict(v=(0.05, 0.15, 0.3), w=(0.2, 0.4, 0.8), pi=(0.25, 0.5, 0.9), pe=(0.1, 0.3, 0.7)),
}
SP_MODE = {"haddock": "gadoid", "cod": "gadoid", "dab": "flat", "plaice": "flat", "witch": "flat",
           "lemon sole": "flat", "long rough dab": "lrd", "grey gurnard": "gurnard",
           "starry ray": "ray"}
EPI = (0.04, 0.08, 0.15)  # epifauna share of the ACCESSIBLE benthos biomass (1.6 of 20.3 t/km2)
H_BENTH, B_BENTH, F_ACC = 110.7, 101.6, 0.2

print("=" * 100)
print("1. benthivorous_fish -> benthic_community (2-D swept area)")
print("=" * 100)
wB = pair_weights("B")
epi_c, epi_mc, _, _ = draw(EPI, NMC)
mode_draws = {}
for m, spec in MODE_B.items():
    mode_draws[m] = {k: draw(v, NMC) for k, v in spec.items()}
tot_c = np.zeros(1)
tot_mc = np.zeros(NMC)
tot_lo = tot_hi = 0.0
per_sp = {}
for (sp, st), wt in wB.items():
    md = mode_draws[SP_MODE[sp]]
    M2 = per_tonne_moment(sp, st, 2)  # area ~ (v_BL L)(w_rel L)
    pc = (1 - epi_c) * md["pi"][0] + epi_c * md["pe"][0]
    pm = (1 - epi_mc) * md["pi"][1] + epi_mc * md["pe"][1]
    c = md["v"][0] * md["w"][0] * pc * TICK * M2 / 1e6
    mc = md["v"][1] * md["w"][1] * pm * TICK * M2 / 1e6
    lo = md["v"][2] * md["w"][2] * min(md["pi"][2], md["pe"][2]) * TICK * M2 / 1e6
    hi = md["v"][3] * md["w"][3] * md["pi"][3] * TICK * M2 / 1e6
    acc = per_sp.setdefault(sp, [0.0, np.zeros(NMC), 0.0, 0.0, 0.0])
    acc[0] += wt * c
    acc[1] = acc[1] + wt * mc
    acc[2] += wt * lo
    acc[3] += wt * hi
    acc[4] += wt
    tot_c = tot_c + wt * c
    tot_mc = tot_mc + wt * mc
    tot_lo += wt * lo
    tot_hi += wt * hi
print(f"{'species (weight in pair)':<38} a [km2 t-1 tick-1] per tonne of that species' share")
for sp, (c, mc, lo, hi, wt) in sorted(per_sp.items(), key=lambda kv: -kv[1][4]):
    d = CL[sp]
    print(f"  {sp:<20} w {wt:5.3f}  N/t {d.n_per_t.sum():8.0f}  ", end="")
    summarise("", c / wt, mc / wt, lo / wt, hi / wt)
print()
B_acc = F_ACC * B_BENTH
aB = summarise("GROUP a (benthos)", float(tot_c[0]), tot_mc, tot_lo, tot_hi, h=H_BENTH,
               B=[("accessible f x B", B_acc), ("accessible at 10 % of today's stock", 0.1 * B_acc),
                  ("accessible at 1 % of today's stock", 0.01 * B_acc)])
print(f"  half-saturation in TOTAL benthos units = 1/(a h f): central "
      f"{1 / (float(tot_c[0]) * H_BENTH * F_ACC):.2f} t/km2 (today's stock {B_BENTH})")
gadoid_w = sum(wt for (sp, st), wt in wB.items() if SP_MODE[sp] == "gadoid")
flat_w = sum(wt for (sp, st), wt in wB.items() if SP_MODE[sp] in ("flat", "lrd"))
print(f"  weights in the pair: gadoid chemo-tactile {gadoid_w:.3f}, visual flatfish (incl. LRD) "
      f"{flat_w:.3f}, gurnard + ray {1 - gadoid_w - flat_w:.3f}")
# share of the pair's a contributed by each mode (central)
contrib = {}
for (sp, st), wt in wB.items():
    md = MODE_B[SP_MODE[sp]]
    pc = (1 - EPI[1]) * md["pi"][1] + EPI[1] * md["pe"][1]
    contrib[SP_MODE[sp]] = contrib.get(SP_MODE[sp], 0) + wt * md["v"][1] * md["w"][1] * pc * TICK \
        * per_tonne_moment(sp, st, 2) / 1e6
print("  share of central a by mode: " + ", ".join(f"{k} {v / float(tot_c[0]):.2f}"
                                                   for k, v in contrib.items()))
# night/day for the benthic pair: gadoids ~1 (or more), visual flatfish day feeders
ND_B = {"gadoid": (1.0, 1.0, 1.5), "flat": (0.2, 0.5, 1.0), "lrd": (0.5, 0.8, 1.0),
        "gurnard": (0.8, 1.0, 1.2), "ray": (0.8, 1.0, 1.5)}
for k in (0, 1, 2):
    num = sum(contrib[m] * ND_B[m][k] for m in contrib)
    print(f"  night/day ratio of the pair's a ({['low', 'central', 'high'][k]} mode ratios): "
          f"{num / float(tot_c[0]):.2f}")

# ----------------------------------------------------------------------------------------------
# 2. Light at the bottom (for the fish pairs)
# ----------------------------------------------------------------------------------------------
LAT = 58.1
K_PROF = {"lowK": (0.20, 0.08, 0.04), "central": (KD_TOP, 0.12, 0.06), "highK": (0.45, 0.15, 0.08)}


def surface_lux(h):
    """clear-sky illuminance vs sun elevation (same construction as light_at_depth.py)."""
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
E0 = np.array([surface_lux(h) for h in ELEV]) * 0.5  # cloud factor 0.5 (overcast 0.1-0.3, clear 1)
ZB = np.percentile(DEPTH, np.linspace(1, 99, 50))  # 50 depth quantiles of the grid cells
print()
print("=" * 100)
print("2. Light at the bottom (lux), grid cells weighted equally, cloud factor 0.5")
print("=" * 100)
for prof in K_PROF:
    tr = np.array([transmission(z, prof) for z in ZB])
    Eb = np.outer(E0, tr)
    for lab, mask in [("noon June (elev > 50)", ELEV > 50), ("noon Dec (8 < elev < 9)", (ELEV > 8) & (ELEV < 9)),
                      ("all daylight (elev > 0)", ELEV > 0), ("night (elev < -12)", ELEV < -12)]:
        q = np.percentile(Eb[mask], [10, 50, 90])
        print(f"  {prof:<8} {lab:<26} bottom lux p10/p50/p90 {q[0]:9.2e} {q[1]:9.2e} {q[2]:9.2e}")

# ----------------------------------------------------------------------------------------------
# 3. Fish pairs
# ----------------------------------------------------------------------------------------------
# predator factors (low, central, high)
R_SAT = (1.5, 2.0, 3.0)    # reactive distance to fish prey in saturating light, x L (Lindholm 1999 >= 1.5 L)
R_NV = (0.3, 0.6, 1.0)     # non-visual (lateral line / olfaction) reactive distance, x L  [J]
KE = (0.005, 0.02, 0.1)    # half-saturation light for R^2 (lux at the eye)          [J, vision note]
V_PRED = (0.2, 0.4, 0.8)   # predator search speed, BL/s (Winger 2004 modes and means)
P_CAP = (0.1, 0.3, 0.6)    # capture per encounter                                     [J]
H_BAND = (5.0, 10.0, 20.0)  # thickness of the predator's near-bottom search band, m
U_PREY = 0.10              # prey cruising speed, m/s (12-25 cm fish at ~0.5-1 BL/s)  [J]

# prey pools: share s_j and share of j's biomass inside the bottom band (day, night),
# each (low, central, high)
POOL = {
    "P": {  # planktivorous_fish (params_planktivorous_fish.md weights)
        "herring": (0.788, (0.05, 0.15, 0.34), (0.005, 0.02, 0.05)),
        "blue whiting": (0.106, (0.1, 0.3, 0.6), (0.05, 0.15, 0.4)),
        "Norway pout": (0.076, (0.4, 0.7, 0.9), (0.1, 0.4, 0.7)),
        "sprat": (0.030, (0.02, 0.1, 0.3), (0.005, 0.01, 0.05)),
    },
    "W": {  # predatory_fish (params_predatory_fish.md weights, rest lumped as pelagic)
        "saithe": (0.409, (0.05, 0.15, 0.3), (0.03, 0.1, 0.2)),
        "whiting": (0.372, (0.2, 0.5, 0.8), (0.05, 0.2, 0.5)),
        "mackerel + horse mackerel": (0.115, (0.01, 0.05, 0.15), (0.005, 0.02, 0.1)),
        "spurdog, hake, other demersal": (0.104, (0.1, 0.3, 0.6), (0.05, 0.2, 0.4)),
    },
}
H_PAIR = {"P": 262.9, "W": 193.8}
B_PAIR = {"P": 3.33, "W": 1.20}
THR = -5.0  # engine threshold for the dark ratio on the fish pair

tr_c = {prof: np.array([transmission(z, prof) for z in ZB]) for prof in K_PROF}
day_prey = ELEV > 0.0      # prey day/night switch at sunset (HERAS twilight logs: herring off bottom)
day_eng = ELEV > THR


def light_factor(rnv_over_rsat, ke, prof):
    """mean over cells of R^2/R_sat^2 = q + (1-q) E/(E+Ke), per time step."""
    q = rnv_over_rsat ** 2
    Eb = np.outer(E0, tr_c[prof])
    g = q + (1 - q) * Eb / (Eb + ke)
    return g.mean(axis=1)


def vrel(v, u):
    return (u * u + 3 * v * v) / (3 * v) if v >= u else (v * v + 3 * u * u) / (3 * u)


for pair in ("P", "W"):
    name = {"P": "planktivorous_fish", "W": "predatory_fish"}[pair]
    print()
    print("=" * 100)
    print(f"3. benthivorous_fish -> {name} (3-D encounter in a near-bottom band)")
    print("=" * 100)
    wP = pair_weights(pair)
    print("  phi-weighted members: " + ", ".join(f"{sp} st{st} {w:.3f}" for (sp, st), w in
                                                  sorted(wP.items(), key=lambda kv: -kv[1])))

    def a_of(rs, rn, ke, v, p, hb, prof, fday, fnight):
        """a annual mean, day mean (elev > THR), night mean, all in km2/t/tick."""
        g = light_factor(rn / rs, ke, prof)                     # R^2 / R_sat^2 per time step
        F = np.where(day_prey, fday, fnight)                   # prey share in the band
        series = g * F
        tot = 0.0
        for (sp, st), wt in wP.items():
            d = CL[sp][CL[sp].stage == st]
            s = d.s / d.s.sum()
            n = s * 1e6 / d.w_g
            L = d.L.values
            vv = v * L
            vr = np.array([vrel(x, U_PREY) for x in vv])
            vol = math.pi * (rs * L) ** 2 * vr * p * TICK      # m3 per fish per tick at R_sat
            tot += wt * float((n * vol).sum())
        base = tot / hb / 1e6                                   # km2/t/tick per unit (g F)
        return base * series.mean(), base * series[day_eng].mean(), base * series[~day_eng].mean()

    def pool_F(k_day, k_night):
        fd = sum(s * dn[k_day] for s, dn, nt in POOL[pair].values())
        fn = sum(s * nt[k_night] for s, dn, nt in POOL[pair].values())
        return fd, fn

    fd_c, fn_c = pool_F(1, 1)
    print(f"  prey share inside the band, pool-weighted: day {fd_c:.3f}, night {fn_c:.3f} (central)")
    c_ann, c_day, c_night = a_of(R_SAT[1], R_NV[1], KE[1], V_PRED[1], P_CAP[1], H_BAND[1], "central",
                                 fd_c, fn_c)
    # Monte Carlo (fewer draws: each needs a light series)
    n = min(NMC, 3000)
    rs, rn, ke = logu(*R_SAT[::2], n), logu(*R_NV[::2], n), logu(*KE[::2], n)
    v, p, hb = logu(*V_PRED[::2], n), logu(*P_CAP[::2], n), logu(*H_BAND[::2], n)
    profs = RNG.choice(list(K_PROF), n)
    fdays = {}
    fds, fns = np.zeros(n), np.zeros(n)
    for j, (sj, dn, nt) in POOL[pair].items():
        fds += sj * logu(dn[0], dn[2], n)
        fns += sj * logu(nt[0], nt[2], n)
    # light factors are pre-computed on a grid of (q, ke, prof) to keep the MC fast
    ann, ratio = np.zeros(n), np.zeros(n)
    base_unit = a_of(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "central", 1.0, 1.0)[0]  # g = 1, F = 1, unit factors
    for i in range(n):
        q = min(rn[i] / rs[i], 1.0)
        g = light_factor(q, ke[i], profs[i])
        F = np.where(day_prey, fds[i], fns[i])
        ser = g * F
        # base scales as R_sat^2 x v_rel x p / H; v_rel ~ v for v >> u (approximation in the MC)
        scale = rs[i] ** 2 * v[i] * p[i] / hb[i]
        ann[i] = base_unit * scale * ser.mean()
        ratio[i] = ser[~day_eng].mean() / ser[day_eng].mean()
    # corners
    lo = a_of(R_SAT[0], R_NV[0], KE[2], V_PRED[0], P_CAP[0], H_BAND[2], "highK", *pool_F(0, 0))[0]
    hi = a_of(R_SAT[2], R_NV[2], KE[0], V_PRED[2], P_CAP[2], H_BAND[0], "lowK", *pool_F(2, 2))[0]
    summarise(f"GROUP a (-> {name}), annual mean", c_ann, ann, lo, hi, h=H_PAIR[pair],
              B=[("pool, all visible", B_PAIR[pair]), ("pool at visibility floor 0.25", 0.25 * B_PAIR[pair])])
    print(f"  night/day ratio of a (engine threshold {THR} deg): central {c_night / c_day:.2f}; "
          f"MC 5/50/95 % {np.percentile(ratio, 5):.2f} {np.percentile(ratio, 50):.2f} "
          f"{np.percentile(ratio, 95):.2f}")
    # decomposition of the central night/day ratio
    g_c = light_factor(R_NV[1] / R_SAT[1], KE[1], "central")
    print(f"    vision part (R^2 night/day) {g_c[~day_eng].mean() / g_c[day_eng].mean():.2f}; "
          f"overlap part (F night/day) {fn_c / fd_c:.2f}; mean R^2/R_sat^2 by day "
          f"{g_c[day_eng].mean():.3f}, by night {g_c[~day_eng].mean():.3f}")
    # same-layer a (without the prey vertical share), for an accessible_fraction convention
    print(f"  same-layer a (prey all inside the band, F = 1): central {c_ann / (0.5 * (fd_c + fn_c)):.4f} "
          f"km2/t/tick; with F the annual-mean pool share in the band ~{0.5 * (fd_c + fn_c):.3f}")


# ----------------------------------------------------------------------------------------------
# 4. One-at-a-time sensitivity of the fish pairs (others central): a_low/a_c and a_high/a_c
# ----------------------------------------------------------------------------------------------
print()
print("=" * 100)
print("4. One-at-a-time sensitivity, fish pairs (annual-mean a relative to central; night/day ratio)")
print("=" * 100)
for pair in ("P", "W"):
    wP = pair_weights(pair)

    def a_simple(rs=R_SAT[1], rn=R_NV[1], ke=KE[1], v=V_PRED[1], p=P_CAP[1], hb=H_BAND[1],
                 prof="central", fd=None, fn=None):
        if fd is None:
            fd = sum(s * dn[1] for s, dn, nt in POOL[pair].values())
        if fn is None:
            fn = sum(s * nt[1] for s, dn, nt in POOL[pair].values())
        g = light_factor(min(rn / rs, 1.0), ke, prof)
        ser = g * np.where(day_prey, fd, fn)
        return rs ** 2 * v * p / hb * ser.mean(), ser[~day_eng].mean() / ser[day_eng].mean()

    a0, r0 = a_simple()
    print(f"  pair -> {pair}: central ratio {r0:.2f}")
    tests = {
        "R_sat": [dict(rs=R_SAT[0]), dict(rs=R_SAT[2])],
        "R_nonvisual": [dict(rn=R_NV[0]), dict(rn=R_NV[2])],
        "Ke (low Ke = better dim vision)": [dict(ke=KE[2]), dict(ke=KE[0])],
        "K profile": [dict(prof="highK"), dict(prof="lowK")],
        "v_pred": [dict(v=V_PRED[0]), dict(v=V_PRED[2])],
        "p_capture": [dict(p=P_CAP[0]), dict(p=P_CAP[2])],
        "H_band": [dict(hb=H_BAND[2]), dict(hb=H_BAND[0])],
        "prey share in band (day)": [dict(fd=sum(s * dn[0] for s, dn, nt in POOL[pair].values())),
                                     dict(fd=sum(s * dn[2] for s, dn, nt in POOL[pair].values()))],
        "prey share in band (night)": [dict(fn=sum(s * nt[0] for s, dn, nt in POOL[pair].values())),
                                       dict(fn=sum(s * nt[2] for s, dn, nt in POOL[pair].values()))],
    }
    for k, (lo_kw, hi_kw) in tests.items():
        al, rl = a_simple(**lo_kw)
        ah, rh = a_simple(**hi_kw)
        print(f"    {k:<34} a x {al / a0:5.2f} .. x {ah / a0:5.2f}   (decades {math.log10(ah / al):4.2f})"
              f"   night/day {rl:.2f} .. {rh:.2f}")

print()
print("One-at-a-time sensitivity, benthic pair (all modes moved together; decades of a)")
for k in ("v", "w", "pi"):
    lo_ = sum(wt * MODE_B[SP_MODE[sp]][k][0] / MODE_B[SP_MODE[sp]][k][1] for (sp, st), wt in wB.items())
    hi_ = sum(wt * MODE_B[SP_MODE[sp]][k][2] / MODE_B[SP_MODE[sp]][k][1] for (sp, st), wt in wB.items())
    print(f"    {k:<4} a x {lo_:.2f} .. x {hi_:.2f}  (decades {math.log10(hi_ / lo_):.2f})")

print()
print("=" * 100)
print("5. Fish pairs per member stage (central factors; a per tonne of that stage, annual mean)")
print("=" * 100)
g_c = light_factor(R_NV[1] / R_SAT[1], KE[1], "central")
for pair in ("P", "W"):
    fd = sum(s * dn[1] for s, dn, nt in POOL[pair].values())
    fn = sum(s * nt[1] for s, dn, nt in POOL[pair].values())
    ser = g_c * np.where(day_prey, fd, fn)
    print(f"  pair -> {pair}")
    for (sp, st), wt in sorted(pair_weights(pair).items(), key=lambda kv: -kv[1]):
        d = CL[sp][CL[sp].stage == st]
        s = d.s / d.s.sum()
        n = s * 1e6 / d.w_g
        L = d.L.values
        vr = np.array([vrel(x, U_PREY) for x in V_PRED[1] * L])
        vol = math.pi * (R_SAT[1] * L) ** 2 * vr * P_CAP[1] * TICK
        a_st = float((n * vol).sum()) / H_BAND[1] / 1e6 * ser.mean()
        Lm = 100 * float((s * L).sum())
        print(f"    {sp:<16} stage {st}  weight {wt:5.3f}  N/t {n.sum():8.0f}  biomass-mean L {Lm:5.1f} cm"
              f"  R_sat {R_SAT[1] * Lm / 100:5.2f} m  v {V_PRED[1] * Lm / 100:5.3f} m/s  a {a_st:7.4f}"
              f"  1/(a h) {1 / (a_st * H_PAIR[pair]):6.3f} t/km2")
