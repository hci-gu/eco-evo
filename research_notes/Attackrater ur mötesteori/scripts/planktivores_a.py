"""Attack rate a of planktivorous_fish -> zooplankton from encounter theory.

Usage: python3 -I planktivores_a.py [--foraging-overlap]
                                               (no inputs; every number is written below
                                                with its source key from the evidence notes)

  --foraging-overlap: the engine's a is per FORAGING fish (the eat share is a learned action),
      so herring and sprat that forage on this pool are taken to be inside its 0-25 m layer:
      O = 1.0 (0.7-1.2, J) in every period and season instead of the all-fish echo shares.
      Pout and blue whiting keep theirs (their feeding is mostly outside the pool).
      summary.md, decision 1 (2026-10-26).

Definitions (planktivores_encounter.md section 2):
- engine: intake = eat x hunger x a B / (1 + a h B + w B_pred), B = visible pool biomass,
  depth-integrated t/km2; a in km2 per t of foraging predator per 6 h tick.
- the pool is the resident 0-25 m community at A17 (resume 183.6), so its mean concentration in
  its own layer is B / 25 m. A fish meets B / H_eff with H_eff = 25 m / O, O = encounter-weighted
  overlap (evidence_vertical_overlap.md section 0).
- particulate clearance per individual: f pi R^2 v p (half visual field f, reactive distance R,
  search speed v, capture success p); R = k_v sqrt(A_prey) x g(I), g(I) = I / (Ke + I) on R^2,
  0 below the biting threshold (Aksnes & Utne 1997 small-attenuation limit).
- filter clearance (herring only): gape area x speed x retention; used where it exceeds biting
  (in practice: darkness).
- per tonne: sum over the length distribution, via the moments of tables/size_structure_stdout.txt.
  With R independent of fish length (central; the k_v anchors are size-free) the per-tonne
  clearance uses M1 = sum n L / sum n W; the gape uses M2.
- a_i(t) = sum_prey s_j(month) clear_i,j(t) x O_i(period, season) / 25 m / 1e6.
- library a = TIME mean over the year (the engine normalises m(t) to a time mean of 1);
  the implied dark_ratio is the least-squares fit of a(t) = a_L (rho + (1 - rho) F(t)) with the
  engine's F (logistic on sun elevation, -8 deg, 2 deg transition, lib/world/daylight.py).
"""
import math
import sys

import numpy as np

TICK_S = 6 * 3600.0
H_POOL = 25.0
LAT = 58.15
H_HANDLING = 40.0
RNG = np.random.default_rng(20261025)

# ------------------------------------------------------------------ calendar and light
STEP_H = 1.0 / 6.0
t = np.arange(0.0, 8760.0, STEP_H) + STEP_H / 2
decl = np.radians(23.44) * np.sin(2 * np.pi * (284.0 + 1.0 + t / 24.0) / 365.0)
ha = np.radians(15.0 * (np.mod(t, 24.0) - 12.0))
phi = math.radians(LAT)
elev = np.degrees(np.arcsin(np.clip(np.sin(phi) * np.sin(decl)
                                    + math.cos(phi) * np.cos(decl) * np.cos(ha), -1, 1)))
F_ENGINE = 0.5 * (1.0 + np.tanh(0.5 * (elev + 8.0) / 2.0))
month = np.minimum((t / (8760.0 / 12.0)).astype(int), 11)
summer = (month >= 3) & (month <= 8)                       # Apr-Sep (overlap note 4.2)
DAY, TW, NIGHT = 0, 1, 2
period = np.where(elev > 0, DAY, np.where(elev > -12, TW, NIGHT))

# surface illuminance, lx: 1.3e5 sin(h) by day (overlap note 3, J), twilight from the AMS
# glossary values (450 lx at 0 deg, 3 lx at -6, 0.008 at -12, 6e-4 at -18), overcast 0.2 (J)
TW_H = np.array([-90.0, -18.0, -12.0, -6.0, 0.0])
TW_LOG = np.log10([6e-4, 6e-4, 8e-3, 3.0, 450.0])
E_day = 1.3e5 * np.sin(np.radians(np.maximum(elev, 0.0)))
E_tw = 10 ** np.interp(elev, TW_H, TW_LOG)
CLOUD = 0.2
E0 = CLOUD * np.maximum(E_day, E_tw)


def light_at(z, k_upper=0.30, k_lower=0.12):
    """lx at depth z; K 0.30 m-1 in 0-20 m (Aas et al. 2014), 0.12 below (Aksnes et al. 2009, J)."""
    zu = min(z, 20.0)
    return E0 * np.exp(-k_upper * zu - k_lower * max(z - 20.0, 0.0))


# ------------------------------------------------------------------ prey (the 0-25 m pool)
A_CAL = 0.75 * 2.5e-3 * 0.8e-3      # Calanus CIV-adult image area, m2 (herring note 2.1)
A_SMALL = 0.75 * 1.0e-3 * 0.4e-3    # small copepod, m2
# Calanus share of the pool biomass by month: A17 carbon shares (winter_food note Q1),
# zero in Dec-Mar where the target is the resident stock without Calanus (resume 183.6)
S_CAL = np.array([0, 0, 0, .88, .76, .35, .47, .65, .78, .29, .23, 0.0])
# monthly target biomass t/km2 (resume 183.6 hybrid row), for the biomass-weighted mean
B_TARGET = np.array([0.21, 0.18, 1.56, 19.3, 26.0, 11.4, 6.9, 4.6, 3.7, 2.9, 3.5, 0.92])
THR_CAL, THR_SMALL = 1e-3, 1e-2     # biting thresholds, lx (Batty et al. 1990 via herring note)

# ------------------------------------------------------------------ members
# moments per tonne from tables/size_structure_stdout.txt (pooled Q1/Q3, GOV 2021-25)
M = {  # (M1 all, M1 low-N/t bound, M1 high-N/t bound, M2 all)
    "herring": (4.591e5, 2.434e5, 4.885e5, 7.949e6),     # low: L >= 20 cm; high: Q1
    "sprat": (1.235e6, 1.174e6, 1.896e6, 1.351e7),       # low: Q3; high: Q1
    "pout": (7.849e5, 7.624e5, 8.69e5, 1.012e7),          # Q3 / Q1
    "bw_small": (2.936e5, 2.685e5, 4.92e5, 7.195e6),      # L < 29 cm; low all; high Q1
    "bw_large": (2.685e5, 2.676e5, 2.685e5, 6.825e6),     # proxy: all sizes
}
W_GROUP = {"herring": 0.79, "sprat": 0.03, "pout": 0.075, "bw_small": 0.105 * 0.787,
           "bw_large": 0.105 * 0.213}

# factor triplets (low, central, high); keys in planktivores_encounter.md section 2
FACT = {
    "f_field": (0.25, 0.5, 1.0),
    "kv": {"herring": (60, 115, 218), "sprat": (60, 100, 155), "pout": (60, 100, 155),
           "bw_small": (60, 100, 155), "bw_large": (60, 100, 155)},
    "v_bl": {"herring": (0.6, 1.0, 1.4), "sprat": (0.3, 0.6, 1.0), "pout": (0.2, 0.33, 0.6),
             "bw_small": (0.2, 0.33, 0.6), "bw_large": (0.2, 0.33, 0.6)},
    "p": {"herring": (0.3, 0.6, 0.9), "sprat": (0.33, 0.7, 0.86), "pout": (0.33, 0.7, 0.86),
          "bw_small": (0.33, 0.7, 0.86), "bw_large": (0.33, 0.7, 0.86)},
    "ke_lx": (10.0, 1.0, 0.1),        # half-saturation of R^2 for the clupeids (J; low a = high Ke)
    "z_forage": (25.0, 15.0, 8.0),    # depth at which the clupeids' light is evaluated (J)
}
# overlap O (low, central, high) by (period, season) - evidence_vertical_overlap.md 4.2
O = {
    "herring": {(DAY, 1): (0.25, 0.45, 1.0), (TW, 1): (0.3, 0.6, 1.2), (NIGHT, 1): (0.5, 0.8, 1.2),
                (DAY, 0): (0.05, 0.15, 0.4), (TW, 0): (0.3, 0.5, 0.8), (NIGHT, 0): (0.3, 0.55, 0.8)},
    "sprat": {(DAY, s): (0.25, 0.45, 1.0) for s in (0, 1)} | {(TW, s): (0.3, 0.6, 1.2) for s in (0, 1)}
    | {(NIGHT, s): (0.5, 0.8, 1.2) for s in (0, 1)},
    "pout": {(DAY, s): (0.01, 0.03, 0.07) for s in (0, 1)} | {(TW, s): (0.03, 0.06, 0.12) for s in (0, 1)}
    | {(NIGHT, s): (0.05, 0.1, 0.2) for s in (0, 1)},
    "bw_small": {(DAY, s): (0.0, 0.0, 0.01) for s in (0, 1)} | {(TW, s): (0.1, 0.35, 0.6) for s in (0, 1)}
    | {(NIGHT, s): (0.4, 0.7, 1.2) for s in (0, 1)},
    "bw_large": {(DAY, s): (0.0, 0.0, 0.01) for s in (0, 1)} | {(TW, s): (0.01, 0.03, 0.07) for s in (0, 1)}
    | {(NIGHT, s): (0.03, 0.07, 0.15) for s in (0, 1)},
}
if "--foraging-overlap" in sys.argv[1:]:
    for _m in ("herring", "sprat"):
        O[_m] = {k: (0.7, 1.0, 1.2) for k in O[_m]}
    print("variant: --foraging-overlap (herring and sprat O = 1.0, 0.7-1.2)")
# R^2 light factor for the gadoid members by period (dim-light visual + tactile; J):
# day / twilight / night; pout stomach fullness night/day 0.9 (Onsrud 2004), blue whiting
# night-biased on krill; copepods are small prey for them (J)
G_GADOID = {DAY: (0.5, 1.0, 1.0), TW: (0.3, 0.7, 1.0), NIGHT: (0.1, 0.4, 0.8)}
# herring filtering: gape 0.00359 TL^2 (Hamann 2023, isometric, herring note), dark speed 0.11 m/s
# (Batty et al. 1986), retention 0.9 (J)
GAPE_TL2, V_DARK, RETENTION = 0.00359, 0.11, 0.9


def herring_like_g(ke, z, thr):
    """R^2 light factor at the foraging depth for a clupeid."""
    i = light_at(z)
    return np.where(i >= thr, i / (ke + i), 0.0)


def a_series(par, members=None):
    """a_i(t) per member [km2/t/tick] for one parameter set; returns dict and group series."""
    members = members or list(M)
    sc = S_CAL[month]
    out = {}
    g_cal = herring_like_g(par["ke_lx"], par["z_forage"], THR_CAL)
    g_small = herring_like_g(par["ke_lx"], par["z_forage"], THR_SMALL)
    for m in members:
        m1 = par["M1"][m]
        base = par["f_field"] * math.pi * par["kv"][m] ** 2 * par["v_bl"][m] * m1 / 100.0 * par["p"][m] * TICK_S
        if m in ("herring", "sprat"):
            clear = base * (sc * A_CAL * g_cal + (1 - sc) * A_SMALL * g_small)
            if m == "herring":
                filt = GAPE_TL2 * M[m][3] * 1e-4 * V_DARK * RETENTION * TICK_S
                clear = np.maximum(clear, filt)
        else:
            g = np.choose(period, [par["g_gad"][DAY], par["g_gad"][TW], par["g_gad"][NIGHT]])
            clear = base * (sc * A_CAL + (1 - sc) * A_SMALL) * g
        ov = np.zeros_like(t)
        for (p_, s_), val in par["O"][m].items():
            ov[(period == p_) & (summer == bool(s_))] = val
        out[m] = clear * ov / H_POOL / 1e6
    grp = sum(W_GROUP[m] * out[m] for m in members) / sum(W_GROUP[m] for m in members)
    return out, grp


def central():
    return dict(
        f_field=FACT["f_field"][1], kv={k: v[1] for k, v in FACT["kv"].items()},
        v_bl={k: v[1] for k, v in FACT["v_bl"].items()}, p={k: v[1] for k, v in FACT["p"].items()},
        ke_lx=FACT["ke_lx"][1], z_forage=FACT["z_forage"][1],
        M1={k: v[0] for k, v in M.items()},
        O={m: {k: v[1] for k, v in d.items()} for m, d in O.items()},
        g_gad={k: v[1] for k, v in G_GADOID.items()})


def corner(which):
    i = {"low": 0, "high": 2}[which]
    j = {"low": 1, "high": 2}[which]   # M1 bound columns
    return dict(
        f_field=FACT["f_field"][i], kv={k: v[i] for k, v in FACT["kv"].items()},
        v_bl={k: v[i] for k, v in FACT["v_bl"].items()}, p={k: v[i] for k, v in FACT["p"].items()},
        ke_lx=FACT["ke_lx"][i], z_forage=FACT["z_forage"][i],
        M1={k: v[j] for k, v in M.items()},
        O={m: {k: v[i] for k, v in d.items()} for m, d in O.items()},
        g_gad={k: v[i] for k, v in G_GADOID.items()})


def tri_log(lo, mode, hi, n):
    """Triangular draw in log space (zero-safe: a zero low bound is drawn linearly)."""
    if lo <= 0:
        return RNG.triangular(lo, mode, hi, n) if hi > lo else np.full(n, mode)
    l, m_, h = np.log([lo, mode, hi])
    return np.exp(RNG.triangular(l, m_, h, n)) if h > l else np.full(n, mode)


def fit_engine(a):
    """Least-squares a(t) = c0 + c1 F(t); returns (a_light, rho, R^2)."""
    X = np.vstack([np.ones_like(F_ENGINE), F_ENGINE]).T
    c, *_ = np.linalg.lstsq(X, a, rcond=None)
    pred = X @ c
    r2 = 1 - ((a - pred) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    a_l = c[0] + c[1]
    return a_l, c[0] / a_l, r2


def fit_engine_monthly(a):
    """Diel-only fit: per month a(t) = c0m + c1m F(t); returns the 12 implied dark_ratios.

    The plain fit over the year mixes the seasonal swing (prey size, overlap) with the diel
    one, because long days and Calanus-rich months coincide; the engine's rho is a diel ratio.
    """
    out = []
    for k in range(12):
        sel = month == k
        X = np.vstack([np.ones(sel.sum()), F_ENGINE[sel]]).T
        c, *_ = np.linalg.lstsq(X, a[sel], rcond=None)
        out.append(c[0] / (c[0] + c[1]) if c[0] + c[1] > 0 else float("nan"))
    return np.array(out)


def half_point(a, k):
    """Sun elevation at which a(t) crosses half its daytime mean in month k (mean over crossings)."""
    sel = month == k
    d = a[sel & (period == DAY)].mean()
    e = elev[sel]
    x = a[sel] - 0.5 * d
    idx = np.where(np.sign(x[:-1]) != np.sign(x[1:]))[0]
    return float(np.mean(e[idx])) if len(idx) else float("nan")


def night_day(a):
    """Time mean over months of a(night, sun < -12) / a(day, sun > 0); months without night skipped."""
    r = []
    for k in range(12):
        n = (month == k) & (period == NIGHT)
        if n.any():
            r.append(a[n].mean() / a[(month == k) & (period == DAY)].mean())
    return float(np.mean(r))


def summary(name, par):
    per, grp = a_series(par)
    print(f"\n=== {name}")
    print(f"{'member':10s} {'w':>6s} {'clear/t day sat (m3/tick)':>26s} {'a annual':>10s} {'a day':>9s} "
          f"{'a twil':>9s} {'a night':>9s} {'night/day':>9s} {'1/(a h)':>8s}")
    for m in list(M) + ["GROUP"]:
        a = grp if m == "GROUP" else per[m]
        w = 1.0 if m == "GROUP" else W_GROUP[m]
        if m == "GROUP":
            sat = sum(W_GROUP[k] * par["f_field"] * math.pi * par["kv"][k] ** 2 * par["v_bl"][k]
                      * par["M1"][k] / 100 * par["p"][k] * TICK_S * A_CAL for k in M)
        else:
            sat = par["f_field"] * math.pi * par["kv"][m] ** 2 * par["v_bl"][m] * par["M1"][m] / 100 \
                * par["p"][m] * TICK_S * A_CAL
        d, tw, n = (a[period == k].mean() for k in (DAY, TW, NIGHT))
        am = a.mean()
        print(f"{m:10s} {w:6.3f} {sat:26.3e} {am:10.4f} {d:9.4f} {tw:9.4f} {n:9.5f} {n / d if d > 0 else float('nan'):9.4f} "
              f"{1 / (am * H_HANDLING) if am > 0 else float('inf'):8.2f}")
    for m in ("herring", "GROUP"):
        a = grp if m == "GROUP" else per[m]
        cells = []
        for sname, sv in (("Apr-Sep", True), ("Oct-Mar", False)):
            for pname, pv in (("day", DAY), ("twilight", TW), ("night", NIGHT)):
                sel = (summer == sv) & (period == pv)
                cells.append(f"{sname} {pname} {a[sel].mean() if sel.any() else float('nan'):.4f}")
        print(f"{m} a by season x period: " + "; ".join(cells))
    a_l, rho, r2 = fit_engine(grp)
    bw = (grp * B_TARGET[month]).sum() / B_TARGET[month].sum()
    print(f"group: time-mean a {grp.mean():.4f}; prey-biomass-weighted a {bw:.4f}; engine-form fit "
          f"a_light {a_l:.4f}, implied dark_ratio {rho:.3f} (R2 {r2:.2f}); check a_light (rho + (1-rho) Fbar) = "
          f"{a_l * (rho + (1 - rho) * F_ENGINE.mean()):.4f}")
    print("(the engine-form fits are NOT a usable dark_ratio: the derived a drops at sun ~0 deg, the "
          "engine F at -8 deg; use the night/day ratio below)")
    print("elevation (deg) where the group a falls to half of its month's daytime mean, by month:",
          " ".join(f"{half_point(grp, k):.1f}" for k in range(12)))
    nd = [grp[(month == k) & (period == NIGHT)].mean() / grp[(month == k) & (period == DAY)].mean()
          if ((month == k) & (period == NIGHT)).any() else float("nan") for k in range(12)]
    print("night (< -12 deg) / day (> 0 deg) a by month:", " ".join(f"{x:.3f}" for x in nd),
          f"| time mean over months with night {night_day(grp):.3f}")
    shares = {m: W_GROUP[m] * per[m].mean() / grp.mean() for m in M}
    print("share of the group a by member:", {k: round(v, 3) for k, v in shares.items()})
    mon = [grp[month == k].mean() for k in range(12)]
    print("group a by month:", " ".join(f"{x:.4f}" for x in mon))
    print("1/(a h) by month (t/km2):", " ".join(f"{1 / (x * H_HANDLING):.2f}" for x in mon))
    print("target B / (1/(a_annual h)) by month:", " ".join(f"{b * grp.mean() * H_HANDLING:.1f}" for b in B_TARGET))
    return grp, per


print(f"light: annual mean engine F (-8 deg) = {F_ENGINE.mean():.4f}; time shares day/twilight/night = "
      + " / ".join(f"{(period == k).mean():.3f}" for k in (DAY, TW, NIGHT)))
for z in (8.0, 15.0, 25.0):
    li = light_at(z)
    print(f"lx at {z:.0f} m: median by day {np.median(li[period == DAY]):.3g}, twilight {np.median(li[period == TW]):.3g}, "
          f"night {np.median(li[period == NIGHT]):.2g}; share of twilight above 1e-3 lx {np.mean(li[period == TW] >= 1e-3):.2f}")
g_c = herring_like_g(1.0, 15.0, THR_CAL)
print("central clupeid R^2 light factor (Ke 1 lx, 15 m): day %.2f, twilight %.2f, night %.3f" %
      tuple(g_c[period == k].mean() for k in (DAY, TW, NIGHT)))
filt = GAPE_TL2 * M["herring"][3] * 1e-4 * V_DARK * RETENTION * TICK_S
print(f"herring filter clearance per t (dark speed): {filt:.3e} m3/tick")

grp_c, per_c = summary("CENTRAL", central())
grp_l, _ = summary("LOW corner (every factor at its low end)", corner("low"))
grp_h, _ = summary("HIGH corner (every factor at its high end)", corner("high"))

# ------------------------------------------------------------------ one-at-a-time sensitivity
print("\n=== one-at-a-time: group time-mean a with one factor at its low / high value")
base = central()
a0 = a_series(base)[1].mean()
for key in ("f_field", "ke_lx", "z_forage"):
    vals = []
    for i in (0, 2):
        p = central()
        p[key] = FACT[key][i]
        vals.append(a_series(p)[1].mean())
    print(f"{key:14s} {vals[0]:.4f} / {vals[1]:.4f}  (x{vals[0] / a0:.2f} / x{vals[1] / a0:.2f})")
for key in ("kv", "v_bl", "p"):
    vals = []
    for i in (0, 2):
        p = central()
        p[key] = {k: v[i] for k, v in FACT[key].items()}
        vals.append(a_series(p)[1].mean())
    print(f"{key:14s} {vals[0]:.4f} / {vals[1]:.4f}  (x{vals[0] / a0:.2f} / x{vals[1] / a0:.2f})")
for lab, j in (("size (N/t)", 1), ("size (N/t)", 2)):
    p = central()
    p["M1"] = {k: v[j] for k, v in M.items()}
    v = a_series(p)[1].mean()
    print(f"{lab:14s} {'low' if j == 1 else 'high'}: {v:.4f} (x{v / a0:.2f})")
for i in (0, 2):
    p = central()
    p["O"] = {m: {k: v[i] for k, v in d.items()} for m, d in O.items()}
    v = a_series(p)[1].mean()
    print(f"overlap O      {'low' if i == 0 else 'high'}: {v:.4f} (x{v / a0:.2f})")

# R proportional to fish length (anchored at 20 cm): per-tonne sum of R^2 v uses M3 / 20^2
M3 = {"herring": 1.488e8, "sprat": 1.497e8, "pout": 1.412e8, "bw_small": 1.778e8, "bw_large": 1.762e8}
p = central()
p["M1"] = {k: M3[k] / 400.0 for k in M}
v = a_series(p)[1].mean()
print(f"R proportional to L (20 cm anchor): group time-mean a {v:.4f} (x{v / a0:.2f}); "
      + ", ".join(f"{k} x{M3[k] / 400.0 / M[k][0]:.2f}" for k in M))

# ------------------------------------------------------------------ Monte Carlo
N_MC = 1500
draws = []
rhos = []
mem_draws = {m: [] for m in M}
for _ in range(N_MC):
    p = central()
    p["f_field"] = tri_log(*FACT["f_field"], 1)[0]
    p["ke_lx"] = tri_log(FACT["ke_lx"][2], FACT["ke_lx"][1], FACT["ke_lx"][0], 1)[0]
    p["z_forage"] = RNG.triangular(FACT["z_forage"][2], FACT["z_forage"][1], FACT["z_forage"][0])
    for key in ("kv", "v_bl", "p"):
        p[key] = {k: tri_log(*v, 1)[0] for k, v in FACT[key].items()}
    p["M1"] = {k: RNG.uniform(min(v[1], v[0]), max(v[2], v[0])) for k, v in M.items()}
    p["O"] = {m: {k: tri_log(*v, 1)[0] for k, v in d.items()} for m, d in O.items()}
    p["g_gad"] = {k: tri_log(*v, 1)[0] for k, v in G_GADOID.items()}
    per_d, g = a_series(p)
    for m in M:
        mem_draws[m].append(per_d[m].mean())
    draws.append(g.mean())
    rhos.append(night_day(g))
draws, rhos = np.array(draws), np.array(rhos)
q = np.percentile(draws, [5, 25, 50, 75, 95])
print(f"\n=== Monte Carlo ({N_MC} draws, log-triangular per factor, independent)")
print("group time-mean a, P5 P25 P50 P75 P95:", " ".join(f"{x:.4f}" for x in q))
print("1/(a h) at those:", " ".join(f"{1 / (x * H_HANDLING):.2f}" for x in q))
print("group night/day a ratio (monthly, time mean) P5 P50 P95:", " ".join(f"{x:.3f}" for x in np.percentile(rhos, [5, 50, 95])))
print(f"P(a >= 0.04) = {np.mean(draws >= 0.04):.3f}")
for m in M:
    qm = np.percentile(mem_draws[m], [5, 50, 95])
    print(f"member {m:9s} time-mean a P5 P50 P95: " + " ".join(f"{x:.4f}" for x in qm))

# ------------------------------------------------------------------ validation (not used above)
print("\n=== validation (not used to set a)")
part_day_sat = central()
pc = part_day_sat
herring_day = pc["f_field"] * math.pi * pc["kv"]["herring"] ** 2 * pc["v_bl"]["herring"] * pc["M1"]["herring"] / 100 \
    * pc["p"]["herring"] * TICK_S * (0.5 * A_CAL + 0.5 * A_SMALL)
print(f"herring particulate clearance per t (box size structure), day, saturating light, 50/50 prey: "
      f"{herring_day:.3e} m3/tick (NOT size-matched to the YOY datum; see next lines)")
# size-matched: A&H fish 4-8 cm, small prey; W from the herring CA length-weight fit
for L_cm in (4.0, 6.0, 8.0):
    W_g = 0.0022306 * L_cm ** 3.3702
    for lab, R in (("R size-free", pc["kv"]["herring"] * math.sqrt(A_SMALL)),
                   ("R prop. to L (20 cm anchor)", pc["kv"]["herring"] * math.sqrt(A_SMALL) * L_cm / 20.0)):
        q_ls = pc["f_field"] * math.pi * R ** 2 * (pc["v_bl"]["herring"] * L_cm / 100) * pc["p"]["herring"] * 1000
        per_g = q_ls * 3600 / W_g
        print(f"  YOY {L_cm:.0f} cm ({W_g:.2f} g), {lab}: R {100 * R:.1f} cm, {q_ls:.3f} L/s = {per_g:.0f} L/g/h "
              f"vs Arrhenius & Hansson 46 L/g/h (theory/field {per_g / 46.2:.1f})")
print(f"A&H as area at O 1 (25 m): 0.0111; at the summer-day central O 0.45: {2.77e5 * 0.45 / 25 / 1e6:.4f} km2/t/tick")
for nm, cl in (("Kulke sprat low", 2.6e5), ("Kulke sprat high", 5.2e5)):
    print(f"{nm}: {cl:.2e} m3/t/tick -> a at O 1: {cl / 25 / 1e6:.4f}, at O 0.45: {cl * 0.45 / 25 / 1e6:.4f}")
sp_c = pc["f_field"] * math.pi * pc["kv"]["sprat"] ** 2 * pc["v_bl"]["sprat"] * pc["M1"]["sprat"] / 100 \
    * pc["p"]["sprat"] * TICK_S * (0.5 * A_CAL + 0.5 * A_SMALL)
print(f"sprat encounter clearance per t, day, saturating, 50/50 prey: {sp_c:.3e} m3/tick")
# potential intake at the central a, visible fraction 0.75 (J), eat share = hunger = 1, vs field rations
a_mon = np.array([grp_c[month == k].mean() for k in range(12)])
bv = 0.75 * B_TARGET
f_pot = a_mon * bv / (1 + a_mon * H_HANDLING * bv)      # t/t/tick
print("potential intake %BW/d by month (central a(month), visible 0.75 B, no interference):",
      " ".join(f"{400 * x:.1f}" for x in f_pot))
print(f"annual mean potential {400 * f_pot.mean():.2f} %BW/d = {1460 * f_pot.mean():.1f} /yr; "
      "field: group zooplankton Q/B x share 4.5 /yr (params note), herring winter 0.4-1 %BW/d, adult summer ~6 %BW/d")
print("foraging-time fraction needed to eat Q/B 4.5 /yr: %.2f" % (4.5 / (1460 * f_pot.mean())))
print(f"ceiling 1/h = {100 * 4 / H_HANDLING:.1f} %BW/d")
