"""Holling attack rate a of the predatory_fish pairs from encounter theory.

Usage: python3 -I pred_encounter_a.py <numbers_per_tonne.csv> <pred_scores.csv> <pred_season.csv> <out_dir>

No model output is read. Every input below is a literature value, a value derived from one, or a
stated judgement (J); see predators_encounter.md for the sources.

Unit chain (a in km2 per tonne of foraging predator per 6-h tick, T = 21600 s):
  3D dispersed prey in a layer of thickness H (m):
      a = scan * A(R, H) * v_rel * T * N_t * p * f_ov / H / 1e6 * s_prey
  schooled prey (encounter with schools of biomass M_s t, k prey of w t caught per encounter):
      a = A(R_s, H) * v_rel * T * N_t * (k * w / M_s) / H / 1e6 * s_prey
  prey on or near the bed: the same 3D form with H = the thin near-bed prey layer; when 2R > H the
      slab-clipped cross-section turns it into the strip form scan * 2 R * v_rel * T * N_t * p / 1e6
      (areal density = the engine's depth-integrated t/km2)
  A(R, H) = cross-section of the detection circle inside the prey slab (pi R^2 when 2R <= H).
  v_rel = Gerritsen-Strickler mean relative speed, (u^2 + 3 v^2) / (3 v) for v >= u.
  Ambush (anglerfish): v is replaced by the prey's own speed u (prey moves into the strike zone).
  s_prey = share of the engine's Holling biomass (visible, accessible) made up by this prey type.
Annual mean a = D a_day + (1 - D) a_night, D = 0.5. Night/day ratio = a_night / a_day.
Pair a = sum_i W_i sum_k score_ik a_ik / sum_i W_i score_i,pair  (W_i = group biomass share of the
stage, score = central capability score of pred_scores.csv, i.e. the members inside predator_share).
Each factor is (value giving LOW a, central, value giving HIGH a). Corners: every factor at one end.
Monte Carlo: each factor drawn independently from a triangular distribution in log space, one draw
per factor shared by all stages (factors are common across members, so they are not averaged out).
"""
import math
import os
import sys

import numpy as np
import pandas as pd

NT_CSV, SCORES_CSV, SEASON_CSV, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
T = 21600.0
D_DAY = 0.5
RNG = np.random.default_rng(2026)
NDRAW = 5000

# ---------------------------------------------------------------- predator stages
NT = pd.read_csv(NT_CSV)
SC = pd.read_csv(SCORES_CSV)
SE = pd.read_csv(SEASON_CSV).set_index("common")
MEMBERS = ["saithe", "whiting", "mackerel", "spurdog", "horse mackerel", "hake", "anglerfish",
           "ling", "velvet belly", "pollack"]
w_sp = SE.loc[MEMBERS, "pooled"] / SE.loc[MEMBERS, "pooled"].sum()
STAGES = {"saithe": ["adult", "large"], "whiting": ["juvenile", "adult"], "hake": ["juvenile", "adult"]}
# foraging type, speed key, night-speed key
TYPE = {"saithe": "cruiser", "pollack": "cruiser", "whiting": "cruiser", "mackerel": "mackerel",
        "horse mackerel": "cruiser", "spurdog": "elasmo", "velvet belly": "elasmo",
        "hake": "midambush", "ling": "midambush", "anglerfish": "bedambush"}
SPEED = {"saithe": "v_saithe", "pollack": "v_saithe", "whiting": "v_whiting", "mackerel": "v_mackerel",
         "horse mackerel": "v_hmack", "spurdog": "v_elasmo", "velvet belly": "v_elasmo",
         "hake": "v_hake", "ling": "v_hake", "anglerfish": None}
NIGHTV = {"saithe": "nv_saithe", "pollack": "nv_saithe", "whiting": "nv_whiting"}

rows = []
for sp in MEMBERS:
    for st in STAGES.get(sp, ["all"]):
        r = NT[(NT.species == sp) & (NT.stage == st)].iloc[0]
        rows.append(dict(sp=sp, st=st, W=float(w_sp[sp] * r.share), Nt=float(r.n_per_t), L=float(r.num_mean_L_cm) / 100.0))
STG = pd.DataFrame(rows)
STG_T = list(STG.itertuples())


SCORE = {(r.species, r.stage, r.pool): float(r.s_c) for r in SC.itertuples()}


def score(sp, st, pool):
    return SCORE[(sp, st, pool)]


# ---------------------------------------------------------------- factors (low-a, central, high-a)
P = {
    # speeds, body lengths per second (search speed while foraging)
    "v_saithe": (0.5, 1.0, 1.4),     # Uopt 1.4 BL/s (swim tunnel); in situ gadoids mostly < 1 BL/s
    "v_whiting": (0.5, 0.8, 1.3),    # Uopt 1.0, sustained 1.3 BL/s
    "v_mackerel": (1.25, 1.8, 3.0),  # Macy 1998: routine 1.25, feeding 1.80; sonar 3.8 over ground
    "v_hmack": (0.5, 1.3, 1.8),      # Mapukata 2002 juveniles feeding 1.30-1.80, resting 0.4-0.5
    "v_elasmo": (0.3, 0.5, 0.8),     # Sulikowski 2010: 0.45-0.82 BL/s net (lower bound on swimming)
    "v_hake": (0.2, 0.5, 1.0),       # J: ambusher in midwater, no measurement
    # night speed as fraction of day speed
    "nv_saithe": (0.3, 0.6, 1.0),    # reduced night activity (Skilbrei & Ottera 2016, Armannsson & Jonsson 2012); Dolly Varden 0.14 (J)
    "nv_whiting": (0.7, 1.0, 1.0),   # pelagic whiting feed at night (Pedersen 2000); J
    "scan": (0.5, 0.5, 1.0),         # half the visual disc scanned (Varpe & Fiksen 2010, Langbehn 2019)
    # detection radius R (m), individual fish prey
    "R_fish_day_cruiser": (0.15, 0.5, 1.5),  # lake-trout RD(I) at 0.1-10 lx in the 30-70 m herring layer; RDmax 1.0-1.9 m
    "R_fish_dark": (0.05, 0.2, 0.5),         # non-visual floor 0.1 m (Langbehn), Chinook low-light asymptote 0.5 m, RD(0.01 lx) 0.03 m
    "R_fish_elasmo": (0.1, 0.3, 0.5),        # J: olfactory / electro / lateral line, light independent
    "R_fish_hake_day": (0.15, 0.4, 1.0),     # J: dimmer, deeper than the cruisers
    "R_strike_angler": (0.2, 0.4, 0.8),      # J: no strike distance found; detection c. 5 m (Laurenson 2004)
    "u_prey_angler": (0.05, 0.1, 0.2),       # J: prey speed past a bed ambusher (m/s)
    "R_bedfish_day": (0.1, 0.25, 0.5),       # J: bed at 44-529 m (77 % deeper than 100 m) is below ~0.5 lx even at noon
    # capture probability on individual fish
    "p_fish": (0.1, 0.2, 0.36),              # Turesson & Bronmark 2004: single prey 0.20, pursuit 0.12, sit-and-wait 0.36; pikeperch 0.085
    # school encounter (herring, sprat by day)
    "R_school": (10.0, 15.0, 30.0),          # school half-length 10-14 m (Fernandes 2001) + detection; J
    "M_school": (10.0, 3.0, 1.0),            # t per school: 25 000 fish (Misund 1995) x 50 g = 1.25 t; 5.7-10 t upper bound (Petitgas 2001)
    "k_school": (0.3, 1.0, 3.0),             # J: prey caught per school encounter ("frequent but sublethal" attacks)
    "w_herring": (0.05e-3, 0.05e-3, 0.05e-3),  # t per herring (box juveniles 1-2 ringers, ~20 cm)
    "u_school": (0.3, 0.3, 0.3),             # m/s school speed (J)
    "H_herring_day": (100.0, 70.0, 40.0),    # HERAS July box: backscatter 10-100 m
    "H_herring_night": (60.0, 40.0, 30.0),   # dispersed in the upper 30-50 m (Onsrud 2004, Falk 1968)
    "u_herring": (0.2, 0.2, 0.2),
    "H_bluewhiting": (100.0, 50.0, 30.0),    # J: near-bottom layer on the slope (Torgersen 1997), no thickness found
    "H_pout": (30.0, 15.0, 10.0),            # 5-10 m above the bed (Albert 1994); up to 100 m off bottom in turbid water
    "u_smallfish": (0.1, 0.1, 0.1),
    "f_ov_night": (0.5, 1.0, 1.0),           # J: share of predator foraging time inside the prey layer at night
    # zooplankton pool
    "s_krill": (0.08, 0.12, 0.25),           # North Sea krill 1.17 (0.78-2.4) g/m2 vs pool 9.72 (Mackinson & Daskalov 2007)
    "R_krill": (0.05, 0.1, 0.2),             # J: Calanus 11.6 cm at saturating light x sqrt-area -> ~0.6 m in bright light; dim/dark floor 0.1 m
    "R_krill_mack_day": (0.1, 0.2, 0.4),     # J: mackerel near the surface by day, brighter
    "p_krill": (0.3, 0.5, 0.7),              # J: Langbehn pc 0.7; tail-flip escape
    "H_krill_day": (40.0, 30.0, 20.0),       # 20-40 m thick layer below 70 m (Onsrud 1998, 2004)
    "H_krill_night": (60.0, 40.0, 30.0),     # 10-30 m to ~50-70 m at night
    "u_krill": (0.04, 0.04, 0.04),           # in situ mode ~4 cm/s
    "f_mack_krill_day": (0.3, 0.5, 1.0),     # J: mackerel (upper 40 m) meets krill (> 70 m by day) only in part
    "R_cop": (0.01, 0.02, 0.05),             # Macy-derived 1.7 cm; Rohan pollock 9.6-11.6 cm only in bright light
    "p_cop": (0.5, 0.7, 0.9),
    "H_cop": (100.0, 50.0, 30.0),            # J: Calanus in the upper 30-100 m in the feeding season
    "u_cop": (0.005, 0.005, 0.005),
    "gape_mack": (3.3e-4, 5.0e-4, 7.8e-4),   # m2: Hamann 2023 3.3 cm2; MacKay 1979 5.0 (23 cm), 7.8 (29 cm)
    "ret_mack": (0.3, 0.6, 0.9),             # Macy efficiencies 18-45 % at filter concentrations; ~1 for large copepods
    "night_mack": (0.1, 0.2, 0.5),           # mackerel feeds by day (Dahl & Kirkegaard 1986); Macy threshold ~1e-7 uE
    # benthos (engine B_acc = 0.2 x 101.6 = 20.3 t/km2)
    "s_bed_edible": (0.02, 0.05, 0.12),      # exposed Pandalus + small shrimps/mysids/hyperbenthos 0.4/1.0/2.5 t/km2 over 20.3
    "R_bed_inv": (0.05, 0.1, 0.2),           # J: non-visual at the bed; prior gadoid swept width 0.05-0.3 m
    "p_bed_inv": (0.2, 0.4, 0.7),            # J
    "u_bed_inv": (0.03, 0.03, 0.03),
    "H_bed_inv": (1.0, 0.5, 0.2),            # suprabenthos 76-97 % in 0-50 cm by day (Cunha 1997); shrimps on the surface
    "H_juv": (5.0, 2.0, 1.0),                # J: settled juvenile gadoids (> 8 cm demersal, Bromley & Kell 1999) within 1-5 m of the bed
}
for k, v in P.items():
    assert len(v) == 3, k


def area_in_slab(R, H):
    h = H / 2.0
    if R <= h:
        return math.pi * R * R
    return 2.0 * (R * R * math.asin(h / R) + h * math.sqrt(R * R - h * h))


def vrel(v, u):
    return (u * u + 3 * v * v) / (3 * v) if v >= u else (v * v + 3 * u * u) / (3 * u)


def a3d(R, v, u, H, Nt, p, scan, s, f=1.0):
    return scan * area_in_slab(R, H) * vrel(v, u) * T * Nt * p * f / H / 1e6 * s


def a2d(R, v, u, Nt, p, scan, s):
    return scan * 2.0 * R * vrel(v, u) * T * Nt * p / 1e6 * s


def aschool(Rs, v, u, H, Nt, k, w, Ms, s):
    return area_in_slab(Rs, H) * vrel(v, u) * T * Nt * (k * w / Ms) / H / 1e6 * s


RSCALE = False  # True: individual-prey detection radius scales with predator length, R x L / 0.30 m
SCHOOLS = True  # False: herring and sprat dispersed by day as well (if the engine's hide/visibility floor stands for schooling)
PLANK_SHARE = {"herring": 0.79, "blue whiting": 0.105, "pout": 0.075, "sprat": 0.03}


def stage_a(row, pair, q, night):
    """a (km2/t/tick) of one stage on one pair; q maps factor -> value; night bool."""
    sp, Nt, L = row.sp, row.Nt, row.L
    typ = TYPE[sp]
    if SPEED[sp] is not None:
        v = q[SPEED[sp]] * L
        if night and sp in NIGHTV:
            v *= q[NIGHTV[sp]]
    else:
        v = None
    scan = q["scan"]
    fR = (L / 0.30) if RSCALE else 1.0
    out = {}
    if pair == "plank_fish":
        if typ in ("cruiser", "mackerel"):
            R_day = min(q["R_fish_day_cruiser"] * fR, 2 * L)
        elif typ == "elasmo":
            R_day = q["R_fish_elasmo"] * fR
        else:
            R_day = q["R_fish_hake_day"] * fR
        R_dark = (q["R_fish_elasmo"] if typ == "elasmo" else q["R_fish_dark"]) * fR
        p = q["p_fish"]
        if typ == "bedambush":
            u = q["u_prey_angler"]
            Rs = q["R_strike_angler"]
            # ambusher on the bed meets only the near-bed prey (pout, blue whiting on the slope)
            a = (a3d(Rs, u, 0.0, q["H_pout"], Nt, p, 1.0, PLANK_SHARE["pout"])
                 + a3d(Rs, u, 0.0, q["H_bluewhiting"], Nt, p, 1.0, PLANK_SHARE["blue whiting"]))
            return a
        f_n = q["f_ov_night"] if night else 1.0
        for prey in ("herring", "sprat"):
            s = PLANK_SHARE[prey]
            if not night and not SCHOOLS:
                out[prey] = a3d(R_day, v, q["u_herring"], q["H_herring_day"], Nt, p, scan, s)
            elif not night:
                out[prey] = aschool(q["R_school"], v, q["u_school"], q["H_herring_day"], Nt,
                                    q["k_school"], q["w_herring"], q["M_school"], s)
            else:
                out[prey] = a3d(R_dark, v, q["u_herring"], q["H_herring_night"], Nt, p, scan, s, f_n)
        # deep prey: dark at 150-300 m day and night
        out["blue whiting"] = a3d(R_dark, v, q["u_smallfish"], q["H_bluewhiting"], Nt, p, scan,
                                  PLANK_SHARE["blue whiting"])
        out["pout"] = a3d(R_dark, v, q["u_smallfish"], q["H_pout"], Nt, p, scan, PLANK_SHARE["pout"])
        return sum(out.values())
    if pair in ("zoo_krill", "zoo_cop"):
        if typ == "bedambush":
            return 0.0
        sk = q["s_krill"]
        if pair == "zoo_krill":
            H = q["H_krill_night"] if night else q["H_krill_day"]
            if typ == "mackerel" and not night:
                return a3d(q["R_krill_mack_day"] * fR, v, q["u_krill"], H, Nt, q["p_krill"], scan, sk,
                           q["f_mack_krill_day"])
            a = a3d(q["R_krill"] * fR, v, q["u_krill"], H, Nt, q["p_krill"], scan, sk)
            if typ == "mackerel" and night:
                a *= q["night_mack"]
            return a
        sc = 1.0 - sk
        if typ == "mackerel":
            # ram filtering: gape x speed x retention, per fish, spread over the copepod layer
            a = q["gape_mack"] * v * q["ret_mack"] * T * Nt / q["H_cop"] / 1e6 * sc
            return a * (q["night_mack"] if night else 1.0)
        return a3d(q["R_cop"] * fR, v, q["u_cop"], q["H_cop"], Nt, q["p_cop"], scan, sc)
    if pair == "benthos":
        if typ == "bedambush":
            return a3d(q["R_strike_angler"], q["u_bed_inv"], 0.0, q["H_bed_inv"], Nt, q["p_bed_inv"], 1.0,
                       q["s_bed_edible"])
        return a3d(q["R_bed_inv"] * fR, v, q["u_bed_inv"], q["H_bed_inv"], Nt, q["p_bed_inv"], scan,
                   q["s_bed_edible"])
    if pair == "benth_fish":
        p = q["p_fish"]
        if typ == "bedambush":
            return a3d(q["R_strike_angler"], q["u_prey_angler"], 0.0, q["H_juv"], Nt, p, 1.0, 1.0)
        if typ == "elasmo":
            R = q["R_fish_elasmo"] * fR
        else:
            R = (q["R_fish_dark"] if night else max(q["R_bedfish_day"], q["R_fish_dark"])) * fR
        return a3d(R, v, q["u_smallfish"], q["H_juv"], Nt, p, scan, 1.0)
    raise ValueError(pair)


PAIRS = {"plank_fish": ["plank_fish"], "zoo": ["zoo_krill", "zoo_cop"], "benthos": ["benthos"],
         "benth_fish": ["benth_fish"]}
ENGINE = {"plank_fish": dict(name="-> planktivorous_fish", h=279.4, B=3.33, Bvis=(0.35 * 3.33, 3.33)),
          "zoo": dict(name="-> zooplankton", h=176.5, B=9.72, Bvis=(0.5, 9.72)),
          "benthos": dict(name="-> benthic_community", h=117.6, B=101.6 * 0.2, Bvis=(101.6 * 0.2, 101.6 * 0.2)),
          "benth_fish": dict(name="-> benthivorous_fish", h=176.5, B=1.53 * 0.15, Bvis=(1.53 * 0.15 * 0.6, 1.53 * 0.15))}


def pair_weight_denominator(pair):
    tot = 0.0
    for r in STG.itertuples():
        if pair == "zoo":
            s = 0.5 * score(r.sp, r.st, "zoo_krill") + 0.5 * score(r.sp, r.st, "zoo_cop")
        else:
            s = score(r.sp, r.st, pair)
        tot += r.W * s
    return tot


DEN = {p: pair_weight_denominator(p) for p in PAIRS}


def evaluate(q, detail=False):
    res = {}
    det = []
    for pair, sub in PAIRS.items():
        num_d = num_n = 0.0
        for r in STG_T:
            for sp_pool in sub:
                sc = score(r.sp, r.st, sp_pool)
                if sc == 0:
                    continue
                ad = stage_a(r, sp_pool, q, False)
                an = stage_a(r, sp_pool, q, True)
                num_d += r.W * sc * ad
                num_n += r.W * sc * an
                if detail:
                    det.append(dict(pair=pair, prey=sp_pool, species=r.sp, stage=r.st, W=r.W, score=sc,
                                    N_t=r.Nt, L_m=r.L, a_day=ad, a_night=an,
                                    a_annual=D_DAY * ad + (1 - D_DAY) * an,
                                    weight_in_pair=r.W * sc / DEN[pair]))
        ad, an = num_d / DEN[pair], num_n / DEN[pair]
        res[pair] = (ad, an, D_DAY * ad + (1 - D_DAY) * an)
    return (res, det) if detail else res


def level(i):
    return {k: v[i] for k, v in P.items()}


def draw():
    q = {}
    for k, (lo, c, hi) in P.items():
        if lo == hi:
            q[k] = c
            continue
        a, b = sorted((math.log(lo), math.log(hi)))
        m = min(max(math.log(c), a), b)
        q[k] = math.exp(RNG.triangular(a, m, b))
    return q


central, detail = evaluate(level(1), detail=True)
low = evaluate(level(0))
high = evaluate(level(2))
mc = {p: [] for p in PAIRS}
mc_ratio = {p: [] for p in PAIRS}
for _ in range(NDRAW):
    r = evaluate(draw())
    for p in PAIRS:
        mc[p].append(r[p][2])
        mc_ratio[p].append(r[p][1] / r[p][0] if r[p][0] > 0 else np.nan)

pd.DataFrame(detail).to_csv(os.path.join(OUT, "pred_encounter_stage_detail.csv"), index=False)
print("Encounter-theory attack rates of predatory_fish pairs, km2 per tonne foraging predator per 6-h tick")
print(f"Annual mean a = {D_DAY} a_day + {1 - D_DAY} a_night. Corners: all factors at one end. MC: {NDRAW} log-triangular draws.\n")
summary = []
for p in PAIRS:
    e = ENGINE[p]
    c_d, c_n, c_a = central[p]
    lo_a, hi_a = low[p][2], high[p][2]
    m = np.array(mc[p])
    p05, p50, p95 = np.percentile(m, [5, 50, 95])
    rr = np.array(mc_ratio[p])
    r05, r50, r95 = np.nanpercentile(rr, [5, 50, 95])
    h = e["h"]
    line = dict(pair=e["name"], a_low_corner=lo_a, a_central=c_a, a_high_corner=hi_a, mc_p05=p05,
                mc_p50=p50, mc_p95=p95, a_day=c_d, a_night=c_n, night_day=c_n / c_d,
                night_day_mc05=r05, night_day_mc95=r95, h=h,
                halfsat_central=1 / (c_a * h), halfsat_mc05=1 / (p95 * h), halfsat_mc95=1 / (p05 * h),
                B_engine=e["B"],
                x_central_vis_lo=c_a * h * e["Bvis"][0], x_central_vis_hi=c_a * h * e["Bvis"][1],
                x_p05_vis_lo=p05 * h * e["Bvis"][0])
    summary.append(line)
    print(f"{e['name']}: a central {c_a:.3g} (day {c_d:.3g}, night {c_n:.3g}, night/day {c_n / c_d:.2f}; "
          f"MC 5-95 % {r05:.2f}-{r95:.2f})")
    print(f"    corners {lo_a:.3g} - {hi_a:.3g}; MC 5/50/95 % {p05:.3g} / {p50:.3g} / {p95:.3g}")
    print(f"    h {h}: 1/(a h) central {1 / (c_a * h):.3g} t/km2 (MC 5-95 % {1 / (p95 * h):.3g}-{1 / (p05 * h):.3g}); "
          f"engine B {e['B']:.3g}; x = a h B_vis at central {c_a * h * e['Bvis'][0]:.3g}-{c_a * h * e['Bvis'][1]:.3g}, "
          f"at MC 5 % {p05 * h * e['Bvis'][0]:.3g}")
    print(f"    fraction of ceiling x/(1+x) at central, low B_vis: {c_a * h * e['Bvis'][0] / (1 + c_a * h * e['Bvis'][0]):.3f}; "
          f"at MC 5 %: {p05 * h * e['Bvis'][0] / (1 + p05 * h * e['Bvis'][0]):.3f}\n")
pd.DataFrame(summary).to_csv(os.path.join(OUT, "pred_encounter_summary.csv"), index=False)

# per-member annual a (central) and contribution
D = pd.DataFrame(detail)
print("Per member and stage, central (annual a, weight inside the pair, share of the pair's a):")
for p in PAIRS:
    d = D[D.pair == p].copy()
    d["contrib"] = d.weight_in_pair * d.a_annual
    g = d.groupby(["species", "stage"]).agg(a_annual=("a_annual", "sum"), a_day=("a_day", "sum"),
                                           a_night=("a_night", "sum"), weight=("weight_in_pair", "first"),
                                           contrib=("contrib", "sum"))
    g["share_of_a"] = g.contrib / g.contrib.sum()
    print(f"  {ENGINE[p]['name']}")
    for (sp, st), r in g.sort_values("contrib", ascending=False).iterrows():
        nd = r.a_night / r.a_day if r.a_day > 0 else float("nan")
        print(f"    {sp:15s} {st:9s} weight {r.weight:.3f}  a {r.a_annual:.3g} (day {r.a_day:.3g}, night {r.a_night:.3g}, n/d {nd:.2f})  share {r.share_of_a:.2f}")

# one-at-a-time sensitivity: decades spanned by each factor (others central)
print("\nOne-at-a-time sensitivity, log10(a_high/a_low) per factor (others central), top 6 per pair:")
base = level(1)
for p in PAIRS:
    sens = []
    for k, (lo, c, hi) in P.items():
        if lo == hi:
            continue
        ql, qh = dict(base), dict(base)
        ql[k], qh[k] = lo, hi
        al, ah = evaluate(ql)[p][2], evaluate(qh)[p][2]
        if al > 0 and ah > 0 and abs(math.log10(ah / al)) > 1e-6:
            sens.append((math.log10(ah / al), k))
    sens.sort(reverse=True)
    print(f"  {ENGINE[p]['name']}: " + "; ".join(f"{k} {d:.2f}" for d, k in sens[:6]))

# school vs dispersed decomposition of the planktivore pair (central, saithe adult)
print("\nPlanktivore pair, saithe adult, central, by prey type (km2/t/tick, already x prey share):")
r = STG[(STG.sp == "saithe") & (STG.st == "adult")].iloc[0]
q = level(1)
L = r.L
v = q["v_saithe"] * L
vn = v * q["nv_saithe"]
for prey, s in PLANK_SHARE.items():
    if prey in ("herring", "sprat"):
        d_ = aschool(q["R_school"], v, q["u_school"], q["H_herring_day"], r.Nt, q["k_school"], q["w_herring"], q["M_school"], s)
        n_ = a3d(q["R_fish_dark"], vn, q["u_herring"], q["H_herring_night"], r.Nt, q["p_fish"], q["scan"], s)
        disp_day = a3d(q["R_fish_day_cruiser"], v, q["u_herring"], q["H_herring_day"], r.Nt, q["p_fish"], q["scan"], s)
        print(f"  {prey:12s} day (schools) {d_:.3g}; day if dispersed {disp_day:.3g}; night (dispersed, dark) {n_:.3g}")
    else:
        H = q["H_bluewhiting"] if prey == "blue whiting" else q["H_pout"]
        print(f"  {prey:12s} day {a3d(q['R_fish_dark'], v, 0.1, H, r.Nt, q['p_fish'], q['scan'], s):.3g}; "
              f"night {a3d(q['R_fish_dark'], vn, 0.1, H, r.Nt, q['p_fish'], q['scan'], s):.3g}")

# visual range check: lake-trout RD(I) in the herring layer
print("\nLake-trout reactive distance RD = 0.2549 I^0.4747 (I lx, cap 1.012 m) at depth, clear sky x 0.7 cloud, Kd 0.15 / 0.20:")
for label, E0 in (("summer noon (~100 000 lx)", 1.0e5), ("winter noon (~11 400 lx)", 1.14e4), ("sun at +5 deg (~5 200 lx)", 5.2e3)):
    parts = []
    for z in (30, 50, 70, 100):
        vals = []
        for kd in (0.15, 0.20):
            I = E0 * 0.7 * 0.9 * math.exp(-kd * z)
            vals.append(min(0.2549 * I ** 0.4747, 1.012))
        parts.append(f"{z} m {vals[0]:.2f}/{vals[1]:.2f}")
    print(f"  {label}: " + "; ".join(parts))

# variant: no schooling dilution by day (in case the engine's Rest-as-hide / visibility floor 0.25 is meant
# to carry the schooling refuge; then schooling must not be inside a as well)
SCHOOLS = False
c_ns = evaluate(level(1))["plank_fish"]
lo_ns, hi_ns = evaluate(level(0))["plank_fish"][2], evaluate(level(2))["plank_fish"][2]
m_ns = np.array([evaluate(draw())["plank_fish"][2] for _ in range(2000)])
q05, q50, q95 = np.percentile(m_ns, [5, 50, 95])
print("\nVariant without schooling dilution (herring, sprat dispersed by day, visual R):")
print(f"  -> planktivorous_fish: a central {c_ns[2]:.3g} (day {c_ns[0]:.3g}, night {c_ns[1]:.3g}, night/day {c_ns[1] / c_ns[0]:.2f}); "
      f"corners {lo_ns:.3g}-{hi_ns:.3g}; MC 5/50/95 % {q05:.3g}/{q50:.3g}/{q95:.3g}; 1/(a h) central {1 / (c_ns[2] * 279.4):.3g} t/km2")
SCHOOLS = True

# variant: detection radius proportional to predator length (reference 30 cm), central values only
RSCALE = True
print("\nVariant with R proportional to predator length (R x L/0.30 m), central factors:")
for p_ in PAIRS:
    r_ = evaluate(level(1))[p_]
    print(f"  {ENGINE[p_]['name']}: a {r_[2]:.3g} (night/day {r_[1] / r_[0]:.2f}); 1/(a h) {1 / (r_[2] * ENGINE[p_]['h']):.3g} t/km2")
RSCALE = False
