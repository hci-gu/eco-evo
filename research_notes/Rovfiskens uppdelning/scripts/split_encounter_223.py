"""Encounter-theory attack rates a of the two candidate predator groups (section 220 method), 223.15 library.

Usage: python3 -I split_encounter_223.py <attack_dir> <fg_predators_dir> <out_dir> [ndraw]

Copy of split_encounter.py (section-222 library). Changes for 223.15: the epifauna pair is the E1
exposed pool and carries the exposure factor e of research_notes/"Exponering av bottenbyten"
(predators -> exposed epifauna, whiting-dominated composition: e 0.39, MC 5-95 % 0.31-0.49; drawn
log-triangular here), so a(-> mobile_epifauna) = a_enc per edible tonne x e; output file
split_encounter_223.json. Everything else (stage tables, factors, ranges) is section 220 unchanged.

<attack_dir> = research_notes/"Attackrater ur motesteori" (Swedish letters); its
scripts/pred_encounter_a.py is executed up to (not including) its own evaluation, so every factor,
stage table (N per tonne, number-mean length), prey-type share and the low/central/high ranges are
exactly those of section 220. Nothing is re-tuned. What this script adds:

  * a per GROUP: the stage a values are averaged over the members of one group only.
    Variants (km2 per tonne of predator per 6-h tick, annual mean = 0.5 day + 0.5 night):
      layer    sum_k W_k m_k a_k / sum_k W_k, m_k = 1 if the stage's capability score >= 0.1
               (it forages in the prey's layer and can take it), else 0; per tonne of the whole
               group, no graded score (the recommended variant; J threshold 0.1)
      noscore  as layer but m_k = 1 for every stage with a score > 0 (section 221 convention)
      scored   sum W s a / sum W   (capability score folded into a, per tonne of the group)
      capable  sum W s a / sum W s (section 220: per tonne of the capable share)
  * the planktivore pair on DISPERSED prey (SCHOOLS = False; library convention, 220.3 decision 2),
    with the schooled central printed for reference;
  * mobile_epifauna per EDIBLE tonne: the benthos stage a divided by s_bed_edible (section 221,
    research_notes/"Infauna och rorlig epifauna"/params_split.md 2.2);
  * a new pair "-> whiting as prey" (the water-column group on the bed group): the near-bed
    juvenile-gadoid geometry of the benthivore pair with the prey layer H_whiting (J: 10 m,
    20-5; whiting live within ~5-30 m of the bed by day and partly rise at night,
    Pedersen 1999) and prey share 1 (gape is accessible_fraction); capability scores are the
    benth_fish scores as a proxy (J).
"""
import json
import math
import os
import sys

import numpy as np

attack_dir, fgp, out = sys.argv[1:4]
NDRAW = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
path = os.path.join(attack_dir, "scripts", "pred_encounter_a.py")
src = open(path).read()
head = src.split("central, detail = evaluate(level(1), detail=True)")[0]
sys.argv = [path, os.path.join(attack_dir, "tables", "pred_numbers_per_tonne.csv"),
            os.path.join(fgp, "inputs", "pred_scores.csv"), os.path.join(fgp, "tables", "pred_season.csv"),
            os.path.join(out, "_tmp_pe_223")]
G = {"__name__": "pred_encounter_a_head", "__file__": path}
exec(compile(head, path, "exec"), G)
os.rmdir(os.path.join(out, "_tmp_pe_223")) if os.path.isdir(os.path.join(out, "_tmp_pe_223")) and not os.listdir(os.path.join(out, "_tmp_pe_223")) else None
G["P"]["H_whiting"] = (20.0, 10.0, 5.0)
G["P"]["e_pred_epi"] = (0.31, 0.39, 0.49)  # 223.15 exposure factor, predators -> exposed epifauna
STG_T, a3d, score, level, stage_a = G["STG_T"], G["a3d"], G["score"], G["level"], G["stage_a"]
RNG = np.random.default_rng(2026)


def whiting_prey_a(row, q, night):
    sp, Nt, L = row.sp, row.Nt, row.L
    typ = G["TYPE"][sp]
    p = q["p_fish"]
    if typ == "bedambush":
        return a3d(q["R_strike_angler"], q["u_prey_angler"], 0.0, q["H_whiting"], Nt, p, 1.0, 1.0)
    v = q[G["SPEED"][sp]] * L
    if night and sp in G["NIGHTV"]:
        v *= q[G["NIGHTV"][sp]]
    if typ == "elasmo":
        R = q["R_fish_elasmo"]
    else:
        R = q["R_fish_dark"] if night else max(q["R_bedfish_day"], q["R_fish_dark"])
    return a3d(R, v, q["u_smallfish"], q["H_whiting"], Nt, p, q["scan"], 1.0)


PAIRS = {"planktivorous_fish": ["plank_fish"], "zooplankton": ["zoo_krill", "zoo_cop"],
         "mobile_epifauna": ["benthos"], "benthivorous_fish": ["benth_fish"], "whiting_prey": ["whiting_prey"]}


def stage_pair(row, prey_type, q, night):
    if prey_type == "whiting_prey":
        return whiting_prey_a(row, q, night)
    a = stage_a(row, prey_type, q, night)
    if prey_type == "benthos":
        a = a / q["s_bed_edible"] * q["e_pred_epi"]  # per edible tonne (f x B_epi), x exposure e (223.15)
    return a


def sc(row, prey_type):
    return score(row.sp, row.st, "benth_fish" if prey_type == "whiting_prey" else prey_type)


def group_a(q, members, pair, variant, schools=False):
    G["SCHOOLS"] = schools
    num_d = num_n = 0.0
    den = 0.0
    den_s = 0.0
    for r in STG_T:
        if r.sp not in members:
            continue
        den += r.W
        for pt in PAIRS[pair]:
            s = sc(r, pt)
            if s <= 0:
                continue
            if variant == "layer":
                m = 1.0 if s >= 0.1 else 0.0
            elif variant == "noscore":
                m = 1.0
            else:
                m = s
            if m == 0:
                continue
            ad, an = stage_pair(r, pt, q, False), stage_pair(r, pt, q, True)
            num_d += r.W * m * ad
            num_n += r.W * m * an
        if variant == "capable":
            # denominator = phi as the engine had it (zoo: mean of the two prey-type scores)
            ss = [sc(r, pt) for pt in PAIRS[pair]]
            den_s += r.W * (sum(ss) / len(ss) if pair == "zooplankton" else ss[0])
    D = den_s if variant == "capable" else den
    if D <= 0:
        return (0.0, 0.0, 0.0)
    ad, an = num_d / D, num_n / D
    return (ad, an, 0.5 * ad + 0.5 * an)


def draw():
    qq = {}
    for k, (lo, c, hi) in G["P"].items():
        if lo == hi:
            qq[k] = c
            continue
        a, b = sorted((math.log(lo), math.log(hi)))
        m = min(max(math.log(c), a), b)
        qq[k] = math.exp(RNG.triangular(a, m, b))
    return qq


ALL = ["saithe", "whiting", "mackerel", "spurdog", "horse mackerel", "hake", "anglerfish", "ling",
       "velvet belly", "pollack"]
GROUPS = {
    "pooled": ALL,
    "G1_bed": ["whiting"],
    "G1_wc": [s for s in ALL if s != "whiting"],
    "G2_bed": ["whiting", "hake", "spurdog", "anglerfish", "ling", "velvet belly"],
    "G2_wc": ["saithe", "mackerel", "horse mackerel", "pollack"],
}
VARIANTS = ["layer", "noscore", "scored", "capable"]
qc, qlo, qhi = level(1), level(0), level(2)
draws = [draw() for _ in range(NDRAW)]
res = {}
print("Encounter a per group, km2 per tonne of predator per 6-h tick (annual mean); planktivores on dispersed prey")
print(f"MC {NDRAW} log-triangular draws of the section 220 factors (+ H_whiting 20-10-5 m)\n")
for gname, mem in GROUPS.items():
    res[gname] = {}
    print("=" * 110)
    print(f"{gname}: {mem}")
    for pair in PAIRS:
        res[gname][pair] = {}
        for var in VARIANTS:
            c = group_a(qc, mem, pair, var)
            lo = group_a(qlo, mem, pair, var)[2]
            hi = group_a(qhi, mem, pair, var)[2]
            mc = np.array([group_a(dq, mem, pair, var)[2] for dq in (draws if var in ("layer", "capable") else draws[:600])])
            rat = c[1] / c[0] if c[0] > 0 else float("nan")
            p05, p50, p95 = (np.percentile(mc, [5, 50, 95]) if mc.max() > 0 else (0.0, 0.0, 0.0))
            res[gname][pair][var] = dict(central=c[2], day=c[0], night=c[1], night_day=rat, corner_lo=lo,
                                         corner_hi=hi, mc05=float(p05), mc50=float(p50), mc95=float(p95))
            print(f"  {pair:18s} {var:8s} a {c[2]:.4g}  (corners {lo:.3g}-{hi:.3g}; MC 5/50/95 {p05:.3g}/{p50:.3g}/{p95:.3g})"
                  f"  night/day {rat:.2f}")
        if pair == "planktivorous_fish":
            s = group_a(qc, mem, pair, "layer", schools=True)
            res[gname][pair]["layer_schooled"] = dict(central=s[2], night_day=s[1] / s[0] if s[0] > 0 else float("nan"))
            print(f"  {pair:18s} layer, schooled day herring (220 central): a {s[2]:.4g}  night/day {s[1] / s[0]:.2f}")
    # contribution of each member to the layer a (central)
    print("  member contributions to the layer a (central, share of the group a):")
    for pair in PAIRS:
        tot = res[gname][pair]["layer"]["central"]
        if tot <= 0:
            continue
        parts = {}
        for sp in mem:
            parts[sp] = group_a(qc, [sp], pair, "layer")[2] * (sum(r.W for r in STG_T if r.sp == sp)
                                                              / sum(r.W for r in STG_T if r.sp in mem))
        print(f"    {pair:18s} " + ", ".join(f"{k} {v / tot:.2f}" for k, v in sorted(parts.items(), key=lambda kv: -kv[1]) if v / tot >= 0.01))
    # per stage (central, layer membership) for the record
print("\nPer-stage central a (annual; epifauna per edible tonne x e 0.39; whiting_prey new):")
for r in STG_T:
    vals = []
    for pair in PAIRS:
        a = sum(0.5 * (stage_pair(r, pt, qc, False) + stage_pair(r, pt, qc, True)) for pt in PAIRS[pair])
        s = max(sc(r, pt) for pt in PAIRS[pair])
        vals.append(f"{pair[:10]} {a:.4g} (s {s:.2f})")
    print(f"  {r.sp:15s} {r.st:9s} W {r.W:.4f}: " + "; ".join(vals))
with open(os.path.join(out, "split_encounter_223.json"), "w") as fh:
    json.dump(res, fh, indent=1)
