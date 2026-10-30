"""Encounter a of a split-out mackerel group (mackerel + horse mackerel) and the remaining water column.

Usage: python3 -I split_encounter_mack.py <attack_dir> <fg_predators_dir> <out_dir> [ndraw] [densities]

[densities] = "mackerel=0.80,horse mackerel=0.05" (t/km2, field levels; research_notes/"Makrillens
tathet" (Swedish letters: a-umlaut)): the members' weights inside every group are rescaled so their
biomass ratio is the field one (the stage W of the GOV tables x density / GOV density); outputs get a
"_field" suffix. G1_wc is then no longer the reproduction check.

Runs split_encounter_223.py unchanged (section 220 method, 223.15 factors, layer convention) with the
groupings replaced by G1_wc (today's water column, the reproduction check: 0.01171 / 0.000604 /
0.0532 / 0.01064) and its split G3 = mackerel group + rest. Output split_encounter_mack.json.

The new pair rest water column -> mackerel group (spurdog, hake, anglerfish, ling, velvet belly eat
mackerel and horse mackerel; split_fields_223 WCOMP): the encounter is the planktivore pair's (pelagic
schooling fish prey, dispersed; J) summed over those members only, per tonne of the WHOLE rest group;
the prey-side accessible_fraction is the share of the mackerel group's biomass inside each member's
prey-size window (evidence_prey_size.md: p95 prey/predator length spurdog 0.34, hake 0.50, anglerfish
0.57, ling as hake 0.50 (J), velvet belly a fixed 11 cm) over the box length mixes
("Fiskgrupper i rutan"/predators/tables/pred_length_biomass.csv), weighted by each member's share of
the pair's a - the method of "Arsungarnas andel" (A-ring) stage_access.py.
"""
import os
import sys

ARGV = list(sys.argv)
DENS = dict((k, float(v)) for k, v in (kv.split("=") for kv in ARGV[5].split(","))) if len(ARGV) > 5 else {}
SUFFIX = "_field" if DENS else ""
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "split_encounter_223.py")
src = open(path).read()
old_groups = src[src.index("GROUPS = {"):src.index("VARIANTS = [")]
new_groups = '''GROUPS = {
    "G1_wc": [s for s in ALL if s != "whiting"],
    "G3_mackerel": ["mackerel", "horse mackerel"],
    "G3_rest_wc": ["saithe", "spurdog", "hake", "anglerfish", "ling", "velvet belly", "pollack"],
}
'''
src = src.replace(old_groups, new_groups).replace('"split_encounter_223.json"', f'"split_encounter_mack{SUFFIX}.json"')
if DENS:
    # GOV long-run stage weights per species (pooled W): mackerel 0.0802, horse mackerel 0.0345 of the pool,
    # i.e. B_gov = 1.20 x W / sum(W of the ten members); the multiplier puts the field ratio in.
    src = src.replace("r.W", "(r.W * WMUL.get(r.sp, 1.0))")
    src = src.replace("RNG = np.random.default_rng(2026)", "RNG = np.random.default_rng(2026)\n"
                      "_WS = {sp: sum(r.W for r in STG_T if r.sp == sp) for sp in DENS_IN}\n"
                      "WMUL = {sp: DENS_IN[sp] / _WS[sp] for sp in DENS_IN}\n"
                      "print('field density weights:', {k: round(v, 3) for k, v in WMUL.items()})")
G = {"__name__": "__main__", "__file__": path, "DENS_IN": DENS}
exec(compile(src, path, "exec"), G)

import json
import numpy as np
import pandas as pd

EATERS = ["spurdog", "hake", "anglerfish", "ling", "velvet belly"]
REST = G["GROUPS"]["G3_rest_wc"]
W_rest = sum(r.W for r in G["STG_T"] if r.sp in REST)
W_eat = sum(r.W for r in G["STG_T"] if r.sp in EATERS)
ad, an, a = G["group_a"](G["qc"], EATERS, "planktivorous_fish", "noscore")
scale = W_eat / W_rest
lo = G["group_a"](G["qlo"], EATERS, "planktivorous_fish", "noscore")[2] * scale
hi = G["group_a"](G["qhi"], EATERS, "planktivorous_fish", "noscore")[2] * scale
print("=" * 110)
print(f"rest water column -> mackerel group (eaters {EATERS}, noscore: every eater counts; per whole rest tonne)")
print(f"  a {a * scale:.4g} (corners {lo:.3g}-{hi:.3g})  night/day {an / ad:.2f}")
parts = {sp: G["group_a"](G["qc"], [sp], "planktivorous_fish", "noscore")[2]
         * sum(r.W for r in G["STG_T"] if r.sp == sp) / W_eat for sp in EATERS}
tot = sum(parts.values())
print("  member shares of the a: " + ", ".join(f"{k} {v / tot:.3f}" for k, v in parts.items()))
LB = pd.read_csv(os.path.join(ARGV[2], "tables", "pred_length_biomass.csv"))


def dist(sp):
    t = LB[LB.common == sp]
    return t.L5.values + 2.5, (t.pooled / t.pooled.sum()).values


RATIO = {"spurdog": 0.34, "hake": 0.50, "anglerfish": 0.57, "ling": 0.50}
prey_w = ({k: v / sum(DENS.values()) for k, v in DENS.items()} if DENS
          else {"mackerel": 0.6993, "horse mackerel": 0.3007})
fs = {}
for sp in EATERS:
    acc = 0.0
    for prey, wp in prey_w.items():
        Lq, vq = dist(prey)
        if sp == "velvet belly":
            acc += wp * vq[Lq <= 11.0].sum()
        else:
            Lp, wpred = dist(sp)
            acc += wp * sum(wpp * vq[Lq <= RATIO[sp] * L].sum() for L, wpp in zip(Lp, wpred))
    fs[sp] = acc
f = sum(parts[sp] / tot * fs[sp] for sp in EATERS)
print("  prey-size window share of the mackerel group's biomass: " + ", ".join(f"{k} {v:.3f}" for k, v in fs.items()))
print(f"  pair accessible_fraction (a-weighted) {f:.3f}")
json.dump(dict(a=a * scale, a_lo=lo, a_hi=hi, night_day=an / ad, member_share={k: v / tot for k, v in parts.items()},
               f_member=fs, f=f), open(os.path.join(ARGV[3], f"split_encounter_mack_rest_on_mackerel{SUFFIX}.json"), "w"), indent=1)
