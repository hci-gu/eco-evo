"""Per-pair stage access [f_J, f_A]: the share of the juvenile and of the adult tonnes inside each
member's prey-size window, member-weighted by the member's share of the pair's attack rate.

Usage: python3 -I stage_access.py <stage_length_biomass.csv> <out_dir>

Prey stages: tables/stage_length_biomass.csv (juvenile_share_by_quarter.py; q-corrected kg/h by 1-cm
class; J = below the species' length at age 1 in Q3). Q1 and Q3 from 2001-25, Q2 and Q4 from 1991-96
(the only years with those quarters); the shares inside a window are ratios within one quarter, and
the annual value is the mean of the four. Species weights inside a prey group: planktivores the pool
weights, benthivores the model's member weights, restricted to the species each member eats (diet
identity, section 223.31). Species without a length at age 1 (long rough dab, dab, lemon sole, rays,
gurnard) are all adult in that table and enter the A shares.

A member's window is a maximum prey length L_max(predator length) = ratio x length (fish), or a fixed
prey-length range (mammals), from evidence_prey_size.md; the predator's length distribution in the box
weights it (fish). f = share of the stage's biomass inside the window; pair f = sum over members of
(member's share of the pair's a) x (member's f).
"""
import os
import sys

import numpy as np
import pandas as pd

sl, out = sys.argv[1:3]
sl = os.path.abspath(sl)
os.makedirs(out, exist_ok=True)
SL = pd.read_csv(sl)
POOL_W = {1: {"herring": 0.88, "blue whiting": 0.0, "Norway pout": 0.11, "sprat": 0.01},
          2: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030},
          3: {"herring": 0.78, "blue whiting": 0.15, "Norway pout": 0.04, "sprat": 0.03},
          4: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030}}
BEN_W = {"haddock": 38.6, "cod": 31.4, "long rough dab": 12.6, "dab": 4.5, "plaice": 3.8, "witch": 1.7,
         "starry ray": 1.5, "grey gurnard": 0.9, "lemon sole": 0.9}
PERIOD = {1: "2001-25", 2: "1991-96", 3: "2001-25", 4: "1991-96"}
PLK = set(POOL_W[1])
GADOIDS = {"haddock", "cod"}
FLAT = {"long rough dab", "dab", "plaice", "witch", "starry ray", "grey gurnard", "lemon sole"}

# Windows: filled from evidence_prey_size.md. ("ratio", r) = prey length <= r x predator length over the
# predator's box length distribution (lengths, weights); ("range", lo, hi) = prey lengths lo-hi cm.
PLB = pd.read_csv(os.path.join(os.path.dirname(sl), "..", "..", "Fiskgrupper i rutan", "predators", "tables",
                               "pred_length_biomass.csv"))


def plen(name, col="pooled"):
    t = PLB[PLB.common == name]
    return (t.L5.values + 2.5, (t[col] / t[col].sum()).values)


cod = SL[(SL.species == "cod") & (SL.period == "2001-25")].groupby("L_cm")[["J", "A"]].sum().sum(axis=1)
cod = cod[cod.index >= 20]  # piscivory from ~20 cm
PRED_LEN = {"saithe": plen("saithe"), "spurdog": plen("spurdog"), "hake": plen("hake"),
            "anglerfish": plen("anglerfish"), "mackerel": plen("mackerel", "q3"),
            "horse mackerel": plen("horse mackerel", "q3"), "whiting": plen("whiting"),
            "cod": (cod.index.values + 0.5, (cod / cod.sum()).values)}
# evidence_prey_size.md (p95 prey/predator length; ranges for the mammals; velvet belly a fixed 11 cm, J)
RATIO = {"saithe": 0.30, "spurdog": 0.34, "hake": 0.50, "anglerfish": 0.57, "mackerel": 0.33,
         "horse mackerel": 0.31, "whiting": 0.39, "cod": 0.30}
if len(sys.argv) > 3:  # sensitivity: name=ratio,...
    for kv in sys.argv[3].split(","):
        k, v = kv.split("=")
        RATIO[k] = float(v)
WINDOWS = {m: ("ratio", r, m) for m, r in RATIO.items()}
WINDOWS["velvet belly"] = ("range", 0.0, 11.0)
WINDOWS["porpoise"] = ("range", 0.0, 40.0)
WINDOWS["seal"] = ("range", 7.0, 35.0)
# Pairs: (prey group, [(member, share of the pair's a, diet identity set or None = all)]).
# Member a shares: research_notes/"Rovfiskens uppdelning" split_encounter_223 (planktivores) and
# 223.31 (benthivores: saithe 0.77, spurdog 0.10, hake 0.07, velvet belly 0.04, anglerfish 0.01).
PAIRS = {
    "water column -> benthivores": ("ben", [("saithe", 0.77, GADOIDS), ("spurdog", 0.10, FLAT | {"haddock"}),
                                            ("hake", 0.07, None), ("velvet belly", 0.04, None),
                                            ("anglerfish", 0.01, None)]),
    "whiting -> benthivores": ("ben", [("whiting", 1.0, None)]),
    "water column -> planktivores": ("plk", [("mackerel", 0.49, None), ("saithe", 0.33, None),
                                             ("horse mackerel", 0.12, None), ("hake", 0.03, None),
                                             ("spurdog", 0.02, None), ("velvet belly", 0.01, None)]),
    "benthivores -> planktivores": ("plk", [("cod", 1.0, None)]),
    "porpoises -> planktivores": ("plk", [("porpoise", 1.0, None)]),
    "seals -> benthivores": ("ben", [("seal", 1.0, None)]),
    "seals -> planktivores": ("plk", [("porpoise", 1.0, None)]),
    "whiting -> planktivores": ("plk", [("whiting", 1.0, None)]),
    "porpoises -> benthivores": ("ben", [("porpoise", 1.0, None)]),
    # resume 230: whiting as prey (its own juvenile compartment). Water column members by their share
    # of the pair's a (research_notes/"Rovfiskens uppdelning" split_encounter_mack: saithe 0.77, spurdog
    # 0.10, hake 0.07, velvet belly 0.04, anglerfish 0.01); cod as the benthivores' piscivores.
    "water column -> whiting": ("wht", [("saithe", 0.77, None), ("spurdog", 0.10, None), ("hake", 0.07, None),
                                        ("velvet belly", 0.04, None), ("anglerfish", 0.01, None)]),
    "benthivores -> whiting": ("wht", [("cod", 1.0, None)]),
    "seals -> whiting": ("wht", [("seal", 1.0, None)]),
    "porpoises -> whiting": ("wht", [("porpoise", 1.0, None)]),
}


def share_inside(q, group, species_set, window, stage):
    d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q)]
    W = {"plk": POOL_W[q], "ben": BEN_W, "wht": {"whiting": 1.0}}[group]
    tot = inside = 0.0
    for sp, w in W.items():
        if w <= 0 or (species_set is not None and sp not in species_set):
            continue
        e = d[d.species == sp]
        if e.empty or e[stage].sum() <= 0:
            continue
        L = e.L_cm.values + 0.5
        v = e[stage].values / e[stage].sum()
        if window[0] == "ratio":
            Ls, ws = PRED_LEN[window[2]]
            acc = sum(wp * v[L <= window[1] * Lp].sum() for Lp, wp in zip(Ls, ws))
        else:
            acc = v[(L >= window[1]) & (L <= window[2])].sum()
        tot += w
        inside += w * acc
    return inside / tot if tot > 0 else np.nan


def main():
    rows = []
    for pair, (group, members) in PAIRS.items():
        for stage in ("J", "A"):
            fq = []
            for q in (1, 2, 3, 4):
                f = sum(share * share_inside(q, group, ident, WINDOWS[m], stage) for m, share, ident in members)
                fq.append(f)
            rows.append(dict(pair=pair, stage=stage, Q1=fq[0], Q2=fq[1], Q3=fq[2], Q4=fq[3], mean=float(np.nanmean(fq))))
    R = pd.DataFrame(rows)
    piv = R.pivot(index="pair", columns="stage", values="mean")
    print("pair stage_access [f_J, f_A] (annual mean) and f_J / f_A:")
    for pr, r in piv.iterrows():
        print(f"  {pr:30s} [{r.J:.3f}, {r.A:.3f}]  ratio {r.J / r.A if r.A > 0 else float('inf'):.1f}")
    R.to_csv(os.path.join(out, "stage_access.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(R.round(3).to_string(index=False))


if __name__ == "__main__" and PAIRS:
    main()
