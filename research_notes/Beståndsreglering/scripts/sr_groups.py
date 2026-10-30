#!/usr/bin/env python3
"""Aggregate the per-stock compensation at today's SSB to the Mareld fish groups.

Usage:
    python3 -I scripts/sr_groups.py COMP_CSV DRAWS_CSV OUT_CSV
    e.g. python3 -I research_notes/Beståndsreglering/scripts/sr_groups.py \
             research_notes/Beståndsreglering/tables/sr_compensation.csv \
             research_notes/Beståndsreglering/tables/sr_boot_draws.csv \
             research_notes/Beståndsreglering/tables/sr_groups.csv

Runs four variants: PERIOD "post2000" (today's regime) or "full", times COLUMN
"elast_avg" (AICc average of the stationary models: productivity assumed constant)
or "elast_avg_all" (AICc average over the stationary, trend and, in the full
period, shift models: productivity allowed to change).
Member elasticities are clipped to [-1, 0], the range a Beverton-Holt
recruitment term can represent (Ricker over-compensation < -1 and positive
"depensatory" values from extrapolated Ricker fits are cut); the unclipped
point value is kept in the column elast_raw.

Each member takes the elasticity e = d ln(R/S)/d ln S at today's SSB of its
stock (or, for a member without an assessment, of an analogue stock - grade A -
or a judgement value - grade J). Group e = sum_i w_i e_i with the member biomass
shares w_i (renormalised over the members listed). Group x = -e/(1+e) is the
Beverton-Holt S_today/b that gives the same elasticity, and 1+e the BH ratio
(R/S today)/(R/S at the origin). Uncertainty: member bootstrap draws combined
independently (draw j of every stock); judgement members get a uniform draw over
their stated range.
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np

# member, share, source stock (or None), grade, judgement range (lo, hi) if no stock
GROUPS = {
    "planktivorous_fish": [
        ("herring NSAS (77 % of herring in the box)", 0.79 * 0.77, "her.27.3a47d", "D", None),
        ("herring WBSS (23 % of herring in the box)", 0.79 * 0.23, "her.27.20-24", "D", None),
        ("blue whiting", 0.105, "whb.27.1-91214", "D", None),
        ("Norway pout", 0.075, "nop.27.3a4", "D", None),
        ("sprat", 0.03, "spr.27.3a4", "D", None),
    ],
    "benthivorous_fish": [
        ("haddock", 0.386, "had.27.46a20", "D", None),
        ("cod (Viking substock)", 0.314, "cod.27.46a7d20_viking", "D", None),
        ("long rough dab (as plaice + witch)", 0.126, "flatfish_analogue", "A", None),
        ("dab (as plaice + witch)", 0.045, "flatfish_analogue", "A", None),
        ("plaice", 0.038, "ple.27.420", "D", None),
        ("witch", 0.017, "wit.27.3a47d", "D", None),
        ("starry ray", 0.015, None, "J", (-0.6, 0.0)),
        ("hagfish", 0.013, None, "J", (-1.0, 0.0)),
        ("others", 0.046, "group_mean", "J", None),
    ],
    "predatory_fish": [
        ("saithe", 0.651, "pok.27.3a46", "D", None),
        ("mackerel", 0.128, "mac.27.nea", "D", None),
        ("spurdog (annex pup-survival term)", 0.080, None, "A", (-0.87, -0.25)),
        ("horse mackerel (western stock)", 0.055, "hom.27.2a3a4a5b6a7a-ce-k8", "D", None),
        ("hake (SS3)", 0.040, "hke.27.3a46-8abd", "D", None),
        ("anglerfish (SS3)", 0.022, "anf.27.3a46", "D", None),
        ("ling + velvet belly + pollack", 0.024, "group_mean", "J", None),
    ],
    "whiting": [
        ("whiting (whg.27.47d)", 1.0, "whg.27.47d", "D", None),
    ],
}
SEED = 20261030


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    comp = {(r["stock"], r["period"]): r for r in csv.DictReader(open(sys.argv[1]))}
    alld = list(csv.DictReader(open(sys.argv[2])))
    rows = []
    for period in ("post2000", "full"):
        for col in ("elast_avg", "elast_avg_all"):
            rows += variant(comp, alld, period, col)
    with open(sys.argv[3], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def clip(v):
    return np.clip(v, -1.0, 0.0)


def variant(comp, alld, period, col):
    draws = defaultdict(list)
    for r in alld:
        if r["period"] == period:
            draws[r["stock"]].append(float(r[col]))
    nd = min(len(v) for v in draws.values())
    rng = np.random.default_rng(SEED)

    def raw(stock):
        return float(comp[(stock, period)][col])

    def point(stock):
        return float(clip(raw(stock)))

    def dr(stock):
        return clip(np.array(draws[stock][:nd]))

    flat_pt = 0.5 * (point("ple.27.420") + point("wit.27.3a47d"))
    flat_dr = 0.5 * (dr("ple.27.420") + dr("wit.27.3a47d"))
    rows = []
    for g, members in GROUPS.items():
        known = [(m, w, s) for m, w, s, _, _ in members if s not in (None, "group_mean")]
        wk = sum(w for _, w, _ in known)
        mean_pt = sum(w * (flat_pt if s == "flatfish_analogue" else point(s)) for _, w, s in known) / wk
        mean_dr = sum(w * (flat_dr if s == "flatfish_analogue" else dr(s)) for _, w, s in known) / wk
        tot_w = sum(w for _, w, _, _, _ in members)
        e_pt = 0.0
        e_dr = np.zeros(nd)
        for m, w, s, grade, rng_j in members:
            if s is None:
                p, d = 0.5 * (rng_j[0] + rng_j[1]), rng.uniform(rng_j[0], rng_j[1], nd)
            elif s == "group_mean":
                p, d = mean_pt, mean_dr
            elif s == "flatfish_analogue":
                p, d = flat_pt, flat_dr
            else:
                p, d = point(s), dr(s)
            e_pt += w / tot_w * p
            e_dr += w / tot_w * d
            rows.append(dict(group=g, member=m, share=f"{w / tot_w:.3f}",
                             stock=s or "none", grade=grade, period=period, column=col,
                             elast_raw=(f"{raw(s):.3f}" if s not in (None, "group_mean",
                                                                     "flatfish_analogue") else ""),
                             elast=f"{p:.3f}", elast_lo=f"{np.quantile(d, .05):.3f}",
                             elast_hi=f"{np.quantile(d, .95):.3f}",
                             x_bh="", ratio_bh=""))
        ec = min(e_pt, -1e-9)
        x = -ec / (1 + ec) if ec > -1 else math.inf
        rows.append(dict(group=g, member="GROUP", share="1.000", stock="", grade="",
                         period=period, column=col, elast_raw="", elast=f"{e_pt:.3f}",
                         elast_lo=f"{np.quantile(e_dr, .05):.3f}",
                         elast_hi=f"{np.quantile(e_dr, .95):.3f}",
                         x_bh=f"{x:.3g}" if math.isfinite(x) else "inf",
                         ratio_bh=f"{max(1 + e_pt, 0):.3f}"))
        print(g, period, col, "e %.3f [%.3f, %.3f]" % (e_pt, np.quantile(e_dr, .05),
                                                      np.quantile(e_dr, .95)), "x", x)
    return rows


if __name__ == "__main__":
    main()
