"""Refuge persistence, budget rule and taxon-resolved compartment rates.

Companion to refuge_equilibrium.py (same model, variant S: separate
capacities K_A = f K, K_R = (1 - f) K, exchange k_in = f k, k_out = (1-f) k).
Run with ``python3 -I`` from the note folder; output to
tables/refuge_persistence_stdout.txt.

1. Unlimited predation on A (m -> infinity, A -> 0): the refuge obeys
   dR/dt = r R (1 - R/K_R) - (k_in + F) R, so
       R_inf = K_R (1 - (k_in + F)/r)    (> 0 iff k_in < r - F)
   and the removal that the fish can sustain from it is k_in R_inf
   (maximised at k_in = (r - F)/2: (r - F)^2 K_R/(4 r)).
   The refuge protects the group on the scale of years only when
   k_in < r - F; a behavioural exchange (k_in of order 1/day) empties it.
2. The growth rate the loader must use so that the whole group still sits at
   today's B* when the fish take the budget flux C0 = M2 B* from A only
   (equilibrium of both compartments), as a function of k.
3. A per-cell time course: how fast a cell is emptied when an aggregated
   predator group eats A at its Holling ceiling (benthivores doubled on a
   rich bed, resume 223.8), for behavioural vs demographic k, 6-h ticks.
4. Taxon-resolved epifauna: the exposed and refuge pools have different
   members (exposed = crabs, shrimps, mysids, emerged Nephrops; refuge =
   Nephrops in burrows): compartment-weighted r_max, to show the bias of one
   group r on both pools.
"""
import math

import numpy as np
from scipy.optimize import brentq

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "refeq", os.path.join(HERE, "refuge_equilibrium.py"))
refeq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refeq)

TICKS = 1460.0


def persistence(g, k):
    r, K, f, F = g["r"], g["K"], g["f"], g["F"]
    k_in = f * k
    K_R = (1 - f) * K
    R_inf = max(K_R * (1 - (k_in + F) / r), 0.0)
    return R_inf, k_in * R_inf


def r_needed(g, k):
    """r_gross so that B*(C0) = today's single-logistic B*."""
    B_target = g["K"] * (1 - (g["F"] + g["M2"]) / (g["r_max"] + g["M2"]))
    C0 = g["M2"] * B_target

    def resid(r):
        gg = dict(g, r=r)
        cm, m_at, _, _ = refeq.c_max(gg, k, "S")
        if cm < C0:
            return -B_target
        def res2(logm):
            m = math.exp(logm)
            A, _ = refeq.equilibrium_fast(gg, k, m, "S")
            return m * A - C0
        logm = brentq(res2, math.log(1e-5), math.log(m_at), xtol=1e-7)
        A, R = refeq.equilibrium_fast(gg, k, math.exp(logm), "S")
        return A + R - B_target
    lo, hi = g["r_max"] + g["M2"] * 0.2, 6.0
    try:
        return brentq(resid, lo, hi, xtol=1e-5), B_target
    except ValueError:
        return float("nan"), B_target


def time_course(g, k, demand_per_year, years=3.0, start_frac=None):
    """Per-cell, 6-h ticks. Predation removes min(demand, cap) from A each
    tick (demand at the Holling ceiling of an aggregated group; cap 0.9 A,
    the engine's MAX_HARVEST_FRAC order). Exchange exact, growth explicit."""
    r, K, f, F = g["r"], g["K"], g["f"], g["F"]
    dt = 1.0 / TICKS
    B0 = g["B_today"] if start_frac is None else start_frac * K
    A, R = f * B0, (1 - f) * B0
    ek = math.exp(-k * dt)
    rows = []
    for t in range(int(years * TICKS) + 1):
        if t % int(TICKS / 4) == 0:
            rows.append((t / TICKS, A, R))
        eat = min(demand_per_year * dt, 0.9 * A)
        A -= eat
        gA = r * A * (1 - A / (f * K)) - F * A
        gR = r * R * (1 - R / ((1 - f) * K)) - F * R
        A = max(A + dt * gA, 0.0)
        R = max(R + dt * gR, 0.0)
        B = A + R
        A_new = f * B + (A - f * B) * ek
        R, A = B - A_new, A_new
    return rows


def main():
    out = []
    p = out.append
    ks = [0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 36.5, 365.0, 1460.0]
    for name, g0 in refeq.GROUPS.items():
        g = dict(g0)
        g["r"] = g["r_max"] + g["M2"]
        p("=" * 78)
        p(f"{name}: r_gross {g['r']:.3f}, F {g['F']}, f {g['f']}, K {g['K']}")
        thr = (g["r"] - g["F"])
        p(f"  refuge persists under unlimited predation on A iff k_in < r - F "
          f"= {thr:.3f} /yr, i.e. k < (r - F)/f = {thr / g['f']:.2f} /yr "
          f"(exchange time 1/k > {365 * g['f'] / thr:.0f} d; refuge "
          f"residence 1/k_in > {365 / thr:.0f} d)")
        p(f"  {'k /yr':>8} {'k_in /yr':>9} {'1/k_in d':>9} {'R_inf':>7} "
          f"{'R_inf/K_R':>9} {'k_in R_inf':>10} {'r_needed':>9}")
        for k in ks:
            R_inf, flux = persistence(g, k)
            rn, Bt = r_needed(g, k)
            k_in = g["f"] * k
            p(f"  {k:8.1f} {k_in:9.3f} {365 / k_in:9.1f} {R_inf:7.2f} "
              f"{R_inf / ((1 - g['f']) * g['K']):9.3f} {flux:10.3f} "
              f"{rn:9.3f}")
        p(f"  (r_needed: r_gross that keeps B* at {Bt:.2f} when C0 is taken "
          f"from A only; the loader's rule gives {g['r']:.3f})")
        # time courses
        for pn, pr in g["pairs"].items():
            demand = 2 * TICKS / pr["h"] * pr["B_pred"] * pr["f"] / g["f"]
            p(f"  Time course, one cell: {pn} aggregated x2 at their ceiling, "
              f"demand on A {demand:.1f} t/km2/yr (x pair share "
              f"{pr['f'] / g['f']:.2f} of A)")
            for k in (1460.0, 36.5, 2.0, 0.5):
                rows = time_course(g, k, demand)
                txt = "  ".join(f"{t:.2f}y B {a + rr:6.2f}" for t, a, rr in
                                rows[::2])
                p(f"    k {k:7.1f}: {txt}")
        p("")
    # 4. taxon-resolved epifauna
    p("=" * 78)
    p("Taxon-resolved mobile_epifauna (stock t/km2, exposed share, r_max "
      "from research_notes/'Infauna och rorlig epifauna' 1.4):")
    taxa = [("Norway lobster", 4.57, 0.10, 0.34),
            ("northern shrimp", 0.56, 0.70, 1.2),
            ("crabs, hermits, Munida", 2.0, 0.60, 0.9),
            ("small shrimps, mysids", 0.5, 0.60, 1.86)]
    B = sum(t[1] for t in taxa)
    A = sum(t[1] * t[2] for t in taxa)
    rA = sum(t[1] * t[2] * t[3] for t in taxa) / A
    rR = sum(t[1] * (1 - t[2]) * t[3] for t in taxa) / (B - A)
    rB = sum(t[1] * t[3] for t in taxa) / B
    for t in taxa:
        p(f"  {t[0]:24s} B {t[1]:5.2f}  exposed {t[1] * t[2]:5.2f} "
          f"({t[1] * t[2] / A:5.1%} of A)  refuge {t[1] * (1 - t[2]):5.2f} "
          f"({t[1] * (1 - t[2]) / (B - A):5.1%} of R)  r_max {t[3]}")
    p(f"  B {B:.2f}, A {A:.2f} (f = {A / B:.3f}); stock-weighted r_max {rB:.2f}"
      f" (group 0.70 by K matching); exposed-weighted {rA:.2f}; "
      f"refuge-weighted {rR:.2f}")
    pred_taxa = [("northern shrimp", 0.56 * 0.7), ("small shrimps, mysids",
                 0.5 * 0.6), ("small crabs (1/4 of crabs)", 0.25 * 2.0 * 0.6)]
    Ap = sum(v for _, v in pred_taxa)
    p(f"  predators' prey in A: {Ap:.2f} t/km2 = {Ap / A:.2f} of A = "
      f"{Ap / B:.3f} of B (library accessible_fraction 0.13)")
    print("\n".join(out))


if __name__ == "__main__":
    main()
