"""Two-compartment benthic prey: accessible pool A and refuge R, per cell.

Biology only - no simulator run, no library edit. Run with ``python3 -I``.
Output: tables/refuge_equilibrium_stdout.txt (redirect by the caller or tee).

Model (per cell, annual rates; the engine would apply them per 6-h tick):

  dA/dt = r A (1 - A/K_A) + k_in R - k_out A - F_A A - C(A)
  dR/dt = r R (1 - R/K_R) - k_in R + k_out A - F_R R

  Variant S ("separate capacity", recommended): K_A = f K, K_R = (1 - f) K.
  Variant P ("pooled density dependence"): both grow at r (1 - B/K) per
  capita, B = A + R (the refuge can take over K when A is eaten down).

  Exchange: k_in = f k, k_out = (1 - f) k, so with no predation and no
  selective loss A/B relaxes to f = k_in/(k_in + k_out) at the total rate k.
  r = r_gross = r_max + sum M2 (the loader's derived NDM rate, resume 129),
  C(A) = m A, m the per-tonne-of-A predation rate.

What it prints:
  1. Limits: k -> infinity reproduces today's engine (A = f B every tick);
     k -> 0 gives two decoupled logistics.
  2. Equilibrium B*, A*/B* and delivered consumption when the predators take
     the BUDGETED flux C0 = M2 B*_today (constant flux), as a function of k.
  3. Maximum sustainable removal from A, C_max(k) = max_m m A*(m), compared
     with C0, with the removals realised in resume 223 and with the demand at
     the Holling ceiling.
  4. The k at which C_max equals each of those - the discriminating number:
     behavioural exchange (per day) vs demographic exchange (per year).
"""
import math
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar

TICKS_PER_YEAR = 1460.0

# ---------------------------------------------------------------- groups
# Library values (fgconfig/fg_library.yaml, sections 221-222) and the
# central budget of research_notes/"Infauna och rorlig epifauna".
GROUPS = {
    "benthic_infauna": dict(
        r_max=1.0, M2=0.0291, F=0.136, K=110.0, B_today=93.6,
        f=0.20,           # benthivores' accessible_fraction
        pairs={"benthivores": dict(f=0.20, M2=0.0291, h=95.9,
                                   B_pred=1.53)},
        # resume 223.3: realised M2 0.12-0.16 /yr on the infauna
        realised_223={"benthivores": (0.12 * 93.6, 0.16 * 93.6)},
    ),
    "mobile_epifauna": dict(
        r_max=0.70, M2=0.177, F=0.136, K=12.2, B_today=7.88,
        f=0.31,           # exposed share = cod's reach (all taxa)
        pairs={"benthivores": dict(f=0.31, M2=0.141, h=138.4, B_pred=1.53),
               "predators": dict(f=0.13, M2=0.036, h=147.1, B_pred=1.20)},
        # resume 223.2: removal ~6 t/km2/yr (pred 4.1, ben 1.3, trawl 0.7)
        realised_223={"predators": (4.1, 4.1), "benthivores": (1.3, 1.3)},
    ),
}


def rhs(state, g, k, m, variant, C_flux=None):
    A, R = state
    r, K, f, F = g["r"], g["K"], g["f"], g["F"]
    k_in, k_out = f * k, (1.0 - f) * k
    if variant == "S":
        gA = r * A * (1.0 - A / (f * K))
        gR = r * R * (1.0 - R / ((1.0 - f) * K))
    else:
        B = A + R
        gA = r * A * (1.0 - B / K)
        gR = r * R * (1.0 - B / K)
    pred = C_flux if C_flux is not None else m * A
    dA = gA + k_in * R - k_out * A - F * A - pred
    dR = gR - k_in * R + k_out * A - F * R
    return np.array([dA, dR])


def r_of_a(A, g, k, variant):
    """Refuge biomass on dR/dt = 0 for a given A (positive quadratic root)."""
    r, K, f, F = g["r"], g["K"], g["f"], g["F"]
    k_in, k_out = f * k, (1.0 - f) * k
    if variant == "S":
        a2 = -r / ((1.0 - f) * K)
        a1 = r - k_in - F
    else:
        a2 = -r / K
        a1 = r * (1.0 - A / K) - k_in - F
    a0 = k_out * A
    disc = a1 * a1 - 4.0 * a2 * a0
    if a0 == 0.0:
        return max(-a1 / a2, 0.0)
    return (-a1 - math.sqrt(max(disc, 0.0))) / (2.0 * a2)


def equilibrium_fast(g, k, m, variant):
    """Stable interior steady state for per-capita predation m on A: the
    largest A at which dA/dt changes sign from + to - along dR/dt = 0."""
    f, K = g["f"], g["K"]

    def gA(A):
        R = r_of_a(A, g, k, variant)
        return rhs((A, R), g, k, m, variant)[0]

    top = K
    grid = np.linspace(0.0, top, 801)
    vals = [gA(a) for a in grid]
    root = None
    for i in range(len(grid) - 1, 0, -1):
        if vals[i - 1] > 0.0 and vals[i] <= 0.0:
            root = brentq(gA, grid[i - 1], grid[i], xtol=1e-12)
            break
    if root is None:
        root = 0.0
    return root, r_of_a(root, g, k, variant)


def c_max(g, k, variant):
    """Maximum sustainable removal from A over per-capita predation m."""
    def neg(logm):
        m = math.exp(logm)
        A, _ = equilibrium_fast(g, k, m, variant)
        return -m * A
    res = minimize_scalar(neg, bounds=(math.log(1e-3), math.log(1e4)),
                          method="bounded", options={"xatol": 1e-4})
    m = math.exp(res.x)
    A, R = equilibrium_fast(g, k, m, variant)
    return -res.fun, m, A, R


def budget_equilibrium(g, k, variant):
    """Equilibrium when predation takes the budget flux C0 = M2 B_today as a
    per-capita m on A chosen so that m A* = C0 (the stable root with the
    larger A). Returns None if C0 is not deliverable."""
    C0 = g["M2"] * g["B_today"]
    cm, m_at_max, _, _ = c_max(g, k, variant)
    if cm < C0:
        return None
    # on the stable branch m < m_at_max: m A*(m) increases from 0 to cm
    def resid(logm):
        m = math.exp(logm)
        A, _ = equilibrium_fast(g, k, m, variant)
        return m * A - C0
    lo, hi = math.log(1e-4), math.log(m_at_max)
    logm = brentq(resid, lo, hi, xtol=1e-6)
    m = math.exp(logm)
    A, R = equilibrium_fast(g, k, m, variant)
    return m, A, R


def main():
    out = []
    p = out.append
    p("Two-compartment benthic prey: accessible A and refuge R (annual rates)")
    p("r = r_gross = r_max + sum M2; k_in = f k, k_out = (1 - f) k")
    p("")
    ks = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 365.0, 1460.0, 5840.0]
    for name, g0 in GROUPS.items():
        g = dict(g0)
        g["r"] = g["r_max"] + g["M2"]
        C0 = g["M2"] * g["B_today"]
        p("=" * 78)
        p(f"{name}: r_max {g['r_max']}, sum M2 {g['M2']}, r_gross "
          f"{g['r']:.3f}, F {g['F']}, K {g['K']}, B*_today {g['B_today']}, "
          f"f {g['f']}")
        Bstar_logistic = g["K"] * (1 - (g["F"] + g["M2"]) / g["r"])
        p(f"  today's single logistic: B* = K (1 - (F + M2)/r) = "
          f"{Bstar_logistic:.2f} (library equilibrium map mean "
          f"{g['B_today']}); budget flux C0 = M2 B* = {C0:.3f} t/km2/yr")
        # Holling-ceiling demand of each pair's predator group
        for pn, pr in g["pairs"].items():
            ceil = TICKS_PER_YEAR / pr["h"]
            p(f"  {pn}: ceiling Q/B {ceil:.1f} t prey/t/yr; at the ceiling "
              f"with B_pred {pr['B_pred']} t/km2 the demand is "
              f"{ceil * pr['B_pred']:.1f} t/km2/yr (x2 if the group "
              f"aggregates: {2 * ceil * pr['B_pred']:.1f})")
        for pn, (lo, hi) in g["realised_223"].items():
            p(f"  resume 223 realised removal by {pn}: {lo:.1f}-{hi:.1f} "
              f"t/km2/yr")
        surplus = g["r"] * g["K"] / 4.0
        surplus_net = (g["r"] - g["F"]) ** 2 * g["K"] / (4 * g["r"])
        p(f"  whole-pool maximum surplus r K/4 = {surplus:.2f}; net of the "
          f"trawl (r - F)^2 K/(4 r) = {surplus_net:.2f} t/km2/yr")
        for variant in ("S", "P"):
            p("")
            p(f"  Variant {variant} "
              f"({'separate K_A = fK, K_R = (1-f)K' if variant == 'S' else 'pooled density dependence r(1 - B/K)'})")
            p(f"  {'k /yr':>8} {'exch. time':>11} {'C_max':>7} {'C_max/C0':>8} "
              f"{'A/B at C0':>9} {'B*(C0)':>7} {'B*/B_today':>10} {'R*(C0)':>7}")
            for k in ks:
                cm, m_star, A_m, R_m = c_max(g, k, variant)
                be = budget_equilibrium(g, k, variant)
                tau = (f"{365.0 / k:8.1f} d" if 0 < k < 365 * 5
                       else (f"{24 * 365.0 / k:8.1f} h" if k > 0 else "     inf"))
                if be is None:
                    p(f"  {k:8.1f} {tau:>11} {cm:7.2f} {cm / C0:8.2f} "
                      f"{'n/a':>9} {'n/a':>7} {'n/a':>10} {'n/a':>7}")
                else:
                    m, A, R = be
                    B = A + R
                    p(f"  {k:8.1f} {tau:>11} {cm:7.2f} {cm / C0:8.2f} "
                      f"{A / B:9.3f} {B:7.2f} {B / g['B_today']:10.3f} "
                      f"{R:7.2f}")
            # k at which C_max equals given removals
            targets = {"C0 (budget)": C0}
            for pn, (lo, hi) in g["realised_223"].items():
                targets[f"223 {pn}"] = hi
            tot223 = sum(hi for (_, hi) in g["realised_223"].values())
            targets["223 all fish"] = tot223
            for pn, pr in g["pairs"].items():
                targets[f"ceiling {pn}"] = TICKS_PER_YEAR / pr["h"] * pr["B_pred"]
            p("  k (per yr) at which C_max(k) equals:")
            for tn, tv in targets.items():
                c_lo = c_max(g, 1e-6, variant)[0]
                c_hi = c_max(g, 1e5, variant)[0]
                if tv <= c_lo:
                    p(f"    {tn:22s} {tv:7.2f}: already met at k = 0 "
                      f"(C_max(0) = {c_lo:.2f})")
                elif tv >= c_hi:
                    p(f"    {tn:22s} {tv:7.2f}: never (C_max(inf) = "
                      f"{c_hi:.2f}, the whole-pool surplus)")
                else:
                    kk = math.exp(brentq(
                        lambda lk: c_max(g, math.exp(lk), variant)[0] - tv,
                        math.log(1e-6), math.log(1e5), xtol=1e-4))
                    p(f"    {tn:22s} {tv:7.2f}: k = {kk:9.3f} /yr "
                      f"(exchange time {365.0 / kk:8.1f} d)")
        p("")
    text = "\n".join(out)
    print(text)


if __name__ == "__main__":
    main()
