"""Field two-stage box model: recruitment schedule rho(t), growth ratio kappa and graduation gamma.

Usage: python3 -I stage_box_model.py <sms summary.out> <size_stage_share_by_group.csv> [Y0 Y1]

Field rates per stage from SMS 2023 (years Y0-Y1, default 2013-22), stage J = age 0 + 1 (the
size-defined stage below the length at age 1 in Q3, juvenile_share_by_quarter.py, as ages): per-tonne somatic growth g (from weight at age), total mortality Z = M1 + M2 + F, both
biomass-weighted, for J and for the adults A. Group rates weight the species by their box biomass
(planktivores herring 0.79 / pout 0.08 / sprat 0.03, blue whiting counted as adult herring-like;
benthivores haddock 0.55 / cod 0.45 of the SMS-covered part).

Monthly box model, steady annual cycle:
  dJ/dt = R(t) + (g_J - Z_J - gamma) J
  dA/dt = gamma J + (g_A - Z_A) A + s A            (s: the stock's own compensation, solved so the
                                                    total B repeats each year)
with R(t) = R0 w(t), w the arrival window (Jun 0.15, Jul 0.35, Aug 0.35, Sep 0.15: haddock settle mid-Jul
to mid-Aug, herring 0-group from Jun-Jul; age 0 absent from the May survey). R0 is solved so the
annual-mean J/B equals the target (the box survey, 2001-25 Q1/Q3 mean). Graduation gamma is the
stage-structured biomass model's maturation rate (de Roos, Schellekens, van Kooten, van de Wolfshaar,
Claessen & Persson 2008, Theor. Popul. Biol. 73: 47-62): gamma = (nu - mu) / (1 - z^(1 - mu/nu)), nu
the juvenile per-tonne growth, mu its mortality, z = mass at entry / mass at graduation (SMS weight at
age 0 in Q3 over age 1 in Q3). A constant exponential gamma below nu - mu makes J self-sustaining
(it grows faster than it dies and leaves) - the 1 /yr guess failed that way. The quarterly J/B of the
cycle is printed against the survey (2001-25 Q1 and Q3; 1991-96 Q1-Q4 for the shape).

Model parameters out of the cycle: kappa = g_J / g_A (J tonnes take kappa times an adult tonne's share
of the group's new tonnes) and rho(t) = R(t) / (g_A A(t)), the recruitment weight per adult tonne in
units of an adult tonne's growth, by month.
"""
import sys

import numpy as np
import pandas as pd

path, juv = sys.argv[1:3]
y0, y1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2013, 2022)
S = pd.read_csv(path, sep=r"\s+")
S = S[(S.Year >= y0) & (S.Year <= y1)].copy()
for c in ("M1", "M2", "F", "N.bar", "west"):
    S[c] = pd.to_numeric(S[c], errors="coerce")
S["B"] = S["N.bar"] * S.west
S["Z"] = S.M1 + S.M2 + S.F


def is_j(a, q):
    # the size-defined stage (below the length at age 1 in Q3) holds age 0 and most of age 1 all year;
    # SMS has ages, so J = age 0 + 1 here
    return a <= 1


def stage_rates(sp):
    d = S[S["Species.n"] == sp]
    out = {}
    w = d.groupby(["Year", "Quarter", "Age"]).west.mean()
    b = d.groupby(["Year", "Quarter", "Age"]).B.sum()
    z = d.groupby(["Year", "Quarter", "Age"]).Z.mean()
    f = d.groupby(["Year", "Quarter", "Age"]).F.mean()
    acc = {k: [0.0, 0.0, 0.0, 0.0] for k in ("J", "A")}  # sum g*b, b_g, sum z*b, sum f*b
    for (y, q, a), wv in w.items():
        k = "J" if is_j(a, q) else "A"
        bb = b[(y, q, a)]
        acc[k][2] += 4 * z[(y, q, a)] * bb
        acc[k][3] += 4 * f[(y, q, a)] * bb
        nxt = (y, q + 1, a) if q < 4 else (y + 1, 1, a + 1)
        if nxt in w.index and wv > 0 and w[nxt] > 0:
            acc[k][0] += 4 * np.log(w[nxt] / wv) * bb
            acc[k][1] += bb
    bt = {k: b[[is_j(a, q) == (k == "J") for (_, q, a) in b.index]].sum() for k in ("J", "A")}
    for k in ("J", "A"):
        out[f"g_{k}"] = acc[k][0] / acc[k][1]
        out[f"Z_{k}"] = acc[k][2] / bt[k]
        out[f"F_{k}"] = acc[k][3] / bt[k]
    w03 = d[(d.Age == 0) & (d.Quarter == 3)].west.mean()
    w13 = d[(d.Age == 1) & (d.Quarter == 3)].west.mean()
    out["z"] = w03 / w13
    return out


SPEC = {"planktivores": {21: 0.89, 24: 0.08, 25: 0.03}, "benthivores": {18: 0.55, 16: 0.45}}
G = pd.read_csv(juv)
months = np.arange(12)
WIN = np.zeros(12)
WIN[[5, 6, 7, 8]] = [0.15, 0.35, 0.35, 0.15]
for grp, sp in SPEC.items():
    rr = {s: stage_rates(s) for s in sp}
    rate = {k: sum(sp[s] * rr[s][k] for s in sp) for k in ("g_J", "g_A", "Z_J", "Z_A", "F_J", "F_A", "z")}
    print(f"\n{grp}: field rates /yr (SMS {y0}-{y1}), " + ", ".join(f"{k} {v:.2f}" for k, v in rate.items()))
    for s in sp:
        print(f"   species {s}: " + ", ".join(f"{k} {v:.2f}" for k, v in rr[s].items()))
    g = G[G.group == ("plk" if grp == "planktivores" else "ben")]
    sv = {(r.period, r.quarter): r.J_share for r in g.itertuples()}
    target = np.mean([sv[("2001-25", 1)], sv[("2001-25", 3)]])
    nu, mu, zz = rate["g_J"], rate["Z_J"], rate["z"]
    gamma = (nu - mu) / (1 - zz ** (1 - mu / nu))
    print(f"  de Roos gamma: nu {nu:.2f}, mu {mu:.2f}, z {zz:.3f} -> gamma {gamma:.2f} /yr")
    dt = 1.0 / 12

    def cycle(R0, gam):
        J, A = 0.3, 0.7
        for _ in range(30):
            Js, As = [], []
            J0, A0 = J, A
            for m in months:
                for _ in range(10):
                    h = dt / 10
                    R = R0 * WIN[m] * 12
                    dJ = R + (rate["g_J"] - rate["Z_J"] - gam) * J
                    dA = gam * J + (rate["g_A"] - rate["Z_A"]) * A
                    J, A = J + h * dJ, A + h * dA
                Js.append(J)
                As.append(A)
            # rescale so B repeats (the stock's compensation, not resolved here)
            Bend = J + A
            J, A = J / Bend, A / Bend
        Js, As = np.array(Js), np.array(As)
        return Js, As

    lo, hi = 1e-4, 50.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        Js, As = cycle(mid, gamma)
        x = Js / (Js + As)
        if x.mean() < target:
            lo = mid
        else:
            hi = mid
    R0 = mid
    Js, As = cycle(R0, gamma)
    x = Js / (Js + As)
    q = {1: x[1], 2: x[4], 3: x[7], 4: x[10]}  # Feb, May, Aug, Nov (end of month)
    print(f"  target annual-mean J/B {target:.3f} (2001-25 Q1 {sv[('2001-25', 1)]:.2f}, Q3 {sv[('2001-25', 3)]:.2f}); "
          f"gamma {gamma:.2f}; R0 {R0:.3f} of B per year")
    print("  cycle J/B Feb / May / Aug / Nov: " + " / ".join(f"{q[k]:.2f}" for k in (1, 2, 3, 4)))
    # month midpoints (the start_share table): mean of the month's start and end
    prev = np.roll(x, 1)
    print("  start_share by month (Jan-Dec): [" + ", ".join(f"{(a + b) / 2:.3f}" for a, b in zip(prev, x)) + "]")
    print("  survey 2001-25 Q1 / Q3: {:.2f} / {:.2f}; 1991-96 Q1-Q4: ".format(sv[("2001-25", 1)], sv[("2001-25", 3)])
          + " / ".join(f"{sv[('1991-96', k)]:.2f}" for k in (1, 2, 3, 4)))
    kappa = rate["g_J"] / rate["g_A"]
    rho = R0 * WIN * 12 / (rate["g_A"] * As / (Js + As))
    newJ = (R0 * WIN * 12 + rate["g_J"] * Js) / (R0 * WIN * 12 + rate["g_J"] * Js + rate["g_A"] * As)
    print(f"  kappa = g_J / g_A = {kappa:.1f}")
    print("  rho(t) by month (recruitment weight per adult tonne, adult-growth units): "
          + " ".join(f"{v:.1f}" for v in rho))
    print("  share of the group's new tonnes going to J by month: " + " ".join(f"{v:.2f}" for v in newJ))
    for gam in (0.8 * gamma, 1.25 * gamma):
        lo, hi = 1e-4, 50.0
        for _ in range(60):
            mid = (lo * hi) ** 0.5
            Jq, Aq = cycle(mid, gam)
            if (Jq / (Jq + Aq)).mean() < target:
                lo = mid
            else:
                hi = mid
        Jq, Aq = cycle(mid, gam)
        xx = Jq / (Jq + Aq)
        print(f"  sensitivity gamma {gam:.2f}: R0 {mid:.3f}, J/B Feb / May / Aug / Nov "
              + " / ".join(f"{xx[i]:.2f}" for i in (1, 4, 7, 10)))
