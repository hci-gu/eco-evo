"""State-dependent optimal reserve of a FIELD fish (Houston & McNamara style dynamic programme):
which fill would a fish choose if carrying reserve had a cost, and does any cost give the field's
lean spring and fat autumn? Research only: no model run, no model output sets a value.

Objective: maximise long-run expected reproductive value, taken proportional to mass: each day
the value is multiplied by survival (1 - hazard) and by the mass gain (1 + G / ec) (periodic
year, value renormalised every year). This is the growth / mortality currency of state-dependent
life history (Gilliam's mu/g; McNamara & Houston 1990; Houston et al. 1993); mass=False gives
pure mortality minimisation.
- State: fill f in [0, 1] (reserve / max), food patch state z in {good, bad}, day of year.
- Decision: foraging effort e in [0, 1] (share of the potential intake taken).
- Energy per day (MJ/t): intake e X_z(d); cost RM(T) (1 + e (feed - 1)); growth paid first from
  food (no growth below maintenance, as the material law): G = min(Gmax, max(eX - cost, 0));
  the rest goes to the reserve up to f = 1 (reserve room). Extra metabolism k RM f for a
  carrying cost. f < 0 = starvation death.
- Hazard per day: [M1 + M2 (1 + (r - 1) e)(1 + beta f)] / 365; r is the activity risk ratio
  ("Vilans avkastning": 1.0-1.13 for fish predators; 1.1 used), beta the fill-dependent risk.
- Food: potential X_good = kpot x need / (1 - phi) in good spells, 0 in bad spells (mean
  bad-spell length L days, bad share phi), so the mean potential is kpot x need.
  'flat' = the brief's model yardstick (kpot 2, no season). 'seasonal' = the whiting SMS
  quarterly ration shape (ICES WGSAM 2024 Table 3-8, 250 g, via "Mättnad hos rovfisk"
  model_side.py) as the potential, scaled to an annual mean kpot x mean need, cost at the
  bottom-layer temperature (Q10 2.3).
Whiting library values: ec 3900, ME 2000, RM 3.87 MJ/t/tick, feeding_cost 1.5, P 0.61 /yr,
M1 0.25 /yr, booked M2 0.49 /yr.

Run: python3 -I scripts/optimal_fill.py > tables/optimal_fill_stdout.txt
"""
import numpy as np

DAYS = 365
NF = 201
F = np.linspace(0.0, 1.0, NF)
E = np.linspace(0.0, 1.0, 41)
BOTTOM = [7.8, 6.99, 6.59, 6.58, 6.69, 6.96, 7.22, 7.41, 7.7, 8.04, 8.36, 8.5]
SMS_250 = [0.725, 0.773, 1.155, 0.977]          # %BW/d, Q1-Q4 (whiting 250 g)
FIELD_WHITING = [0.69, 0.69, 0.56, 0.43, 0.3, 0.3, 0.3, 0.52, 0.73, 0.73, 0.73, 0.71]

W = dict(ec=3900.0, me=2000.0, rm=3.87 * 4, feed=1.5, P=0.61, m1=0.25, m2=0.492)


def month_of(d):
    return min(int(d / (DAYS / 12)), 11)


def daily_temperature():
    return np.array([BOTTOM[month_of(d)] for d in range(DAYS)])


GROWTH_SHAPE = [0.3, 1.5, 1.5, 0.7]   # [J] Q1-Q4 somatic growth share x 4 (field whiting grow
                                      # mainly Q2-Q3: "Mättnad hos rovfisk" model_side caveat)


def setup(scen, kpot, seasonal_T, growth_season=False):
    tmean = np.mean(BOTTOM)
    if seasonal_T:
        q10 = 2.3 ** ((daily_temperature() - tmean) / 10)
    else:
        q10 = np.ones(DAYS)
    rm = W["rm"] * q10
    gmax = W["P"] * W["ec"] / 365.0 * np.ones(DAYS)
    if growth_season:
        gmax = gmax * np.array([GROWTH_SHAPE[min(d // 92, 3)] for d in range(DAYS)])
    need = rm * W["feed"] + gmax
    if scen == "flat":
        xmean = kpot * need.mean() * np.ones(DAYS)
    else:
        shape = np.array([SMS_250[min(d // 92, 3)] for d in range(DAYS)])
        shape = shape / shape.mean()
        xmean = kpot * need.mean() * shape
    return rm, gmax, xmean


def solve(scen="flat", kpot=2.0, phi=0.0, L=1.0, beta=0.0, k=0.0, r=1.1, seasonal_T=False,
          years=12, mass=True, growth_season=False):
    rm, gmax, xmean = setup(scen, kpot, seasonal_T, growth_season)
    me = W["me"]
    if phi > 0:
        p_bg = 1.0 / L                  # bad -> good per day
        p_gb = phi / (1 - phi) * p_bg   # good -> bad per day
        trans = np.array([[1 - p_gb, p_gb], [p_bg, 1 - p_bg]])
        xg = xmean / (1 - phi)
        xz = np.stack([xg, np.zeros(DAYS)])
    else:
        trans = np.array([[1.0, 0.0], [0.0, 1.0]])
        xz = np.stack([xmean, xmean])
    V = np.ones((2, NF))
    V[:, 0] = 0.0
    policy = np.zeros((DAYS, 2, NF), dtype=int)
    for _ in range(years):
        for d in range(DAYS - 1, -1, -1):
            EV = trans @ V                       # expected V next day given today's z
            newV = np.zeros((2, NF))
            for z in range(2):
                intake = E[:, None] * xz[z, d]                         # (E,1)
                cost = rm[d] * (1 + E[:, None] * (W["feed"] - 1)) + k * rm[d] * F[None, :]
                surplus = intake - cost
                growth = np.clip(surplus, 0, gmax[d])
                df = (surplus - growth) / me
                f2 = np.minimum(F[None, :] + df, 1.0)
                haz = (W["m1"] + W["m2"] * (1 + (r - 1) * E[:, None]) * (1 + beta * F[None, :])) / 365
                vnext = np.where(f2 < 0, 0.0, np.interp(np.clip(f2, 0, 1), F, EV[z]))
                val = (1 - haz) * vnext
                if mass:   # reproductive value proportional to mass: x (1 + G / ec) per day
                    val = val * (1 + growth / W["ec"])
                best = val.argmax(axis=0)
                policy[d, z] = best
                newV[z] = val[best, np.arange(NF)]
            V = newV
        V = V / V.max()
    return policy, xz, trans, rm, gmax


def simulate(policy, xz, trans, rm, gmax, beta, k, r, years=6):
    """Forward distribution of (z, f) under the optimal policy (survivors only, renormalised)."""
    me = W["me"]
    dist = np.zeros((2, NF))
    dist[0, int(0.5 * (NF - 1))] = 1.0
    monthly = np.zeros(12)
    counts = np.zeros(12)
    effort = np.zeros(12)
    surv = 1.0
    starve = 0.0
    for y in range(years):
        for d in range(DAYS):
            new = np.zeros((2, NF))
            for z in range(2):
                e = E[policy[d, z]]
                intake = e * xz[z, d]
                cost = rm[d] * (1 + e * (W["feed"] - 1)) + k * rm[d] * F
                surplus = intake - cost
                growth = np.clip(surplus, 0, gmax[d])
                f2 = np.minimum(F + (surplus - growth) / me, 1.0)
                alive = f2 >= 0
                pos = np.clip(f2, 0, 1) * (NF - 1)
                lo = np.floor(pos).astype(int)
                hi = np.minimum(lo + 1, NF - 1)
                w = pos - lo
                mass = dist[z] * alive
                if y == years - 1:
                    starve += (dist[z] * ~alive).sum()
                    effort[month_of(d)] += (dist[z] * e).sum()
                for z2 in range(2):
                    tz = trans[z, z2]
                    if tz == 0:
                        continue
                    np.add.at(new[z2], lo, mass * (1 - w) * tz)
                    np.add.at(new[z2], hi, mass * w * tz)
            tot = new.sum()
            if y == years - 1:
                surv *= tot
            dist = new / tot
            if y == years - 1:
                m = month_of(d)
                monthly[m] += (dist.sum(axis=0) * F).sum()
                counts[m] += 1
    return monthly / counts, effort / counts, starve


def run(label, **kw):
    sim_kw = dict(beta=kw.get("beta", 0.0), k=kw.get("k", 0.0), r=kw.get("r", 1.1))
    pol, xz, trans, rm, gmax = solve(**kw)
    m, eff, starve = simulate(pol, xz, trans, rm, gmax, **sim_kw)
    print(f"  {label:44s} mean {m.mean():.2f}  " + " ".join(f"{x:.2f}" for x in m)
          + f" | effort {eff.mean():.2f}, starved/yr {starve:.3f}")
    return m


if __name__ == "__main__":
    print("Whiting, optimal fill by month (J F M A M J J A S O N D), survivors' mean;")
    print("effort = mean foraging share; starved/yr = annual starvation probability under the policy")
    print("  field (start_fill)                           mean %.2f  " % np.mean(FIELD_WHITING)
          + " ".join(f"{x:.2f}" for x in FIELD_WHITING))
    print("\nA. Degenerate case: flat potential 2x need, food without gaps. The reserve has no insurance")
    print("   value, so its level is arbitrary at beta 0 (stays at the start, 0.5) and any beta > 0 empties it")
    for beta in (0.0, 0.5):
        run(f"beta {beta}", scen="flat", kpot=2.0, beta=beta)
    print("\nB. Flat potential 2x need (the brief's model yardstick), patchy food (bad spells with no food)")
    for phi, L in ((0.3, 5.0), (0.3, 20.0), (0.5, 40.0)):
        for beta in (0.0, 0.3, 1.0, 3.0):
            run(f"bad share {phi}, spell {L:.0f} d, beta {beta}", scen="flat", kpot=2.0, phi=phi, L=L,
                beta=beta)
    for k in (0.1, 0.3, 1.0):
        run(f"bad 0.3, 20 d, metabolic k {k} x RM per unit fill", scen="flat", kpot=2.0, phi=0.3, L=20.0,
            k=k)
    for r in (1.0, 1.5):
        run(f"bad 0.3, 20 d, beta 1, activity risk r {r}", scen="flat", kpot=2.0, phi=0.3, L=20.0,
            beta=1.0, r=r)
    print("\nC. Seasonal potential (SMS whiting ration shape, annual mean kpot x need) + patchy food")
    print("   (bad share 0.3, spells 10 d); cost at the bottom-layer temperature; growth even over the year")
    for kpot in (1.0, 1.1, 1.3, 2.0):
        for beta in (0.0, 0.3, 1.0, 3.0):
            run(f"kpot {kpot}, beta {beta}", scen="seasonal", kpot=kpot, beta=beta, seasonal_T=True,
                phi=0.3, L=10.0)
    print("\nD. As C, with field growth timing [J]: somatic growth x0.3 / 1.5 / 1.5 / 0.7 by quarter, paid")
    print("   before the reserve (growth priority, as in the material law's food-first rule)")
    for kpot in (1.0, 1.1, 1.3, 2.0):
        for beta in (0.0, 0.3, 1.0):
            run(f"kpot {kpot}, beta {beta}", scen="seasonal", kpot=kpot, beta=beta, seasonal_T=True,
                phi=0.3, L=10.0, growth_season=True)
