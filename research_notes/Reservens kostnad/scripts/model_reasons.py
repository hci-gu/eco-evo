"""What a full reserve is worth to the model's learner, and what a cost would have to be to
outweigh it (research only; no model run, no model output sets any value).

Library constants are copied from fgconfig/fg_library.yaml (2026-11-20). M2 are the BOOKED
values tabulated in research_notes/"Vilans avkastning" (rest_payoff_stdout.txt). The "potential
~2x need" is the brief's yardstick (frozen fix105 decomposition), used only to size a threshold.

Engine facts used (read from the code, lib/gpu/rollout.py and movement.income_growth):
- the reward is the mean over the rollout ticks of log((B ec + R) / E0) (integral_reward True,
  ticks 1460 = one year in the fix105 run config): a stored MJ counts like a lean-tissue MJ;
- the material-law growth ceiling is B g (s - u) p with g derived at s* = 1, so the somatic
  production P is reached only at fill 1 and scales as (s - u)/(1 - u) below it;
- losses (M1, M2, emigration) take the reserve pro rata, so they remove the same fraction of
  E whatever the fill: no fill cost in log(E);
- benthivores only: a new tonne costs ec + c (ec + s ME) (growth_cost c 0.69).

Run: python3 -I scripts/model_reasons.py > tables/model_reasons_stdout.txt
"""
import math

T = 1460
# ec, ME, RM (MJ/t/tick, library reference T), feeding_cost, u, P (/yr), M1 (/yr),
# booked M2 fish + mammal (/yr), field annual-mean fill (start_fill / outside_fill mean),
# field spring low, growth_cost c
G = {
    "planktivores": dict(ec=4000, me=4990, rm=8.0, feed=1.65, u=0.079, P=0.66,
                         m1=0.000186986 * T, m2=0.349 + 0.071, c=0.0,
                         fill=[0.5, 0.36, 0.26, 0.2, 0.32, 0.81, 0.94, 0.96, 0.88, 0.88, 0.76, 0.68]),
    "benthivores": dict(ec=3600, me=1800, rm=3.33, feed=1.02, u=0.078, P=0.74,
                        m1=0.00019861 * T, m2=0.165 + 0.085, c=0.69,
                        fill=[0.76, 0.76, 0.62, 0.47, 0.33, 0.33, 0.33, 0.57, 0.81, 0.81, 0.81, 0.79]),
    "whiting": dict(ec=3900, me=2000, rm=3.87, feed=1.50, u=0.07, P=0.61,
                    m1=0.000171233 * T, m2=0.337 + 0.155, c=0.0,
                    fill=[0.69, 0.69, 0.56, 0.43, 0.3, 0.3, 0.3, 0.52, 0.73, 0.73, 0.73, 0.71]),
    "water_column": dict(ec=4180, me=2000, rm=3.29, feed=2.14, u=0.07, P=0.42,
                         m1=0.000189238 * T, m2=0.018, c=0.0, fill=[0.71] * 12),
    # mackerel: P from the derived g (0.000285/tick) x (1 - u) x 1460; outside_fill
    "mackerel": dict(ec=4670, me=6980, rm=4.59, feed=1.02, u=0.057,
                     P=0.0002849840446 * (1 - 0.057) * T,
                     m1=0.000108783 * T, m2=0.021, c=0.0,
                     fill=[0.70, 0.58, 0.45, 0.37, 0.13, 0.51, 0.57, 0.91, 1.0, 1.0, 0.94, 0.85]),
}
H = 0.5   # effective horizon of a one-year integral reward for a rate term (mean of t over [0, 1])


def price(g, s):
    return g["ec"] + g["c"] * (g["ec"] + s * g["me"])


def J(g, s, beta=0.0, mu_fill=0.0):
    """Reward of holding fill s all year (log units, relative), with a hypothetical predation
    hazard M2 (1 + beta s) and/or an extra loss mu_fill * s (/yr) on B and R."""
    level = math.log(g["ec"] + s * g["me"])
    prod = g["P"] * max(s - g["u"], 0.0) / (1 - g["u"])
    # tonnes: the ceiling B g (s - u) p in MJ buys G / p tonnes, so the price cancels
    return level + H * (prod - beta * g["m2"] * s - mu_fill * s)


def argmax(g, **kw):
    grid = [i / 1000 for i in range(0, 1001)]
    return max(grid, key=lambda s: J(g, s, **kw))


print("1. What a full reserve is worth to the learner (log-energy reward, one-year integral)")
print("   group          ME/(ec+ME)  dlogE/ds at s=1  growth term 0.5 P/(1-u)  reward(s=1) - reward(field mean)")
for n, g in G.items():
    fm = sum(g["fill"]) / 12
    lvl = g["me"] / (g["ec"] + g["me"])
    gr = H * g["P"] / (1 - g["u"])
    dJ = J(g, 1.0) - J(g, fm)
    print(f"   {n:13s}  {lvl:9.3f}  {lvl:15.3f}  {gr:22.3f}  {dJ:+.3f} (field mean fill {fm:.2f};"
          f" = {12 * dJ / g['P']:.1f} months of P)")
print("   Reading: holding fill 1 instead of the field mean is worth +0.16 to +0.36 log units of reward,"
      "\n   i.e. the same as 4.5-9 months of the group's somatic production. Both terms are engine"
      "\n   conventions (reward currency; growth ceiling g (s - u) with g derived at s* = 1), not costs"
      "\n   or benefits a wild fish has in this form [D].")

print("\n2. Hazard-type cost: predation M2 (1 + beta s). beta a full fish would need for the learner's")
print("   optimum fill to fall to the field annual mean, and to the field spring low")
print("   group          M2 /yr  beta* (optimum < 1)  beta (field mean)  beta (spring low)  hazard full/empty at field mean")
for n, g in G.items():
    fm = sum(g["fill"]) / 12
    lo = min(g["fill"])

    def beta_for(target):
        b_lo, b_hi = 0.0, 1e4
        for _ in range(80):
            b = 0.5 * (b_lo + b_hi)
            if argmax(g, beta=b) > target:
                b_lo = b
            else:
                b_hi = b
        return b_hi
    b1 = beta_for(0.999)
    bm = beta_for(fm)
    bl = beta_for(lo)
    print(f"   {n:13s}  {g['m2']:6.3f}  {b1:18.1f}  {bm:16.1f}  {bl:16.1f}  {1 + bm:8.1f}x")
print("   Reading: for whiting, the group with the largest booked M2, a full fish would need a predation"
      "\n   hazard ~3.7-4x that of an empty one before the learner keeps it below fill 1 (planktivores"
      "\n   5.4-7x, benthivores 6.9-7.3x); the water column and mackerel (M2 ~0.02) would need 60-140x. Any such beta is season-constant, so it"
      "\n   gives a flat lower fill, not a spring low and an autumn high [D].")

print("\n3. The same as a fill-proportional extra loss mu * s on B and R (/yr), independent of M2")
for n, g in G.items():
    fm = sum(g["fill"]) / 12
    b_lo, b_hi = 0.0, 100.0
    for _ in range(80):
        b = 0.5 * (b_lo + b_hi)
        if argmax(g, mu_fill=b) > fm:
            b_lo = b
        else:
            b_hi = b
    print(f"   {n:13s}  mu* for optimum = field mean ({fm:.2f}): {b_hi:.2f} /yr extra mortality at fill 1")

print("\n4. Energy-type cost: extra metabolism k x RM x s (MJ/t/tick). With potential ~2x need it is paid out of")
print("   the spare intake; it binds only when k RM exceeds the spare capacity at fill 1 (yardstick)")
print("   group          need at s=1 (MJ/t/tick)  spare = need (2x yardstick)  k_crit = spare/RM")
for n, g in G.items():
    need = g["rm"] * g["feed"] + g["P"] / T * price(g, 1.0)
    print(f"   {n:13s}  {need:22.2f}  {need:26.2f}  {need / g['rm']:6.2f} (a full reserve would have to raise"
          f" metabolism by {need / g['rm']:.1f}x RM)")
print("   Below k_crit an energy cost lowers no fill; it only lengthens foraging, whose risk cost is"
      "\n   0.01-0.05 of the production at stake ('Vilans avkastning' fill_tradeoff) [S/D].")

print("\n5. What the model already charges for a full reserve")
g = G["benthivores"]
print(f"   benthivores tonne price at s=1 {price(g, 1.0):.0f} vs s=0.33 {price(g, 0.33):.0f} MJ/t "
      f"(x{price(g, 1.0) / price(g, 0.33):.3f}); other groups: no c, price ec at any fill")
print("   losses take the reserve pro rata: the fraction of E lost per tick is the same at any fill (no cost in log E)")
print("   the reserve room cap discards surplus at fill 1: no cost, only a ceiling")
