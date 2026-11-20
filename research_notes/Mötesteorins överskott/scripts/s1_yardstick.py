"""Yardstick: what has to move, per fish group, from the frozen fix105 decomposition (model output used ONLY to
size the gap, never to choose a value).

Usage: python3 -I s1_yardstick.py <fill_decomp_all_s0.json> <fill_decomp_all_s1.json>

Shared handling: while foraging the share of the ceiling is S = X_eff / (1 + X_eff), X_eff = sum_j x_j
(x_j = a_j B_vis,j h_j; interference w B folded in, rough). unc = T S (T = foraging share).
Group-mean X_eff = S / (1 - S): rough, because the mean over cells and ticks of a saturating function is not the
function of the mean.

Targets (feeding level on the model ceiling):
  field   = the field ration (Q/B x E / ceiling), from "Mattnad hos rovfisk"/model/tables/pair_vs_field_fix105.txt
  need    = the realised (capped) level at fill ~1 = cost + field P (the decomposition's 'feed')
Needed corrections:
  k_a     generic factor on every a of the group:   T k X / (1 + k X) = target  ->  k = target / ((T - target) X)
  tau     share of the foraging time (whole response, a and ceiling):  tau unc = target
  k_a(j)  factor on prey j's a alone: T (X - x_j + k x_j) / (1 + X - x_j + k x_j) = target; 'none' if prey j
          removed entirely still leaves the potential above the target (wo_j > target)
"""
import json
import sys

FIELD = {  # pair_vs_field_fix105.txt, 'field total' (feeding level on the ceiling)
    "planktivorous_fish": 0.231, "benthivorous_fish": 0.521, "whiting": 0.455,
    "predatory_fish": 0.375, "mackerel": 0.137,
}
d0, d1 = (json.load(open(p)) for p in sys.argv[1:3])


def mean2(g, k):
    return 0.5 * (d0[g][k] + d1[g][k])


def solve_k_all(T, X, f):
    return f / ((T - f) * X) if f < T else float("nan")


def solve_k_one(T, X, xj, f):
    # T Y/(1+Y) = f -> Y = f/(T-f); Y = X - xj + k xj
    Y = f / (T - f)
    k = (Y - (X - xj)) / xj
    return k if k >= 0 else None


print("group                T     unc    S     X_eff   feed(need) field | k_a need/field | tau need/field")
rows = {}
for g in FIELD:
    T, unc, feed = mean2(g, "T"), mean2(g, "unc"), mean2(g, "feed")
    S = unc / T
    X = S / (1 - S)
    f_field = FIELD[g]
    ka = [solve_k_all(T, X, f) for f in (feed, f_field)]
    tau = [f / unc for f in (feed, f_field)]
    rows[g] = dict(T=T, unc=unc, X=X, feed=feed, field=f_field, ka=ka, tau=tau)
    print(f"{g:20s} {T:5.3f} {unc:6.3f} {S:5.3f} {X:7.2f}  {feed:6.3f}    {f_field:5.3f} | "
          f"{ka[0]:.3f} / {ka[1]:.3f}  | {tau[0]:.2f} / {tau[1]:.2f}")

print("\nPer prey (seed-mean share of the uncapped handling time -> x_j = share X; wo = potential without prey j)")
print("k_a(j) = factor on that prey's a alone to reach need / field ('none' = even removing it leaves the potential"
      " above the target)")
for g, r in rows.items():
    prey = d0[g]["prey"]
    print(f"\n{g}  (X {r['X']:.2f}, T {r['T']:.3f}, need {r['feed']:.3f}, field {r['field']:.3f})")
    for i, p in enumerate(prey):
        sh = 0.5 * (d0[g]["share"][i] + d1[g]["share"][i])
        wo = 0.5 * (d0[g]["wo"][i] + d1[g]["wo"][i])
        xj = sh * r["X"]
        ks = []
        for f in (r["feed"], r["field"]):
            if wo > f:          # exact test from the decomposition: removing prey j is not enough
                ks.append("none")
                continue
            k = solve_k_one(r["T"], r["X"], xj, f) if xj > 0 else None
            ks.append("~0" if k is None else f"{k:.3f}")
        print(f"   {p:20s} share {sh:5.3f}  x_j {xj:6.2f}  wo {wo:5.3f}  k_a(j) need {ks[0]:>6s}  field {ks[1]:>6s}")

print("\nRatio of the needed generic a-factor between groups (max/min, need):",
      f"{max(r['ka'][0] for r in rows.values()) / min(r['ka'][0] for r in rows.values()):.1f}x;",
      "tau (max/min):",
      f"{max(r['tau'][0] for r in rows.values()) / min(r['tau'][0] for r in rows.values()):.2f}x")
print("Ratio (field target): a-factor",
      f"{max(r['ka'][1] for r in rows.values()) / min(r['ka'][1] for r in rows.values()):.1f}x; tau",
      f"{max(r['tau'][1] for r in rows.values()) / min(r['tau'][1] for r in rows.values()):.2f}x")
json.dump(rows, open(sys.argv[0].replace("scripts/s1_yardstick.py", "tables/s1_yardstick.json"), "w"), indent=1)
