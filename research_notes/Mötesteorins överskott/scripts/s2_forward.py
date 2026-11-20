"""Forward combination of the evidenced corrections per term, against the needed factor (Q3).

Usage: python3 -I s2_forward.py <tables/s1_yardstick.json> <fill_decomp_all_s0.json> <fill_decomp_all_s1.json>
                                <tables/term_corrections.csv>

term_corrections.csv: one row per (group, term): group, term, form, prey, weak, central, strong, in_chain, tag, basis
  form  a      factor on the a of the listed prey ('all' = every prey of the group)
        time   share of the foraging time on the whole response (a and the ceiling together, the engine's T)
  weak / central / strong = the correction factor (in situ / chain), weak = closest to no change.
Nothing here is fitted: every factor comes from the CSV (literature, [D] or [J]); the yardstick (frozen fix105:
T, X_eff, per-prey x_j shares) only maps factors onto the feeding level.

Potential feeding level while uncapped:  P = tau T Y / (1 + Y),  Y = sum_j k_j x_j,
k_j = product of the a-factors that reach prey j, tau = product of the time factors.
Printed per group: each term alone, the a-side product, the time-side product, all together; and the share of
the needed log-reduction of the potential (log(P/unc) / log(target/unc)) each one carries.
"""
import csv
import json
import math
import sys

Y = json.load(open(sys.argv[1]))
d0, d1 = (json.load(open(p)) for p in sys.argv[2:4])
ROWS = list(csv.DictReader(open(sys.argv[4])))
LEVELS = ("weak", "central", "strong")


def prey_x(g):
    X = Y[g]["X"]
    return {p: 0.5 * (d0[g]["share"][i] + d1[g]["share"][i]) * X for i, p in enumerate(d0[g]["prey"])}


def potential(g, rows, lev):
    xs = prey_x(g)
    k = {p: 1.0 for p in xs}
    tau = 1.0
    for r in rows:
        f = float(r[lev])
        if r["form"] == "time":
            tau *= f
        else:
            targets = xs if r["prey"] == "all" else r["prey"].split(";")
            for p in targets:
                if p in k:
                    k[p] *= f
    Ysum = sum(k[p] * xs[p] for p in xs)
    return tau * Y[g]["T"] * Ysum / (1.0 + Ysum)


def carried(g, P):
    unc, need = Y[g]["unc"], Y[g]["feed"]
    return round(math.log(P / unc) / math.log(need / unc), 6) + 0.0


def residual(g, rows):
    # generic a-factor and time share still needed after the central corrections
    xs = prey_x(g)
    k = {p: 1.0 for p in xs}
    tau = 1.0
    for r in rows:
        f = float(r["central"])
        if r["form"] == "time":
            tau *= f
            continue
        for p in (xs if r["prey"] == "all" else r["prey"].split(";")):
            if p in k:
                k[p] *= f
    T, need = Y[g]["T"] * tau, Y[g]["feed"]
    Yc = sum(k[p] * xs[p] for p in xs)
    Yn = need / (T - need)
    Pc = T * Yc / (1 + Yc)
    return Yn / Yc, need / Pc, need / (T * Yc)


for g in Y:
    rows = [r for r in ROWS if r["group"] == g]
    unc, need, field = Y[g]["unc"], Y[g]["feed"], Y[g]["field"]
    print(f"\n=== {g}: uncapped {unc:.3f}, need {need:.3f}, field ration {field:.3f}; X_eff {Y[g]['X']:.2f}; "
          f"needed generic a-factor {Y[g]['ka'][0]:.3f} (field {Y[g]['ka'][1]:.3f}), time share "
          f"{Y[g]['tau'][0]:.2f} ({Y[g]['tau'][1]:.2f})")
    print(f"  {'term':34s} {'form':5s} {'prey':28s} {'w/c/s factor':18s} {'potential w/c/s':20s} share of needed (c)")
    for r in rows:
        Ps = [potential(g, [r], lev) for lev in LEVELS]
        fac = "/".join(f"{float(r[lev]):.2f}" for lev in LEVELS)
        print(f"  {r['term']:34s} {r['form']:5s} {r['prey'][:28]:28s} {fac:18s} "
              + "/".join(f"{P:.3f}" for P in Ps) + f"   {carried(g, Ps[1]) + 0.0:5.2f}   [{r['tag']}]")
    for label, sub in (("a-side product", [r for r in rows if r["form"] == "a"]),
                       ("time-side product", [r for r in rows if r["form"] == "time"]),
                       ("a-side, not in chain", [r for r in rows if r["form"] == "a" and r["in_chain"] == "no"]),
                       ("all terms", rows)):
        if not sub:
            continue
        Ps = [potential(g, sub, lev) for lev in LEVELS]
        print(f"  {label:34s} {'':5s} {'':28s} {'':18s} " + "/".join(f"{P:.3f}" for P in Ps)
              + f"   {carried(g, Ps[1]):5.2f}   (need {need:.3f}, field {field:.3f})")
    ra, rt, rg = residual(g, rows)
    print(f"  residual after all central corrections: generic a-factor {ra:.3f}, or time share {rt:.2f}; "
          f"upper-bound form min(1, T X) (s3; a bound, not a need) a-factor {rg:.3f}")
    print(f"  needed with no correction, upper-bound form min(1, T X): a-factor {need / (Y[g]['T'] * Y[g]['X']):.3f}")
