"""Whiting's uncapped potential under each speed reading: X = sum_j a_j h B_vis,j scaled per pair by the
speed factors (speed_factors.py), on the 289 yardstick (T 0.905, X 5.45, krill 4.47 / other 0.98) with the
non-krill X split by the budget_gate equal-cell encounter shares (planktivores 0.145, epifauna 0.083,
benthivores 0.028, zooplankton 0.009; copepod pair taken at the krill factor, J). Need 0.388.
usage: speed_potential.py <speed_factors.json>"""
import json, sys
F = json.load(open(sys.argv[1]))["factors"]
T, XK, XO = 0.905, 4.47, 0.98
oth = {"whiting->planktivorous_fish": 0.145, "whiting->mobile_epifauna": 0.083, "whiting->benthivorous_fish": 0.028,
       "zoo": 0.009}
tot = sum(oth.values())
s = lambda x: x / (1 + x)
print(f"base: X {XK + XO:.2f}, uncapped {T * s(XK + XO):.3f}, need 0.388")
for n, f in F.items():
    xo = XO * sum(w / tot * f.get(p, f["whiting->krill"]) for p, w in oth.items())
    xk = XK * f["whiting->krill"]
    print(f"{n:11s}: X krill {xk:.2f} other {xo:.2f} -> uncapped {T * s(xk + xo):.3f} "
          f"(change {T * s(xk + xo) - T * s(XK + XO):+.3f}); krill share of X {xk / (xk + xo):.3f} (base {XK / (XK + XO):.3f})")
