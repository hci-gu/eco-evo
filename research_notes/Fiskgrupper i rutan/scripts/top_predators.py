"""Top-predator pairs on the three fish groups: consumption, M2, intake and break-even.

Research arithmetic for params_top_predators.md, not model code. No inputs; prints tables.
Model conventions: tick 6 h (1460/yr), t/km2 = t/cell, Type II intake per t predator per
tick I = a*B/(1 + a*h*B), Type III I = a*B^2/(1 + a*h*B^2), B = VISIBLE prey density.
"""
import math

GRID = 3600.0
PREY_T = {"plank": 12000.0, "benth": 5500.0, "pred": 4300.0}
PREY_D = {k: v / GRID for k, v in PREY_T.items()}
OLD_GAD_D = 2500.0 / GRID

def vis(b, floor, rest=0.4):
    """Visible biomass with a share `rest` hidden at visibility `floor` (judgement: rest 0.4)."""
    return b * (1.0 - rest * (1.0 - floor))

def intake(a, h, b, typ=2):
    x = a * b if typ == 2 else a * b * b
    return x / (1.0 + h * x)

def breakeven(cost, ass, e, a, h, typ=2):
    i = cost / (ass * e)
    if i * h >= 1:
        return float("inf")
    x = i / (1.0 - i * h)  # a*B (II) or a*B^2 (III)
    return x / a if typ == 2 else math.sqrt(x / a)

def a_same_return(i_old, h, b_new):
    """Attack rate giving intake i_old at density b_new (Type II)."""
    return i_old / (b_new * (1.0 - i_old * h))

print("== Consumption and M2")
pred = {
    "seals": dict(B=15.0, Q=320.0, Qlo=250.0, Qhi=460.0,
                  diet={"benth": (0.60, 0.45, 0.70), "pred": (0.15, 0.08, 0.25), "plank": (0.25, 0.10, 0.40)}),
    "porpoises": dict(B=12.0, Q=12.0 * 19.5, Qlo=12.0 * 17.6, Qhi=12.0 * 23.3,
                      diet={"plank": (0.65, 0.55, 0.75), "benth": (0.15, 0.10, 0.25), "pred": (0.20, 0.12, 0.25)}),
}
for p, d in pred.items():
    print(f"{p}: B {d['B']} t, Q {d['Q']:.0f} t/yr (Q/B {d['Q']/d['B']:.1f}), range {d['Qlo']:.0f}-{d['Qhi']:.0f}")
    for prey, (s, lo, hi) in d["diet"].items():
        c = s * d["Q"]
        m2 = c / PREY_T[prey]
        m2lo = lo * d["Qlo"] / PREY_T[prey]
        m2hi = hi * d["Qhi"] / PREY_T[prey]
        print(f"   {prey:6s} share {s:.2f} ({lo:.2f}-{hi:.2f})  C {c:6.1f} t/yr  M2 {m2:.4f} /yr ({m2lo:.4f}-{m2hi:.4f})")

print("\n== Library as copied (old gadoid pair on both demersal pools)")
for p, m2s in {"seals": {"benth": 0.12, "pred": 0.12, "plank": 0.0015},
               "porpoises": {"benth": 0.0557, "pred": 0.0557, "plank": 0.0117}}.items():
    tot = 0.0
    for prey, m2 in m2s.items():
        c = m2 * PREY_T[prey]; tot += c
        print(f"   {p:9s} {prey:6s} M2 {m2:.4f} -> {c:6.0f} t/yr")
    print(f"   {p:9s} total {tot:.0f} t/yr")

print("\n== Seal ration from energy (283 MJ/t/day spend + production, report)")
for eb, ep, el in [(5500, 5500, 7500), (4500, 5500, 7500), (4500, 6000, 6500)]:
    me = 0.85 * (0.60 * eb + 0.15 * ep) + 0.88 * 0.25 * el
    r = 283.0 / me
    print(f"   E benth {eb} pred {ep} plank {el}: ME {me:.0f} MJ/t prey, ration {100*r:.2f} %BM/d, {15*r*365:.0f} t/yr")

print("\n== Handling times")
SEAL_CMAX = 0.85 * 1000 / 4  # MJ/t/tick gross (0.85 MJ/kg/d)
for e in (4000, 4500, 5000, 5500, 6000, 6500, 7500):
    print(f"   seals energy ceiling: E {e} -> h {e/SEAL_CMAX:.1f} (ceiling {SEAL_CMAX/e*400:.1f} %BM/d)")
print("   porpoises mass ceiling: h 28.6 on every pair (0.035 t/t/tick = 14 %BM/d)")

print("\n== Saturation and energy return at grid-mean density (rest share 0.4 at the pair floor)")
cases = [
    # name, a, h, density, floor, ass, E, cost, type
    ("seal->gadoids OLD", 0.2, 26.0, OLD_GAD_D, 0.85, 0.85, 5500, 75, 2),
    ("seal->benth a0.2 (carry)", 0.2, 26.0, PREY_D["benth"], 0.85, 0.85, 5500, 75, 2),
    ("seal->benth a0.1 (rec)", 0.1, 26.0, PREY_D["benth"], 0.85, 0.85, 5500, 75, 2),
    ("seal->pred a0.2 (carry)", 0.2, 26.0, PREY_D["pred"], 0.75, 0.85, 5500, 75, 2),
    ("seal->pred a0.05 (rec)", 0.05, 26.0, PREY_D["pred"], 0.75, 0.85, 5500, 75, 2),
    ("seal->plank III 2e-3", 2e-3, 35.0, PREY_D["plank"], 0.35, 0.88, 7500, 75, 3),
    ("porp->gadoids OLD", 0.3, 28.6, OLD_GAD_D, 0.75, 0.84, 5500, 90, 2),
    ("porp->benth a0.3 (carry)", 0.3, 28.6, PREY_D["benth"], 0.75, 0.84, 5500, 90, 2),
    ("porp->benth a0.12 (rec)", 0.12, 28.6, PREY_D["benth"], 0.75, 0.84, 5500, 90, 2),
    ("porp->pred a0.3 (carry)", 0.3, 28.6, PREY_D["pred"], 0.90, 0.84, 5500, 90, 2),
    ("porp->pred a0.2 (rec)", 0.2, 28.6, PREY_D["pred"], 0.90, 0.84, 5500, 90, 2),
    ("porp->plank a0.0225", 0.0225, 28.6, PREY_D["plank"], 0.95, 0.90, 7500, 90, 2),
    ("porp->plank a0.0225 @5t/km2", 0.0225, 28.6, 5.0, 0.95, 0.90, 7500, 90, 2),
]
for name, a, h, b, fl, ass, e, cost, typ in cases:
    bv = vis(b, fl)
    i = intake(a, h, bv, typ)
    be = breakeven(cost, ass, e, a, h, typ)
    print(f"   {name:30s} B {b:.2f} vis {bv:.2f}  I {i:.4f} ({100*i*h:4.0f} % of ceiling)  "
          f"E {ass*e*i:5.0f} MJ/t/tick  break-even visible {be:.2f} t/km2")

print("\n== a that keeps the old calibrated return at the new grid mean (a*B invariance)")
i_seal = intake(0.2, 26.0, vis(OLD_GAD_D, 0.85))
i_porp = intake(0.3, 28.6, vis(OLD_GAD_D, 0.75))
print(f"   old seal I {i_seal:.4f}: benth a {a_same_return(i_seal, 26.0, vis(PREY_D['benth'],0.85)):.3f}, "
      f"pred a {a_same_return(i_seal, 26.0, vis(PREY_D['pred'],0.75)):.3f}")
print(f"   old porp I {i_porp:.4f}: benth a {a_same_return(i_porp, 28.6, vis(PREY_D['benth'],0.75)):.3f}, "
      f"pred a {a_same_return(i_porp, 28.6, vis(PREY_D['pred'],0.90)):.3f}")

print("\n== Per-tonne selectivity (diet share / prey biomass), relative to planktivores")
for p, d in pred.items():
    s = {k: v[0] / PREY_T[k] for k, v in d["diet"].items()}
    print(f"   {p}: " + ", ".join(f"{k} {s[k]/s['plank']:.2f}" for k in ("plank", "benth", "pred")))
print("   old porpoise fit (151): gadoids 4.8x herring per tonne")

print("\n== Seal equilibrium under K: B* = K (g - M1 - starv) / r_max, K 27 t, r_max 0.11")
for g, s in [(0.18, 0.01), (0.18, 0.0), (0.20, 0.0), (0.17, 0.02)]:
    print(f"   g {g:.2f} starv {s:.2f}: B* {27*(g-0.10-s)/0.11:.1f} t")
