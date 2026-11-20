"""Per-prey intake plateaus implied by the chain's band profiles (no engine change; an independent test against
Rindorf & Gislason 2005, whiting plateaus 0.06-0.22 of Cmax per prey group).

Saturation per 1-m depth bin instead of one shared denominator over the whole band:
  concurrent (engine today): S = X / (1 + X),  X = sum_j a_j h_j B_vis,j (budget_gate equal-cell values)
  per depth:                 S = sum_z f(z) X(z) / (1 + X(z)),  X_j(z, b) = X_j phi_j(z, b) / mean_b O_j,
                             O_j(b) = sum_z f(z, b) phi_j(z, b)  (so the bed-mean encounter equals the library's)
f = the predator member's time-at-depth density, phi = prey profiles (krill_pairs_wc.py / one_rule_pairs.py /
band_pairs.py; [J] epifauna on the bed in the bottom 1 m). The vision radius's depth dependence inside the
overlap is ignored [J]; day and night profiles averaged by the day share. Beds: the chain's BEDS with their cell
weights. Output: S per prey alone (x1 and x10 its density) and all prey together, concurrent vs per depth.
Usage: python3 band_plateaus.py <repo_root> <scratch>
"""
import sys, os, numpy as np
ROOT = os.path.abspath(sys.argv[1]); SCR = os.path.abspath(sys.argv[2])
ORP = os.path.join(ROOT, "research_notes", "En synregel for fiskparen", "scripts", "one_rule_pairs.py")
src = open(ORP).read(); head = src[:src.index("# =============================================================== run")]
sv = list(sys.argv); sys.argv = [ORP, ROOT, SCR, os.path.join(SCR, "_u.json")]
O = {"__name__": "orp_head", "__file__": ORP}; exec(compile(head, ORP, "exec"), O); sys.argv = sv
K = O["K"]; ZG = O["ZG"]; BEDS = np.asarray(O["BEDS"], float); WC_ = np.asarray(O["WCELL"], float); WC_ = WC_ / WC_.sum()
q = dict(O["qk0"]); q.setdefault("H_whiting", O["qf"].get("H_whiting", 10.0)) if "H_whiting" not in q else None
FDAY = O["FDAY"]; uniform = K["uniform"]
print(f"beds {len(BEDS)}, day share {FDAY:.2f}, H_whiting {q.get('H_whiting')}, phi_follow {q.get('phi_follow')}")

def plk(b, qq, n):
    return 0.60 * K["herring_prof"](b, qq, n) + 0.35 * K["pout_prof"](b, qq, n) + 0.05 * K["bw_prof"](False)(b, qq, n)
PREY = {"krill": K["krill_profile"], "planktivorous_fish": plk,
        "mobile_epifauna": lambda b, qq, n: uniform(b - 1.0, b),
        "benthivorous_fish": lambda b, qq, n: uniform(b - O["qf"]["H_juv"], b),
        "whiting": lambda b, qq, n: uniform(b - qq["H_whiting"], b)}
# budget_gate equal-cell saturation per pair (live library): X = sat / (1 - sat)
SAT = {"whiting": {"krill": 0.95, "planktivorous_fish": 0.82, "mobile_epifauna": 0.60, "benthivorous_fish": 0.38},
       "predatory_fish": {"krill": 0.85, "planktivorous_fish": 0.90, "benthivorous_fish": 0.60, "whiting": 0.44}}
PRED = {"whiting": [("whiting", K["whiting_prof"], 1.0)],
        "predatory_fish": [("saithe", K["saithe_field"], 0.80), ("hake", K["hake_field"], 0.20)]}

def prof(fn, b, n):
    v = np.asarray(fn(b, q, n), float); s = v.sum(); return v / s if s > 0 else v

for pred, members in PRED.items():
    X = {j: s / (1 - s) for j, s in SAT[pred].items()}
    print(f"\n=== {pred} (members {', '.join(f'{m} {w}' for m, _, w in members)}); X = a h B_vis {', '.join(f'{j} {x:.1f}' for j, x in X.items())}")
    print(f"  concurrent, all prey: S = {sum(X.values()) / (1 + sum(X.values())):.3f}")
    res = {}
    for mname, mfn, mw in members:
        for night in (False, True):
            wt = (1 - FDAY) if night else FDAY
            F = np.array([prof(mfn, b, night) for b in BEDS])                 # (nb, nz)
            PH = {j: np.array([prof(PREY[j], b, night) for b in BEDS]) for j in X}
            Obar = {j: float((WC_ * (F * PH[j]).sum(1)).sum()) for j in X}
            for x in (1.0, 10.0):
                for j in list(X) + ["ALL"]:
                    js = list(X) if j == "ALL" else [j]
                    Xz = sum(X[k] * x * PH[k] / max(Obar[k], 1e-30) for k in js)   # (nb, nz)
                    per = (F * Xz / (1 + Xz)).sum(1)
                    Xb = sum(X[k] * x * (F * PH[k]).sum(1) / max(Obar[k], 1e-30) for k in js)
                    conc = Xb / (1 + Xb)
                    share = (F * (sum(PH[k] for k in js) > 0)).sum(1)
                    key = (j, x)
                    r = res.setdefault(key, np.zeros(3))
                    r += mw * wt * np.array([(WC_ * conc).sum(), (WC_ * per).sum(), (WC_ * share).sum()])
    print(f"  {'prey (alone)':20s} {'x':>4s} {'concurrent':>10s} {'per depth':>10s} {'time share where prey occurs':>28s}")
    for (j, x), r in res.items():
        print(f"  {j:20s} {x:4.0f} {r[0]:10.3f} {r[1]:10.3f} {r[2]:28.3f}")
