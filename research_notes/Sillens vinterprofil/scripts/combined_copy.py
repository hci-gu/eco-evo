"""Combined derivation (user, 2026-11-11): the per-encounter capture probability of research_notes/
"Fangstsannolikhet for fiskbyten" (hungry convention, central p per pair) and the schooling factor of
research_notes/"Stimskydd i motes-a" with the Norway pout values of research_notes/"Vitlinglojans stim",
recomputed at each pair's NEW dispersed p (S = ... k / (N p): the schooled share keeps its absolute rate).
The benthivore pairs now use their own p (the schooling note had 0.2 where the library carries 0.3).

factor on max_intake_rate = (p_new / p_library) x F(p_new)
night/day' = night/day x F_night / F_day (p is light-independent), except stomach-derived ratios (kept).

budget_gate.py runs unchanged on temporary library copies (library_path=<copy>); nothing in the repository
is written. Research only; no value is chosen from the gate.

Usage: python3 combined.py <repo_root> <scratch_dir> <out_dir>
"""
import json
import os
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTD = os.path.abspath(sys.argv[3])
os.makedirs(SCR, exist_ok=True)
os.makedirs(OUTD, exist_ok=True)
NOTES = os.path.join(ROOT, "research_notes")
P_JSON = os.path.join(NOTES, "Fangstsannolikhet for fiskbyten", "scripts", "p_ppr.json")
POUT_PY = os.path.join(NOTES, "Vitlinglojans stim", "scripts", "school_factor_pout.py")
TERMS = os.path.join(NOTES, "Stimskydd i motes-a", "scripts", "pair_terms.json")
GATE_PY = os.path.join(NOTES, "Fangstsannolikhet for fiskbyten", "scripts", "gate_p.py")

PP = json.load(open(P_JSON))["pairs"]

# --- the pout note's school-factor algebra, up to its variants (prints its section 1-2 header quietly)
src = open(POUT_PY).read()
head = src[:src.index("VARIANTS = {")]
argv0 = list(sys.argv)
sys.argv = ["school_factor_pout.py", TERMS, os.path.join(SCR, "_unused.json")]
import contextlib
import io
NS = {"__name__": "pout_head"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, POUT_PY, "exec"), NS)
sys.argv = argv0
PAR = NS["PAR_NEW"]
SCHOOL_PAIRS = NS["pairs"]

# --- budget_gate runner and parser from gate_p.py (functions only)
gsrc = open(GATE_PY).read()
GNS = {"__name__": "gate_head", "ROOT": ROOT, "os": os, "sys": sys}
BG = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
GNS["BG"] = BG
GNS["BG_SRC"] = open(BG).read()
GNS["OLD"] = "apply_natural_mortality=True, migration=True)"
assert GNS["BG_SRC"].count(GNS["OLD"]) == 1
exec(compile("import contextlib, io, re\n" + gsrc[gsrc.index("def run_gate"):gsrc.index("summary = {}")],
             GATE_PY, "exec"), GNS)
run_gate, parse = GNS["run_gate"], GNS["parse"]

READINGS = {
    # pout central (sigma day 0.4, night 0; N 10 000; R_s 13 m)
    "central": None,
    # pout and blue whiting not schooling (clupeids only)
    "clupeids_only": {"sigma": {"pout": (0.0, 0.0), "blue whiting": (0.0, 0.0)}},
    # pout at its high-protection end of the note (sigma 0.7 / 0.1, N 50 000, R_s 10)
    "pout_high": {"sigma": {"pout": (0.7, 0.1)}, "N": {"pout": 50000}, "R_s": {"pout": 10.0}},
}
STOMACH_ND = {"mackerel->planktivorous_fish"}   # night/day from the stomachs (section 228.4 / 269): kept

LIB0 = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
INTER = LIB0["interaction_definitions"]

res = {"pairs": {}, "gate": {}}
factors = {"library": {}, "p_only": {}}
for r in READINGS:
    factors[r] = {}
print("Per pair: p library -> new, p factor, schooling F at the new p (day / night), total factor, night/day")
for name, pp in PP.items():
    pred, prey = name.split("->")
    key = f"{pred}_preys_on_{prey}"
    a0 = INTER[key]["max_intake_rate"]
    nd0 = INTER[key].get("dark_ratio")
    p_lib, p_new = pp["p_lib"], pp["central"]["p"]
    pf = p_new / p_lib
    factors["library"][name] = 1.0
    factors["p_only"][name] = pf
    res["pairs"][name] = dict(a=a0, dark_ratio=nd0, p_lib=p_lib, p_new=p_new, p_factor=pf, readings={})
    print(f"  {name:40s} a {a0:.4g}  p {p_lib:.2f} -> {p_new:.3f} (x{pf:.2f})  night/day {nd0}")
    for rn, over in READINGS.items():
        if name in SCHOOL_PAIRS:
            NS["P_I"] = p_new
            f = NS["pair_factor"](SCHOOL_PAIRS[name], PAR, 1, over)
        else:
            f = dict(F=1.0, F_day=1.0, F_night=1.0, nd_new=nd0)
        tot = pf * f["F"]
        if nd0 is None or name in STOMACH_ND:
            nd_new = nd0
        else:
            nd_new = nd0 * f["F_night"] / f["F_day"]
        factors[rn][name] = tot
        res["pairs"][name]["readings"][rn] = dict(F=f["F"], F_day=f["F_day"], F_night=f["F_night"],
                                                  factor=tot, a_new=a0 * tot, dark_ratio_new=nd_new)
        print(f"      {rn:14s} F {f['F']:.3f} (day {f['F_day']:.3f}, night {f['F_night']:.3f})  "
              f"factor x{tot:.3f}  a {a0 * tot:.4g}  night/day {nd_new if nd_new is None else round(nd_new, 4)}")

summary = {}
for vn, F in factors.items():
    lib = json.loads(json.dumps(LIB0))
    inter = lib["interaction_definitions"]
    if vn != "library":
        for name in PP:
            pred, prey = name.split("->")
            key = f"{pred}_preys_on_{prey}"
            inter[key]["max_intake_rate"] = float(f"{inter[key]['max_intake_rate'] * F[name]:.6g}")
    path = os.path.join(SCR, f"fg_library_combined_{vn}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False, allow_unicode=True)
    txt = run_gate(path)
    open(os.path.join(OUTD, f"budget_gate_{vn}.txt"), "w").write(txt)
    summary[vn] = parse(txt)
res["gate"] = summary
res["factors"] = factors
json.dump(res, open(os.path.join(OUTD, "combined.json"), "w"), indent=1)

V = list(factors)
FOCUS = ["planktivorous_fish", "benthivorous_fish", "whiting", "mackerel", "krill", "zooplankton", "mobile_epifauna"]
print("\nPrey side: implied / booked M2 per prey (total)")
print(f"  {'prey':20s} " + " ".join(f"{v:>14s}" for v in V))
for p in FOCUS:
    print(f"  {p:20s} " + " ".join(f"{summary[v]['prey_total'].get(p, float('nan')):14.2f}" for v in V))
for title, key in (("Prey side per pair (ratio; * = OFF)", "prey_pair"),
                   ("Predator-side diet gate, equal cells (* = OFF)", "diet"),
                   ("Predator-side diet gate, field spread (* = OFF)", "diet_field")):
    print(f"\n{title}")
    keys = sorted({k for v in V for k in summary[v][key]})
    for k in keys:
        if key == "prey_pair" and not any(x in k.split("->")[1] for x in
                                          ("planktivorous_fish", "benthivorous_fish", "whiting", "mackerel", "krill")):
            continue
        if key != "prey_pair" and k.split("->")[0] not in ("predatory_fish", "whiting", "benthivorous_fish", "mackerel"):
            continue
        row = []
        for v in V:
            r = summary[v][key].get(k)
            row.append(f"{r[0]:13.2f}{'*' if r[1] else ' '}" if r else f"{'-':>14s}")
        print(f"  {k:40s} " + " ".join(row))
print("\nOFF pairs per reading (prey side / equal cells / field spread):")
for v in V:
    offs = sorted({k for key in ("prey_pair", "diet", "diet_field") for k, r in summary[v][key].items() if r[1]})
    print(f"  {v:14s} {', '.join(offs)}")
