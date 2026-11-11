"""tools/probes/budget_gate.py (unchanged source) on temporary library copies whose nine fish-prey pairs carry
the capture-probability factor p_new / p_library of p_ppr.py on max_intake_rate (a is linear in p in every
section 269 term), nothing else changed. Plus one combined reading: central p x the schooling factor of
research_notes/"Stimskydd i motes-a" recomputed with the new dispersed p (S = ... k / (N p), so the schooled
share keeps its absolute rate while the dispersed share rises with p).

Usage: python3 gate_p.py <repo_root> <p_ppr.json> <scratch_dir> <out_dir>

Same mechanism as research_notes/"Stimskydd i motes-a"/scripts/gate_school.py: budget_gate.py is exec'd from
its own file with one textual change (library_path=<copy>); nothing in the repository is written. The
"library" run must reproduce today's gate. The gate reads the annual-mean a only (dark_ratio does not enter).
"""
import contextlib
import io
import json
import os
import re
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
PP = json.load(open(sys.argv[2]))
SCR = os.path.abspath(sys.argv[3])
OUTD = os.path.abspath(sys.argv[4])
os.makedirs(SCR, exist_ok=True)
os.makedirs(OUTD, exist_ok=True)
BG = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
BG_SRC = open(BG).read()
OLD = "apply_natural_mortality=True, migration=True)"
assert BG_SRC.count(OLD) == 1
LIB0 = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
INTER = "interaction_definitions"

# ---------------------------------------------------------------- schooling factor with a new dispersed p
SN = os.path.join(ROOT, "research_notes", "Stimskydd i motes-a", "scripts")
SF_SRC = open(os.path.join(SN, "school_factor.py")).read()
SF_HEAD = SF_SRC[:SF_SRC.index("res = {}")]
argv0 = list(sys.argv)
sys.argv = ["school_factor.py", os.path.join(SN, "pair_terms.json"), os.path.join(SCR, "_unused.json")]
NS = {"__name__": "school_factor_head"}
exec(compile(SF_HEAD, "school_factor.py", "exec"), NS)
sys.argv = argv0


def school_F(pair, p_disp, i=1):
    """Central schooling factor of the pair with dispersed capture probability p_disp (S ~ 1 / p)."""
    NS["P_I"] = p_disp
    d = NS["pairs"][pair]
    tot = 0.0
    for sp, per, share, R in d["rows"]:
        tot += share * (1 - NS["sigma"](sp, per, i) * (1 - NS["S_eff"](sp, R, i)))
    return tot


PAIRS = list(PP["pairs"])
factors = {"library": {p: 1.0 for p in PAIRS}}
for v in ("low", "percid", "central_full", "central", "per_strike", "high"):
    factors[v] = {p: PP["pairs"][p][v]["factor"] for p in PAIRS}
factors["school_only"] = {p: school_F(p, 0.2) for p in PAIRS}
factors["central_x_school"] = {p: PP["pairs"][p]["central"]["factor"] * school_F(p, PP["pairs"][p]["central"]["p"])
                               for p in PAIRS}
factors["full_x_school"] = {p: PP["pairs"][p]["central_full"]["factor"]
                            * school_F(p, PP["pairs"][p]["central_full"]["p"]) for p in PAIRS}
print("Factor on max_intake_rate per pair and reading (school_only = Stimskydd central F, reproduced)")
print(f"  {'pair':40s} " + " ".join(f"{v[:16]:>16s}" for v in factors))
for p in PAIRS:
    print(f"  {p:40s} " + " ".join(f"{factors[v][p]:16.3f}" for v in factors))
print("  schooling F at the new p (central_x_school / central):")
for p in PAIRS:
    print(f"    {p:40s} F(p 0.2) {school_F(p, 0.2):.3f}  F(p {PP['pairs'][p]['central']['p']:.2f}) "
          f"{school_F(p, PP['pairs'][p]['central']['p']):.3f}")


def run_gate(libpath):
    src = BG_SRC.replace(OLD, f"apply_natural_mortality=True, migration=True, library_path={libpath!r})")
    buf = io.StringIO()
    ns = {"__name__": "__main__", "__file__": BG}
    cwd = os.getcwd()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, BG, "exec"), ns)
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    return buf.getvalue()


def parse(txt):
    out = {"prey_total": {}, "prey_pair": {}, "diet": {}, "diet_field": {}}
    sec = txt[txt.index("=== prey-side check"):]
    prey = None
    for line in sec.splitlines():
        m = re.match(r"\s*([a-z_]+): booked M2 ([\d.e-]+) /yr, implied ([\d.e-]+) \(ratio ([\d.]+)\)", line)
        if m:
            prey = m.group(1)
            out["prey_total"][prey] = float(m.group(4))
            continue
        m = re.match(r"\s*([a-z_]+) booked\s+([\d.e-]+) \(([\d.]+)\)\s+implied\s+([\d.e-]+) ratio\s+([\d.]+)\s*(OFF)?", line)
        if m and prey:
            out["prey_pair"][f"{m.group(1)}->{prey}"] = (float(m.group(5)), bool(m.group(6)))
    fs = txt[txt.index("=== diet gate at the field spread"):txt.index("=== prey-side check")]
    pred = None
    for line in fs.splitlines():
        m = re.match(r"\s*([a-z_]+) \(", line)
        if m:
            pred = m.group(1)
            continue
        m = re.match(r"\s*([a-z_]+) enc\s+([\d.]+) booked\s+([\d.]+) ratio\s+([\d.]+)\s+(ok|OFF)", line)
        if m and pred:
            out["diet_field"][f"{pred}->{m.group(1)}"] = (float(m.group(4)), m.group(5) == "OFF")
    for key, a, b in (("diet", "=== diet gate: encounter", "=== diet gate at the field spread"),):
        dg = txt[txt.index(a):txt.index(b)]
        pred = None
        for line in dg.splitlines():
            m = re.match(r"\s*([a-z_]+)\s+prey\s+B\s+B_vis", line)
            if m:
                pred = m.group(1)
                continue
            m = re.match(r"\s*([a-z_]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(ok|OFF)", line)
            if m and pred:
                out[key][f"{pred}->{m.group(1)}"] = (float(m.group(4)), m.group(6) == "OFF")
    return out


summary = {}
for vn, F in factors.items():
    lib = json.loads(json.dumps(LIB0))
    inter = lib[INTER]
    if vn != "library":
        for name in PAIRS:
            pred, prey = name.split("->")
            key = f"{pred}_preys_on_{prey}"
            inter[key]["max_intake_rate"] = float(f"{inter[key]['max_intake_rate'] * F[name]:.6g}")
    path = os.path.join(SCR, f"fg_library_{vn}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False, allow_unicode=True)
    txt = run_gate(path)
    open(os.path.join(OUTD, f"budget_gate_{vn}.txt"), "w").write(txt)
    summary[vn] = parse(txt)
    print(f"{vn}: done")

json.dump({"factors": factors, "gate": summary}, open(os.path.join(OUTD, "gate_p.json"), "w"), indent=1)
V = list(factors)
FOCUS = ["planktivorous_fish", "benthivorous_fish", "whiting", "mackerel", "krill", "zooplankton", "mobile_epifauna"]
print("\nPrey side: implied / booked M2 per prey (total)")
print(f"  {'prey':20s} " + " ".join(f"{v[:16]:>16s}" for v in V))
for p in FOCUS:
    print(f"  {p:20s} " + " ".join(f"{summary[v]['prey_total'].get(p, float('nan')):16.2f}" for v in V))
for title, key in (("Prey side per pair (ratio; * = OFF)", "prey_pair"),
                   ("Predator-side diet gate, equal cells (encounter share / booked share; * = OFF)", "diet"),
                   ("Predator-side diet gate, field spread (* = OFF)", "diet_field")):
    print(f"\n{title}")
    keys = sorted({k for v in V for k in summary[v][key]})
    for k in keys:
        if key == "prey_pair" and not any(x in k.split("->")[1] for x in ("planktivorous_fish", "benthivorous_fish", "whiting", "mackerel", "krill")):
            continue
        if key != "prey_pair" and k.split("->")[0] not in ("predatory_fish", "whiting", "benthivorous_fish", "mackerel"):
            continue
        row = []
        for v in V:
            r = summary[v][key].get(k)
            row.append(f"{r[0]:15.2f}{'*' if r[1] else ' '}" if r else f"{'-':>16s}")
        print(f"  {k:40s} " + " ".join(row))
