"""tools/probes/budget_gate.py (unchanged source) on temporary library copies whose fish-prey pairs carry the
schooling factor of school_factor.py: max_intake_rate x pair factor F, nothing else changed. The gate reads the
annual-mean a only (dark_ratio does not enter it), so night/day is reported by school_factor.py, not here.

Usage: python3 gate_school.py <repo_root> <school_factor.json> <scratch_dir> <out_dir>

Same mechanism as research_notes/"En synregel for fiskparen"/scripts/gate_variants.py: budget_gate.py is exec'd
from its own file with one textual change (library_path=<copy>); nothing in the repository is written. The
"library" run must reproduce today's gate.
"""
import contextlib
import io
import json
import os
import re
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
SF = json.load(open(sys.argv[2]))
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
    out = {"prey_total": {}, "prey_pair": {}, "diet": {}}
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
    dg = txt[txt.index("=== diet gate: encounter"):txt.index("=== diet gate at the field spread")]
    pred = None
    for line in dg.splitlines():
        m = re.match(r"\s*([a-z_]+)\s+prey\s+B\s+B_vis", line)
        if m:
            pred = m.group(1)
            continue
        m = re.match(r"\s*([a-z_]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(ok|OFF)", line)
        if m and pred:
            out["diet"][f"{pred}->{m.group(1)}"] = (float(m.group(4)), m.group(6) == "OFF")
    return out


variants = ["library", "low", "central", "high", "central_clupeids_only", "central_fieldS"]
pairs = [k for k in SF["pairs"] if "->" in k]
summary = {}
for vn in variants:
    lib = json.loads(json.dumps(LIB0))
    inter = lib[INTER]
    if vn != "library":
        for name in pairs:
            pred, prey = name.split("->")
            key = f"{pred}_preys_on_{prey}"
            F = SF["pairs"][name][vn]["F"]
            inter[key]["max_intake_rate"] = float(f"{inter[key]['max_intake_rate'] * F:.6g}")
    path = os.path.join(SCR, f"fg_library_{vn}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False, allow_unicode=True)
    txt = run_gate(path)
    open(os.path.join(OUTD, f"budget_gate_{vn}.txt"), "w").write(txt)
    summary[vn] = parse(txt)
    print(f"{vn}: done")

json.dump(summary, open(os.path.join(OUTD, "gate_school.json"), "w"), indent=1)
FOCUS_PREY = ["planktivorous_fish", "benthivorous_fish", "whiting", "mackerel", "krill", "zooplankton", "mobile_epifauna"]
print("\nPrey side: implied / booked M2 per prey (total)")
print(f"  {'prey':20s} " + " ".join(f"{v[:13]:>13s}" for v in variants))
for p in FOCUS_PREY:
    print(f"  {p:20s} " + " ".join(f"{summary[v]['prey_total'].get(p, float('nan')):13.2f}" for v in variants))
print("\nPrey side per pair (ratio; * = OFF)")
keys = sorted({k for v in variants for k in summary[v]["prey_pair"]})
for k in keys:
    if not any(x in k for x in ("planktivorous_fish", "benthivorous_fish", "whiting", "mackerel", "krill")):
        continue
    row = []
    for v in variants:
        r = summary[v]["prey_pair"].get(k)
        row.append(f"{r[0]:12.2f}{'*' if r[1] else ' '}" if r else f"{'-':>13s}")
    print(f"  {k:40s} " + " ".join(row))
print("\nPredator-side diet gate (encounter share / booked share at equal cells; * = OFF)")
keys = sorted({k for v in variants for k in summary[v]["diet"]})
for k in keys:
    if k.split("->")[0] not in ("predatory_fish", "whiting", "benthivorous_fish", "mackerel"):
        continue
    row = []
    for v in variants:
        r = summary[v]["diet"].get(k)
        row.append(f"{r[0]:12.2f}{'*' if r[1] else ' '}" if r else f"{'-':>13s}")
    print(f"  {k:40s} " + " ".join(row))
