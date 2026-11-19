"""Gate (a check, never an input): tools/probes/budget_gate.py on the live library and on the section 280 copy, with
library_path=<file> patched in memory exactly as research_notes/"Fangstsannolikhet for fiskbyten"/scripts/
gate_p.py does (its run_gate and parse executed from its own file). The runner is validated first: the live
library through library_path must print the same text as a plain `python3 tools/probes/budget_gate.py`. Then the
loader (lib.config.config_loader.load_project_config, mareld2.yaml, seed 1, as final_gate.py) on both files: every
derived growth_rate must be equal (growth_budget derives g from bookings, which section 280 does not touch).

Usage: python3 gate_280.py <repo_root> <copy library yaml> <out_dir>
"""
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import warnings

ROOT = os.path.abspath(sys.argv[1])
COPY = os.path.abspath(sys.argv[2])
OUTD = os.path.abspath(sys.argv[3])
os.makedirs(OUTD, exist_ok=True)
LIVE = os.path.join(ROOT, "fgconfig", "fg_library.yaml")
GATE_PY = os.path.join(ROOT, "research_notes", "Fangstsannolikhet for fiskbyten", "scripts", "gate_p.py")
BG = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
gsrc = open(GATE_PY).read()
G = {"__name__": "gate_head", "ROOT": ROOT, "os": os, "sys": sys, "BG": BG, "BG_SRC": open(BG).read(),
     "OLD": "apply_natural_mortality=True, migration=True)"}
assert G["BG_SRC"].count(G["OLD"]) == 1
exec(compile("import contextlib, io, re\n" + gsrc[gsrc.index("def run_gate"):gsrc.index("summary = {}")], GATE_PY, "exec"), G)
run_gate, parse = G["run_gate"], G["parse"]

direct = subprocess.run([sys.executable, BG], cwd=ROOT, capture_output=True, text=True).stdout
t_live = run_gate(LIVE)
t_copy = run_gate(COPY)
open(os.path.join(OUTD, "budget_gate_live_direct.txt"), "w").write(direct)
open(os.path.join(OUTD, "budget_gate_live.txt"), "w").write(t_live)
open(os.path.join(OUTD, "budget_gate_280.txt"), "w").write(t_copy)
print(f"runner check: library_path=<live> output {'IDENTICAL to' if t_live == direct else 'DIFFERS from'} the plain run")
P = {"live": parse(t_live), "280": parse(t_copy)}

# total ration rows
def ration(txt):
    out = {}
    sec = txt[txt.index("=== total ration"):] if "=== total ration" in txt else ""
    for line in sec.splitlines():
        m = re.match(r"\s*([a-z_]+)\s+.*?([\d.]+)\s*$", line)
        if m:
            out[m.group(1)] = line.strip()
    return out


rows = []
for key, title in (("prey_total", "prey side, total implied / booked M2"), ("prey_pair", "prey side per pair"),
                   ("diet", "predator side, equal cells (encounter / booked share)"),
                   ("diet_field", "predator side, field spread")):
    print(f"\n{title}  (live -> 280; * = OFF)")
    ks = sorted(set(P["live"][key]) | set(P["280"][key]))
    for k in ks:
        a, b = P["live"][key].get(k), P["280"][key].get(k)
        if key == "prey_total":
            print(f"  {k:44s} {a if a is None else f'{a:6.2f}'} -> {b if b is None else f'{b:6.2f}'}")
            continue
        fa = "-" if a is None else f"{a[0]:6.2f}{'*' if a[1] else ' '}"
        fb = "-" if b is None else f"{b[0]:6.2f}{'*' if b[1] else ' '}"
        flag = ""
        if a and b and a[1] != b[1]:
            flag = "  <- goes OFF" if b[1] else "  <- comes back ON"
        if a and b and (abs(b[0] / a[0] - 1) > 0.005 if a[0] else b[0] != 0) or flag:
            print(f"  {k:44s} {fa} -> {fb}{flag}")
        rows.append(dict(kind=key, pair=k, live=a, new=b))
offs = {v: sorted({f"{k} ({kind})" for kind in ("prey_pair", "diet", "diet_field") for k, r in P[v][kind].items() if r[1]})
        for v in P}
print("\nOFF, live :", "; ".join(offs["live"]))
print("OFF, 280  :", "; ".join(offs["280"]))
print("goes OFF  :", "; ".join(sorted(set(offs["280"]) - set(offs["live"]))) or "none")
print("back ON   :", "; ".join(sorted(set(offs["live"]) - set(offs["280"]))) or "none")
# raw check, independent of the parser: every OFF line of the gate text, numbers stripped
def raw_off(txt):
    return [re.sub(r"[0-9.e+-]+", "", l).strip() for l in txt.splitlines() if "OFF" in l]


rl, rn = raw_off(t_live), raw_off(t_copy)
print(f"raw OFF lines (numbers stripped): live {len(rl)}, 280 {len(rn)}; "
      f"{'identical' if rl == rn else 'DIFFERENT: ' + str(sorted(set(rn) ^ set(rl)))}")
for name, txt in (("live", t_live), ("280", t_copy)):
    sec = txt[txt.index("=== total ration"):] if "=== total ration" in txt else ""
    print(f"\ntotal ration ({name}):\n" + "\n".join(l for l in sec.splitlines()[:12]))

# loader: derived growth_rate on both files
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from lib.config.config_loader import load_project_config  # noqa: E402


def growth(lp):
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        res = load_project_config(os.path.join(ROOT, "mareld2.yaml"), library_path=lp, seed=1)
    fgs = res[0] if isinstance(res, tuple) else res
    if not isinstance(fgs, dict):
        fgs = {getattr(f, "id", getattr(f, "name", str(i))): f for i, f in enumerate(fgs)}
    out = {}
    for k, f in fgs.items():
        g = getattr(f, "params", {}).get("growth_rate", None) if hasattr(f, "params") else None
        if g is None:
            g = getattr(f, "growth_rate", None)
        out[str(k)] = g
    return out, [str(x.message) for x in w]


g_live, w_live = growth(LIVE)
g_new, w_new = growth(COPY)
print("\nLoader (mareld2.yaml, seed 1): derived growth_rate live -> 280")
moved = 0
for k in g_live:
    same = g_live[k] == g_new.get(k)
    moved += 0 if same else 1
    print(f"  {k:26s} {g_live[k]} -> {g_new.get(k)}{'' if same else '  CHANGED'}")
print(f"  {moved} growth_rate(s) moved; loader warnings live {len(w_live)}, 280 {len(w_new)}")
json.dump(dict(parsed=P, offs=offs, growth=dict(live=g_live, new=g_new), runner_identical=t_live == direct),
          open(os.path.join(OUTD, "gate_280.json"), "w"), indent=1, default=str)
