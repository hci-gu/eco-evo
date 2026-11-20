"""Bed-class light composite (NOT applied): each visual pair's class a from the section 280 chain run with that bed
class's measured light profile, as research_notes/"Djupljus i synparen" sensitivity.py composes it:
    a_c = row_c(run_c) x a_final(run_c) / a_269(run_c);   a = sum_c share_c a_c
Runs: base = live profile (K 20-50 0.109, 50-65 0.091, below 65 0.09); A (< 100 m beds) = K 0.17 / 0.15 / 0.16;
B (100-150 m beds) = 0.139 / 0.121 / 0.14 (VOTO gliders, research_notes/"Vitlingens skydd pa grunt vatten").
150-250 and >= 250 m keep the base run. dark_ratio and stage_access stay the base run's (printed per run).
Writes a library COPY with the new max_intake_rate and depth_access_table rows (rows = class a's; the loader
normalises the table to an equal-cell mean of 1).
usage: compose.py <scratch> <live library> <out library>
"""
import json, os, re, sys
import numpy as np

SC, LIVE, OUT = sys.argv[1:4]
RUNS = {'base': ('S_base', '0.09'), 'A': ('S_A', '0.16'), 'B': ('S_B', '0.14')}
CLASS_RUN = ('A', 'B', 'base', 'base')
SHARE = np.array([0.213, 0.306, 0.332, 0.149]); SHARE = SHARE / SHARE.sum()   # sensitivity.py's equal-cell shares
CLS_ROWS = (0, 1, 3, 5)

def load(run):
    d, k = RUNS[run]
    f = json.load(open(os.path.join(SC, d, f"final_{k}.json")))["final"]
    r = json.load(open(os.path.join(SC, d, f"rows269_{k}.json")))["rows"]
    return f, r

L = {run: load(run) for run in RUNS}
fb, rb = L['base']
out = {}
print(f"{'pair':46s} {'a base':>9s} {'a new':>9s} {'ratio':>6s}   class a base -> new (< 100 / 100-150 / 150-250 / >= 250)"
      "   <100:150-250 base -> new   night/day base / A / B")
for key in fb:
    if "depth_access_table" not in rb.get(key, {}):
        continue
    ac_b, ac_n = [], []
    for c in range(4):
        row = rb[key]["depth_access_table"][CLS_ROWS[c]]["factor"]
        ac_b.append(row * fb[key]["max_intake_rate"] / rb[key]["max_intake_rate"])
        f, r = L[CLASS_RUN[c]]
        row = r[key]["depth_access_table"][CLS_ROWS[c]]["factor"]
        ac_n.append(row * f[key]["max_intake_rate"] / r[key]["max_intake_rate"])
    a_b = float(SHARE @ np.array(ac_b)); a_n = float(SHARE @ np.array(ac_n))
    # the composite at the base run must reproduce the live a (table rounding)
    scale = fb[key]["max_intake_rate"] / a_b
    a_new = a_n * scale
    nd = [L[x][0][key].get("dark_ratio") for x in ('base', 'A', 'B')]
    rr = lambda v: v[0] / v[2] if v[2] > 0 else float('nan')
    out[key] = dict(a_live=fb[key]["max_intake_rate"], a_new=a_new, classes_base=ac_b, classes_new=ac_n, night_day=nd)
    print(f"{key:46s} {fb[key]['max_intake_rate']:9.4g} {a_new:9.4g} {a_new / fb[key]['max_intake_rate']:6.3f}   "
          + " / ".join(f"{x:.3g}" for x in ac_b) + " -> " + " / ".join(f"{x:.3g}" for x in ac_n)
          + f"   {rr(ac_b):6.2f} -> {rr(ac_n):6.2f}   " + " / ".join(str(x) for x in nd))
json.dump(out, open(os.path.join(SC, "composite.json"), "w"), indent=1)

# library copy
text = open(LIVE, encoding="utf-8").read(); lines = text.split("\n")
def block(name):
    hits = [n for n, l in enumerate(lines) if l == f"  {name}:"]
    assert len(hits) == 1, (name, hits)
    i = hits[0]; j = i + 1
    while j < len(lines) and not re.match(r'^  [A-Za-z_]+:$', lines[j]) and not re.match(r'^[A-Za-z_]+:', lines[j]):
        j += 1
    return i, j
num = lambda v: f"{v:.4g}"
for key, d in out.items():
    i, j = block(key)
    hit_a = [n for n in range(i, j) if re.match(r'^\s+max_intake_rate:', lines[n])]
    assert len(hit_a) == 1, key
    ind = lines[hit_a[0]][:len(lines[hit_a[0]]) - len(lines[hit_a[0]].lstrip())]
    lines[hit_a[0]] = f"{ind}max_intake_rate: {num(d['a_new'])}"
    rows = [n for n in range(i, j) if re.match(r'^\s+- \{depth_m: ', lines[n])]
    assert len(rows) == 6, (key, len(rows))
    cls = [0, 1, 1, 2, 2, 3]
    for n, c in zip(rows, cls):
        m = re.match(r'^(\s+- \{depth_m: [0-9.]+, factor: )[^}]+\}', lines[n]); assert m, lines[n]
        lines[n] = m.group(1) + num(d['classes_new'][c] * d['a_live'] / max(sum(SHARE * np.array(d['classes_base'])), 1e-30)) + "}"
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print(f"wrote {OUT} ({len(out)} pairs)")
