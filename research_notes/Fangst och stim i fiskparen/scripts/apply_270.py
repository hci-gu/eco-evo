"""Apply tables/yaml_edits_270.txt to fgconfig/fg_library.yaml as text (keeps every comment).

Pairs and species: a '# Section 270:' comment (the edit's `why`) before the first edited key, then each
key replaced in place, or inserted after max_intake_rate (pairs) when absent. The benthivores'
juvenile_stage gets recruit_depth_table after recruit_import. Refuses to run twice.
usage: python3 apply_270.py <repo>"""
import re
import sys
import textwrap

import yaml

repo = sys.argv[1]
here = __file__.rsplit('/scripts/', 1)[0]
edits = yaml.safe_load(open(f"{here}/tables/yaml_edits_270.txt"))
path = f"{repo}/fgconfig/fg_library.yaml"
text = open(path, encoding='utf-8').read()
if "# Section 270" in text:
    sys.exit("already applied")
lines = text.split('\n')


def block(name):
    hits = [n for n, l in enumerate(lines) if l == f"  {name}:"]
    assert len(hits) == 1, (name, hits)
    i = hits[0]
    j = i + 1
    while j < len(lines) and not re.match(r'^  [A-Za-z_]+:$', lines[j]) and not re.match(r'^[A-Za-z_]+:', lines[j]):
        j += 1
    return i, j


def fmt(v):
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:g}" if (v != 0 and abs(v) < 1e-3) else repr(v)
    return repr(v)


def comment(why):
    body = textwrap.wrap(f"Section 270: {' '.join(why.split())}", width=94)
    return [f"    # {l}" for l in body]


def apply(name, keys, why, after="max_intake_rate"):
    i, j = block(name)
    seg = lines[i:j]
    first = None
    for key, value in keys.items():
        hit = [n for n, l in enumerate(seg) if l.startswith(f"    {key}:")]
        line = f"    {key}: {fmt(value)}"
        if hit:
            seg[hit[0]] = line
            n = hit[0]
        else:
            anchor = [n for n, l in enumerate(seg) if l.startswith(f"    {after}:")]
            assert anchor, (name, key)
            n = anchor[0] + 1
            seg.insert(n, line)
        first = n if first is None else min(first, n)
    # the comment goes before the first edited line, above any comment block directly over it
    k = first
    while k > 1 and seg[k - 1].startswith("    #"):
        k -= 1
    seg[k:k] = comment(why)
    lines[i:j] = seg


for name, spec in edits["pairs"].items():
    spec = dict(spec)
    why = spec.pop("why")
    apply(name, spec, why)
for name, spec in edits["species"].items():
    spec = dict(spec)
    why = spec.pop("why")
    apply(name, spec, why)

for name, rows in edits["recruit_depth_table"].items():
    i, j = block(name)
    hit = [n for n in range(i, j) if lines[n].startswith("      recruit_import:")]
    assert len(hit) == 1, name
    n = hit[0] + 1
    new = [
        "      # Section 270: bed-depth shape of the recruit import (equal-cell mean 1 on the depth",
        "      # map; ones without one): NS-IBTS GOV 2000-25 in the box, haddock age 0 in Q3 (0.39 of",
        "      # the import tonnes) and cod age 1 in Q1 (0.61), Walker q; 95 % year-bootstrap",
        "      # 2.42-2.93 / 0.85-1.17 / 0.12-0.32 / 0.03-0.09; >= 250 m from the wider window (GOV",
        "      # standard stations stop at 250 m). Only the arrival: J then moves with B.",
        "      # research_notes/\"Rekryternas bottendjup\" summary.md.",
        "      recruit_depth_table:",
    ] + [f"        - {{depth_m: {r['depth_m']}, weight: {r['weight']:.2f}}}" for r in rows]
    lines[n:n] = new

open(path, 'w', encoding='utf-8').write('\n'.join(lines))
print(f"applied {len(edits['pairs'])} pairs, {len(edits['species'])} species, "
      f"{len(edits['recruit_depth_table'])} recruit table(s)")
