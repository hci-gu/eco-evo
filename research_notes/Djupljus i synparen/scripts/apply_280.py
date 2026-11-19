"""Apply tables/yaml_edits_280.txt to a library file as text (keeps every comment). Modelled on
research_notes/"Fangst och stim i fiskparen"/scripts/apply_270.py (comment and key logic) and
research_notes/"Bottenfiskens hojdprofil"/scripts/apply_269.py (depth_access_table replacement).

Per pair: a '# Section 280:' comment (the edit's `why`) before the first edited key; max_intake_rate,
dark_ratio and stage_access replaced in place (dark_ratio inserted after max_intake_rate when absent);
depth_access_table's rows replaced under its key. Then each comment_lines entry: the old line (on every krill
pair, the section 253 light comment) replaced by the new lines. Refuses to run twice.

The target library is an explicit argument, so a run cannot hit the live file by accident; give
fgconfig/fg_library.yaml only when the user has decided to apply section 280.
usage: python3 apply_280.py <library.yaml>"""
import re
import sys
import textwrap

import yaml

path = sys.argv[1]
here = __file__.rsplit('/scripts/', 1)[0]
edits = yaml.safe_load(open(f"{here}/tables/yaml_edits_280.txt"))
text = open(path, encoding='utf-8').read()
if "# Section 280" in text:
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
    body = textwrap.wrap(f"Section 280: {' '.join(why.split())}", width=94)
    return [f"    # {l}" for l in body]


def apply(name, keys, why, after="max_intake_rate"):
    i, j = block(name)
    seg = lines[i:j]
    first = None
    for key, value in keys.items():
        hit = [n for n, l in enumerate(seg) if l.startswith(f"    {key}:")]
        if key == "depth_access_table":
            assert hit, (name, key)
            n = hit[0]
            m = n + 1
            while m < len(seg) and seg[m].startswith("      -"):
                m += 1
            rows = [f"      - {{depth_m: {r['depth_m']}, factor: {fmt(float(r['factor']))}}}" for r in value]
            seg[n + 1:m] = rows
        elif hit:
            seg[hit[0]] = f"    {key}: {fmt(value)}"
            n = hit[0]
        else:
            anchor = [k for k, l in enumerate(seg) if l.startswith(f"    {after}:")]
            assert anchor, (name, key)
            n = anchor[0] + 1
            seg.insert(n, f"    {key}: {fmt(value)}")
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

n_comment = 0
for c in edits.get("comment_lines", []):
    hits = [n for n, l in enumerate(lines) if l == c["old"]]
    assert hits, c["old"]
    for n in reversed(hits):
        lines[n:n + 1] = list(c["new"])
        n_comment += 1

open(path, 'w', encoding='utf-8').write('\n'.join(lines))
print(f"applied {len(edits['pairs'])} pairs, {n_comment} comment line(s) -> {path}")
