"""Apply tables/yaml_edits.txt to fgconfig/fg_library.yaml as text (keeps every comment).

Per pair: a '# Section 269:' comment with the derivation line before max_intake_rate; max_intake_rate,
dark_ratio (replaced or inserted after max_intake_rate), stage_access (replaced) and depth_access_table
(replaced or appended at the end of the pair's keys). usage: python3 apply_269.py <repo>"""
import re, sys
import yaml

repo = sys.argv[1]
here = __file__.rsplit('/scripts/', 1)[0]
edits_txt = open(f"{here}/tables/yaml_edits.txt").read()
edits = yaml.safe_load(edits_txt)
# the first comment line under each pair is its derivation
why = {}
for m in re.finditer(r"^  ([a-z_]+):\n    # (.+)$", edits_txt, re.M):
    why[m.group(1)] = m.group(2)

path = f"{repo}/fgconfig/fg_library.yaml"
lines = open(path).read().split('\n')


def block(name):
    i = lines.index(f"  {name}:")
    j = i + 1
    while j < len(lines) and not re.match(r'^  [a-z_]+:$', lines[j]) and not re.match(r'^[a-z_]+:', lines[j]):
        j += 1
    return i, j


def fmt(v):
    return repr(v) if not isinstance(v, float) else (f"{v:g}" if abs(v) < 1e-3 and v != 0 else repr(v))


for name, new in edits.items():
    i, j = block(name)
    seg = lines[i:j]
    # drop an existing depth_access_table and its rows
    out, skip = [], False
    for l in seg:
        if l.startswith('    depth_access_table:'):
            skip = True
            continue
        if skip and l.startswith('      -'):
            continue
        skip = False
        out.append(l)
    seg = out
    k = next(n for n, l in enumerate(seg) if l.startswith('    max_intake_rate:'))
    seg[k] = f"    max_intake_rate: {fmt(new['max_intake_rate'])}"
    seg.insert(k, f"    # Section 269 (rule A + band overlap; research_notes/\"Bottenfiskens hojdprofil\"): {why[name]}")
    if 'dark_ratio' in new:
        d = [n for n, l in enumerate(seg) if l.startswith('    dark_ratio:')]
        if d:
            seg[d[0]] = f"    dark_ratio: {fmt(new['dark_ratio'])}"
        else:
            k = next(n for n, l in enumerate(seg) if l.startswith('    max_intake_rate:'))
            seg.insert(k + 1, f"    dark_ratio: {fmt(new['dark_ratio'])}")
    if 'stage_access' in new:
        s = [n for n, l in enumerate(seg) if l.startswith('    stage_access:')]
        assert s, name
        seg[s[0]] = f"    stage_access: [{new['stage_access'][0]}, {new['stage_access'][1]}]"
    if 'depth_access_table' in new:
        last = max(n for n, l in enumerate(seg) if l.startswith('    ') and not l.startswith('    #'))
        rows = ["    depth_access_table:"] + [
            f"      - {{depth_m: {r['depth_m']}, factor: {fmt(float(r['factor']))}}}" for r in new['depth_access_table']]
        seg[last + 1:last + 1] = rows
    lines[i:j] = seg
open(path, 'w').write('\n'.join(lines))
print(f"applied {len(edits)} pairs")
