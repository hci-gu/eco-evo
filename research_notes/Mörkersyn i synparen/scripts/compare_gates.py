"""Side-by-side of two budget_gate outputs (gate_copy.py): the equal-cell diet gate, the field-spread gate, the
prey side (implied / booked M2 per pair, and the field-spread reading), the OFF sets (goes OFF / comes back ON),
and the section 298 potential-ration block. Also the class gradient G of the policy-free intake (intake_by_class.py)
for each predator, if the intake files are given.

Usage: python3 compare_gates.py <gate live.txt> <gate copy.txt> [<intake dir> <copy tag>]
"""
import os
import re
import sys

A, B = (open(p).read() for p in sys.argv[1:3])
IDIR = sys.argv[3] if len(sys.argv) > 3 else None
TAG = sys.argv[4] if len(sys.argv) > 4 else None


def section(t, start, end=None):
    i = t.index(start)
    j = t.index(end, i) if end else len(t)
    return t[i:j]


def equal_cell(t):
    s = section(t, "=== diet gate: encounter-implied", "=== diet gate at the field spread")
    out, pred = {}, None
    for l in s.splitlines():
        m = re.match(r"^\s*(\w+)\s+prey\s+B\s+B_vis", l)
        if m:
            pred = m.group(1)
            continue
        m = re.match(r"^\s+(\w+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$", l)
        if m and pred:
            out[(pred, m.group(1))] = (float(m.group(4)), m.group(6).strip())
    return out


def field(t):
    s = section(t, "=== diet gate at the field spread", "=== prey-side check")
    out, pred = {}, None
    for l in s.splitlines():
        m = re.match(r"^\s*(\w+) \((\w+), GOV", l)
        if m:
            pred = m.group(1)
            continue
        m = re.match(r"^\s+(\w+) enc\s+([\d.]+) booked\s+([\d.]+) ratio\s+([\d.]+)\s+(\S+)", l)
        if m and pred:
            out[(pred, m.group(1))] = (float(m.group(4)), m.group(5))
    return out


def prey_side(t):
    s = section(t, "=== prey-side check", "=== total ration")
    out, prey, tot = {}, None, {}
    for l in s.splitlines():
        m = re.match(r"^\s+(\w+): booked M2 ([\d.e-]+) /yr, implied ([\d.e-]+) \(ratio ([\d.]+)\)", l)
        if m:
            prey = m.group(1)
            tot[prey] = float(m.group(4))
            continue
        m = re.match(r"^\s+(\w+) booked\s+([\d.e-]+) \(([\d.]+)\)\s+implied\s+([\d.e-]+) ratio\s+([\d.]+)\s+(ok|OFF)"
                     r"(?:\s+field\s+([\d.e-]+) \(\s*([\d.]+)\))?", l)
        if m and prey:
            out[(m.group(1), prey)] = (float(m.group(5)), m.group(6), float(m.group(8)) if m.group(8) else None)
    return out, tot


def offs(t):
    o = set()
    for l in t.splitlines():
        if l.startswith("Diet gate: OFF pairs") or l.startswith("Diet gate at the field spread: OFF pairs") \
                or l.startswith("Prey-side check: OFF pairs"):
            kind = l.split(":")[0]
            for part in l.split(" - ", 1)[1].split(";"):
                m = re.match(r"\s*(\w+) -> (\w+)", part)
                if m:
                    o.add((kind, m.group(1), m.group(2)))
    return o


def pot(t):
    s = section(t, "=== potential ration at the field spread")
    out = {}
    lines = s.splitlines()
    for k, l in enumerate(lines):
        m = re.match(r"^\s+(\w+) \(([^)]*)\) field Q/B ([\d.]+): potential ([\d.]+) = ([\d.]+) x\s+(.*?); months ([\d.]+)-([\d.]+) x;"
                     r" ceiling share ([\d.]+); krill share ([\d.]+)", l)
        if m:
            short = re.search(r"short in (\d+) of 12", m.group(6))
            mon = lines[k + 1].split(":")[1].split()
            out[m.group(1)] = dict(spread=m.group(2), x=float(m.group(5)), short=int(short.group(1)) if short else 0,
                                   lo=float(m.group(7)), hi=float(m.group(8)), ceil=float(m.group(9)),
                                   krill=float(m.group(10)), months=[float(x) for x in mon])
    return out


ea, eb = equal_cell(A), equal_cell(B)
fa, fb = field(A), field(B)
(pa, ta), (pb, tb) = prey_side(A), prey_side(B)
print("Equal-cell diet gate (encounter / booked share; verdict), live -> copy")
for k in ea:
    if k in eb and ea[k] != eb[k]:
        print(f"  {k[0]:20s} -> {k[1]:22s} {ea[k][0]:6.2f} {ea[k][1]:10s} -> {eb[k][0]:6.2f} {eb[k][1]}")
print("\nField-spread diet gate, live -> copy")
for k in fa:
    print(f"  {k[0]:20s} -> {k[1]:22s} {fa[k][0]:6.2f} {fa[k][1]:4s} -> {fb[k][0]:6.2f} {fb[k][1]}")
print("\nPrey side (implied / booked M2 per pair, equal cells; field spread in brackets), live -> copy")
for k in pa:
    a, b = pa[k], pb[k]
    if a != b:
        fs = f"  [{a[2]} -> {b[2]}]" if a[2] is not None else ""
        print(f"  {k[0]:20s} -> {k[1]:22s} {a[0]:6.2f} {a[1]:4s} -> {b[0]:6.2f} {b[1]}{fs}")
print("  prey totals (implied / booked): " + "; ".join(f"{p} {ta[p]:.2f} -> {tb[p]:.2f}" for p in ta if ta[p] != tb[p]))
oa, ob = offs(A), offs(B)
print("\nOFF in both: " + ("; ".join(f"{k[0]}: {k[1]} -> {k[2]}" for k in sorted(oa & ob)) or "none"))
print("Goes OFF:    " + ("; ".join(f"{k[0]}: {k[1]} -> {k[2]}" for k in sorted(ob - oa)) or "none"))
print("Comes back ON: " + ("; ".join(f"{k[0]}: {k[1]} -> {k[2]}" for k in sorted(oa - ob)) or "none"))
qa, qb = pot(A), pot(B)
print("\nSection 298 potential ration at the field spread (x field Q/B; short months; month range; ceiling share; krill"
      " share), live -> copy")
for g in qa:
    a, b = qa[g], qb[g]
    print(f"  {g:20s} ({a['spread']}) {a['x']:.2f} x, {a['short']} short, {a['lo']:.2f}-{a['hi']:.2f}, ceil {a['ceil']:.2f},"
          f" krill {a['krill']:.2f}  ->  {b['x']:.2f} x, {b['short']} short, {b['lo']:.2f}-{b['hi']:.2f}, ceil {b['ceil']:.2f},"
          f" krill {b['krill']:.2f}")
    print(f"  {'':20s} by month copy: " + " ".join(f"{x:.2f}" for x in b["months"]))

if IDIR:
    print("\nPolicy-free intake by bed class (intake_by_class.py, 60x60): G = I(< 150) / I(150-250), "
          "I(< 100) / I(150-250); I = share of the ceiling per class")
    for pred in ("predatory_fish", "whiting", "benthivorous_fish"):
        res = {}
        for tag in ("live", TAG):
            p = os.path.join(IDIR, f"{tag}_{pred}.txt")
            if not os.path.exists(p):
                continue
            rows = []
            t = open(p).read()
            for blk in t.split("\n== ")[1:]:
                mon = blk.split(":")[0]
                m = re.search(r"\n  library\s+I ([\d. ]+)\s+G\s+([\d.]+)\s+([\d.]+)", blk)
                rows.append((mon, [float(x) for x in m.group(1).split()], float(m.group(2)), float(m.group(3))))
            res[tag] = rows
        print(f"  {pred}")
        for (mon, Ia, ga, g0a), (_, Ib, gb, g0b) in zip(res["live"], res[TAG]):
            print(f"    {mon}: G {ga:.2f} ({g0a:.2f}) -> {gb:.2f} ({g0b:.2f});  I " + " / ".join(f"{x:.3f}" for x in Ia)
                  + "  ->  " + " / ".join(f"{x:.3f}" for x in Ib))
