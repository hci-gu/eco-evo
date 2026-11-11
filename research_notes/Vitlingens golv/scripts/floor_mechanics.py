"""Whiting's hide floor: engine facts, size of the cut, and two bounded candidate refuges.

Usage (from this directory):
    python3 -I floor_mechanics.py <repo> > floor_mechanics_stdout.txt

Reads only repository files (fgconfig/fg_library.yaml and the Arsungarnas andel length table);
no downloads, no model run, nothing fitted to model output.

A. Library facts: whiting's species floor, every checked pair on whiting, whether it carries a pair
   visibility_floor (the engine reads a pair floor only on a checked predation pair; otherwise the
   prey's species floor applies to predation AND to every observer's view).
B. The cut a floor makes: visible = B (1 - rest (1 - floor)); at the fix97 annual-mean rest share 0.066
   (research_notes/"Gom och sinnen" section 4) and at full rest.
C. Jellyfish shelter bound: the box biomass share of whiting small enough to shelter under Cyanea
   (Dahl 1961: 3-9 cm in July; settlement at 8.5 cm, Bastrikin et al. 2014) from the box GOV length
   table (q-corrected, research_notes/"Arsungarnas andel" tables/stage_length_biomass.csv), and the
   cell-level floor it could at most justify: f_eff = 1 - s_small x share_associated x (1 - f_res).
D. Stationary prey: the Gerritsen-Strickler mean relative speed used by every fish-pair derivation,
   v_rel(v, u) = (u^2 + 3 v^2) / (3 v) for v >= u, with the prey cruising at u = 0.10 m/s (the a's
   assumption) vs motionless (u = 0) - the encounter factor a resting, motionless whiting would get.
"""
import csv
import sys
from pathlib import Path

import yaml

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../..")
LIB = REPO / "fgconfig" / "fg_library.yaml"
LEN = REPO / "research_notes" / (chr(0xC5) + "rsungarnas andel") / "tables" / "stage_length_biomass.csv"

lib = yaml.safe_load(LIB.read_text())


def find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


# ------------------------------------------------------------------ A
print("A. Library facts (fg_library.yaml)")
wh = find(lib, "whiting")
print(f"  whiting species visibility_floor = {wh.get('visibility_floor')}")
pairs = {}


def collect(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(k, str) and k.endswith("_preys_on_whiting") and isinstance(v, dict):
                pairs[k] = v
            collect(v)


collect(lib)
for k in sorted(pairs):
    v = pairs[k]
    if not v.get("preys_on"):
        continue
    print(f"  {k:40s} M2 {v.get('predation_mortality')}  a {v.get('max_intake_rate')}  "
          f"stage_access {v.get('stage_access')}  dark_ratio {v.get('dark_ratio')}  "
          f"pair visibility_floor {v.get('visibility_floor', '- (species floor applies)')}")

# ------------------------------------------------------------------ B
print("\nB. Cut on predation (this tick) and on observation (next tick): rest x (1 - floor)")
print("   rest 0.066 = fix97 annual mean over training worlds; rest 1 = a fully hidden cell")
print(f"  {'floor':>6s} {'cut at 0.066':>13s} {'cut at 1.0':>11s}")
for f in (0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0):
    print(f"  {f:6.2f} {0.066 * (1 - f):13.4f} {1.0 * (1 - f):11.2f}")

# ------------------------------------------------------------------ C
print("\nC. Jellyfish-shelter bound from the box GOV length table (whiting, q-corrected kg/h by cm)")
rows = [r for r in csv.DictReader(open(LEN)) if r["species"] == "whiting"]
by = {}
for r in rows:
    key = (r["period"], int(r["quarter"]))
    by.setdefault(key, []).append((int(r["L_cm"]), float(r["J"]), float(r["A"])))
for key in sorted(by):
    t = by[key]
    tot = sum(j + a for _, j, a in t)
    jt = sum(j for _, j, _ in t)
    if tot <= 0:
        continue
    out = []
    for cut in (6, 9):
        s = sum(j + a for L, j, a in t if L <= cut)
        out.append(f"<= {cut} cm: {s / tot:.4f} of B, {s / jt if jt else float('nan'):.4f} of J")
    print(f"  {key[0]} Q{key[1]}: J/B {jt / tot:.3f}; " + "; ".join(out))

print("\n  Cell-level floor the shelter could at most justify, f_eff = 1 - s x assoc x (1 - f_res):")
print("  s = biomass share <= 9 cm in Q3 (the only quarter with such fish in the table);")
q3 = by[("2001-25", 3)]
tot = sum(j + a for _, j, a in q3)
s9 = sum(j + a for L, j, a in q3 if L <= 9) / tot
q3o = by.get(("1991-96", 3))
s9o = sum(j + a for L, j, a in q3o if L <= 9) / sum(j + a for _, j, a in q3o) if q3o else float("nan")
for s_lab, s in (("2001-25", s9), ("1991-96", s9o)):
    for assoc in (0.1, 0.5, 1.0):
        for fres in (0.0, 0.5):
            f_eff = 1 - s * assoc * (1 - fres)
            print(f"    {s_lab} s {s:.4f} assoc {assoc:.1f} f_res {fres:.1f}: f_eff Jun-Aug {f_eff:.4f}; "
                  f"annual equivalent (3 of 12 months) {1 - (1 - f_eff) * 3 / 12:.4f}")

# ------------------------------------------------------------------ D
print("\nD. Stationary prey: v_rel(v, 0) / v_rel(v, 0.10)")


def vrel(v, u):
    return (u * u + 3 * v * v) / (3 * v) if v >= u else (v * v + 3 * u * u) / (3 * u)


U = 0.10
cases = [
    ("saithe (WC 0.77 of a), 1.0 BL/s, L 0.45 m, day", 1.0 * 0.45),
    ("saithe, 1.0 BL/s, L 0.70 m, day", 1.0 * 0.70),
    ("saithe, night 0.6 x day, L 0.45 m", 0.6 * 1.0 * 0.45),
    ("saithe, low 0.5 BL/s x night 0.3, L 0.45 m", 0.5 * 0.3 * 0.45),
    ("cod (benthivores), 0.4 BL/s, L 0.30 m", 0.4 * 0.30),
    ("cod, 0.4 BL/s, L 0.50 m", 0.4 * 0.50),
    ("cod, low 0.2 BL/s, L 0.40 m", 0.2 * 0.40),
]
for lab, v in cases:
    f = vrel(v, 0.0) / vrel(v, U)
    print(f"  {lab:45s} v {v:.3f} m/s: v_rel {vrel(v, U):.3f} -> {vrel(v, 0.0):.3f}, factor {f:.3f}")
print("  (detection of a motionless prey by vision or lateral line is not in this factor: unquantified)")
