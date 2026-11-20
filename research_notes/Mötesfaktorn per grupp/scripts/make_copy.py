"""Write a COPY of fgconfig/fg_library.yaml with the derived factors on the anchored pairs' max_intake_rate
(text edit inside each pair block, comments kept). Never touches the live library.

    python3 make_copy.py OUT.yaml ARM

ARM a: planktivores -> zooplankton at the R ~ L reading of V1 (x0.36 with the summer temperature);
ARM b: the same with the size-free reading (x0.042). Every other factor is the central value of
tables/anchors_stdout.txt; pairs without an anchor are left as they are.
"""
import re, sys
SRC = "/home/simon/eco-evo/fgconfig/fg_library.yaml"
out, arm = sys.argv[1], sys.argv[2]
F = {
    "whiting_preys_on_krill": 0.0099,                # A: geometric mean of the six site rows at D_lib
    "planktivorous_fish_preys_on_krill": 0.0124,     # A: geometric mean of the four rows at D_lib
    "whiting_preys_on_planktivorous_fish": 0.164,    # B: Rindorf clupeid slope vs the bed-class mix
    "whiting_preys_on_benthivorous_fish": 0.012,     # B: Rindorf gadoid slope vs the bed-class mix
    "planktivorous_fish_preys_on_zooplankton": {"a": 0.36, "b": 0.042}[arm],   # C: V1 reading x summer q
}
lines = open(SRC).read().split("\n")
cur, done = None, {}
for i, ln in enumerate(lines):
    m = re.match(r"^  ([a-z_]+):\s*$", ln)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"^(    max_intake_rate: )([0-9.eE+-]+)(.*)$", ln)
    if m and cur in F and cur not in done:
        old = float(m.group(2))
        new = old * F[cur]
        lines[i] = f"{m.group(1)}{new:.6g}{m.group(3)}  # x{F[cur]} (Motesfaktorn per grupp, copy only)"
        done[cur] = (old, new)
assert set(done) == set(F), set(F) - set(done)
open(out, "w").write("\n".join(lines))
for k, (o, n) in done.items():
    print(f"{k}: {o} -> {n:.6g}")
