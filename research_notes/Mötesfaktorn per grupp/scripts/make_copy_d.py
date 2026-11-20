"""Copy d: ONE factor per predator on every one of its pairs' max_intake_rate, from the anchors that measure a
without the field ration (clearance or density-gradient slope; Krillsidan verdict 2):
  planktivores 0.36  (Arrhenius & Hansson 1999 V1, R ~ L, x summer q; tables/anchors_stdout.txt)
  whiting      0.16  (Rindorf & Gislason 2005 clupeid slope at Rindorf's bed mix)   [arm d2: 0.30, equal cells]
  water column 0.16  (whiting's fish slope transferred [J])                         [arm d2: 0.30]
Sharing within a predator is [J]. Mackerel (no anchor; its zooplankton pair is a measured clearance) and the
benthivores (294.4) are left as they are. Never touches the live library.
    python3 make_copy_d.py OUT.yaml d|d2
"""
import re, sys
SRC = "/home/simon/eco-evo/fgconfig/fg_library.yaml"
out, arm = sys.argv[1], sys.argv[2]
fish = {"d": 0.16, "d2": 0.30}[arm]
F = {"planktivorous_fish": 0.36, "whiting": fish, "predatory_fish": fish}
lines = open(SRC).read().split("\n")
cur, done = None, {}
for i, ln in enumerate(lines):
    m = re.match(r"^  ([a-z_]+):\s*$", ln)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"^(    max_intake_rate: )([0-9.eE+-]+)(.*)$", ln)
    if m and cur and cur not in done:
        g = next((g for g in F if cur.startswith(g + "_preys_on_")), None)
        if g:
            old = float(m.group(2)); new = old * F[g]
            lines[i] = f"{m.group(1)}{new:.6g}{m.group(3)}  # x{F[g]} (copy {arm} only)"
            done[cur] = (old, new)
open(out, "w").write("\n".join(lines))
for k, (o, n) in sorted(done.items()):
    print(f"{k}: {o} -> {n:.6g}")
