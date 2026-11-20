"""Copy c: ONE factor per predator on every one of its pairs' max_intake_rate, the factor at which the chain's
equal-cell encounter at T 1 reproduces the booked field feeding level (tables/booked_level_stdout.txt:
X_field / X_chain). A candidate for sizing only - it is calibrated on the booked ration (field diet x field Q/B),
so the booked ration is no longer an independent test for these groups. The benthivores are left out (294.4).
Never touches the live library.   python3 make_copy_c.py OUT.yaml
"""
import re, sys
SRC = "/home/simon/eco-evo/fgconfig/fg_library.yaml"
F = {"planktivorous_fish": 0.0260, "whiting": 0.0278, "predatory_fish": 0.0310, "mackerel": 0.0488}
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
            lines[i] = f"{m.group(1)}{new:.6g}{m.group(3)}  # x{F[g]} (Krillsidan copy c, copy only)"
            done[cur] = (old, new)
open(sys.argv[1], "w").write("\n".join(lines))
for k, (o, n) in sorted(done.items()):
    print(f"{k}: {o} -> {n:.6g}")
