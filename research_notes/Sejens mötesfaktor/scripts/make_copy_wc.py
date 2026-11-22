"""Library COPY with the water column's (predatory_fish) pair a's at factor F instead of section 296's 0.16:
each predatory_fish_preys_on_* max_intake_rate x F / 0.16 (the live values already carry x0.16).
--only PREY restricts the change to predatory_fish_preys_on_PREY. Never touches the live library.
    python3 make_copy_wc.py OUT.yaml F [--only planktivorous_fish]
"""
import re, sys
SRC = "/home/simon/eco-evo/fgconfig/fg_library.yaml"
out, F = sys.argv[1], float(sys.argv[2])
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
lines = open(SRC).read().split("\n")
cur, done = None, {}
for i, ln in enumerate(lines):
    m = re.match(r"^  ([a-z_]+):\s*$", ln)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"^(    max_intake_rate: )([0-9.eE+-]+)(.*)$", ln)
    if m and cur and cur not in done and cur.startswith("predatory_fish_preys_on_"):
        if only and cur != f"predatory_fish_preys_on_{only}":
            continue
        old = float(m.group(2)); new = old * F / 0.16
        lines[i] = f"{m.group(1)}{new:.6g}{m.group(3)}  # x{F}/0.16 (copy only)"
        done[cur] = (old, new)
open(out, "w").write("\n".join(lines))
for k, (o, n) in sorted(done.items()):
    print(f"{k}: {o} -> {n:.6g}")
