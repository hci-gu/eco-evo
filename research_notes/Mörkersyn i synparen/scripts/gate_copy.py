"""tools/probes/budget_gate.py on a library COPY (pattern of research_notes/"Krillparen under fjordljus"/scripts/
gate_pinned.py): budget_gate runs unchanged except for its library (the probe builder's library_path) and grid.

Usage (any cwd): python3 gate_copy.py <repo_root> <library.yaml> [cells per side, default 16]
"""
import contextlib
import io
import os
import sys

ROOT, LIBP = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 16
os.chdir(ROOT)
path = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
src = open(path).read()
assert src.count("\nH = W = 16\n") == 1 and src.count("    apply_natural_mortality=True, migration=True)") == 1
src = src.replace("\nH = W = 16\n", f"\nH = W = {N}\n")
src = src.replace("    apply_natural_mortality=True, migration=True)",
                  f"    apply_natural_mortality=True, migration=True, library_path={LIBP!r})")
i, j = src.index("_ROOT = "), src.index("sys.path.insert(0, _ROOT)")
src = src[:i] + f"_ROOT = {ROOT!r}\n" + src[j:]
G = {"__name__": "__main__", "__file__": path}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, path, "exec"), G)
env = G["env"]
i = env.dm_ids.index("predatory_fish")
print(f"== budget_gate on {os.path.basename(LIBP)}; grid {N}x{N}")
for prey in ("krill", "planktivorous_fish"):
    j = env.global_fg_order.index(prey)
    print(f"   engine a(predatory_fish -> {prey}) {float(env.max_intake_mat[i, j]):.6g}")
print(buf.getvalue())
