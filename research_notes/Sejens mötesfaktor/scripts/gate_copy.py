"""tools/probes/budget_gate.py on a library COPY (pattern: research_notes/"Krillparen under fjordljus"/scripts/
gate_pinned.py): the probe builder gets library_path=COPY; prints the engine's water-column a's first so a silent
fallback to the live library would show. Usage (any cwd): python3 gate_copy.py <repo root> <COPY.yaml|live>
"""
import contextlib, io, os, sys
ROOT = os.path.abspath(sys.argv[1])
lib = sys.argv[2]
libpath = os.path.join(ROOT, "fgconfig", "fg_library.yaml") if lib == "live" else os.path.abspath(lib)
path = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
src = open(path).read()
key = "    apply_natural_mortality=True, migration=True)"
assert src.count(key) == 1
src = src.replace(key, f"    apply_natural_mortality=True, migration=True, library_path={libpath!r})")
i, j = src.index("_ROOT = "), src.index("sys.path.insert(0, _ROOT)")
src = src[:i] + f"_ROOT = {ROOT!r}\n" + src[j:]
os.chdir(ROOT)
G = {"__name__": "__main__", "__file__": path}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, path, "exec"), G)
env = G["env"]
print(f"== budget_gate on {libpath}")
i = env.dm_ids.index("predatory_fish")
for prey in ("planktivorous_fish", "benthivorous_fish", "whiting", "mackerel", "zooplankton", "krill"):
    j = env.global_fg_order.index(prey)
    print(f"   engine a predatory_fish -> {prey:20s} {float(env.max_intake_mat[i, j]):.6g}")
print(buf.getvalue())
