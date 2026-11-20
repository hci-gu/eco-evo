"""Run tools/probes/budget_gate.py against a library COPY (the loader's library resolution patched; the
live fgconfig/fg_library.yaml is not touched).

    python3 gate_on_copy.py COPY.yaml > ../tables/budget_gate_copy_X.txt
"""
import os, runpy, sys
ROOT = "/home/simon/eco-evo"
COPY = os.path.abspath(sys.argv[1])
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import lib.config.config_loader as cl  # noqa: E402
cl.resolve_library_path = lambda *a, **k: COPY
cl.DEFAULT_LIBRARY_PATH = COPY
print(f"library copy: {COPY}")
sys.argv = ["tools/probes/budget_gate.py"]
runpy.run_path("tools/probes/budget_gate.py", run_name="__main__")
