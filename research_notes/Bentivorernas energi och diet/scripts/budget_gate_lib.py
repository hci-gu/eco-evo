"""Run tools/probes/budget_gate.py unchanged against a chosen library file.

Usage (from the repository root):
  python3 "research_notes/Bentivorernas energi och diet/scripts/budget_gate_lib.py" <library.yaml>

budget_gate.py has no library argument, but train._ProbeEnvBuilder accepts library_path; this
wrapper pre-binds it and executes the real probe with runpy, so the probe's code is not copied.
"""
import functools
import os
import runpy
import sys

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
library = os.path.abspath(sys.argv[1])

import train  # noqa: E402

train._ProbeEnvBuilder = functools.partial(train._ProbeEnvBuilder, library_path=library)
print(f"[budget_gate on {library}]")
runpy.run_path(os.path.join(ROOT, "tools", "probes", "budget_gate.py"), run_name="__main__")
