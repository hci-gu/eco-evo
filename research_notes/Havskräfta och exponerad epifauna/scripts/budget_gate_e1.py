"""Run tools/probes/budget_gate.py unchanged against a library COPY and a manifest COPY.

Usage (from the repository root):
  python3 "research_notes/Havskräfta och exponerad epifauna/scripts/budget_gate_e1.py" <library.yaml> <manifest.yaml>

The manifest copy's relative map paths are rewritten to absolute paths (repository root) in a
temporary file next to the manifest copy's directory is NOT used: the temporary file goes to $TMPDIR.
train._ProbeEnvBuilder is wrapped so the probe builds its env from the two copies; the probe's code is
executed with runpy, not copied (as research_notes/"Bentivorernas energi och diet"/scripts/budget_gate_lib.py).
"""
import os
import re
import runpy
import sys
import tempfile

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
library = os.path.abspath(sys.argv[1])
text = open(sys.argv[2], encoding="utf-8-sig").read()
text = re.sub(r"(:\s*)((?:maps/)?[\w.-]+\.npz)", lambda m: m.group(1) + os.path.join(ROOT, m.group(2)), text)
fd, manifest = tempfile.mkstemp(suffix=".yaml")
os.write(fd, text.encode("utf-8"))
os.close(fd)

import train  # noqa: E402

_orig = train._ProbeEnvBuilder


def _builder(*args, **kwargs):
    kwargs["project_path"] = manifest
    kwargs["library_path"] = library
    return _orig(*args, **kwargs)


train._ProbeEnvBuilder = _builder
print(f"[budget_gate on {library} with {sys.argv[2]}]")
try:
    runpy.run_path(os.path.join(ROOT, "tools", "probes", "budget_gate.py"), run_name="__main__")
finally:
    os.remove(manifest)
