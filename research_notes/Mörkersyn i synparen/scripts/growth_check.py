"""Derived growth_rate on the live library and on each copy (lib.config.config_loader.load_project_config,
mareld2.yaml, seed 1; the research_notes/"Djupljus i synparen"/scripts/gate_280.py loader check). growth_budget
derives g from r_max, M1, the bookings and the reserve terms; a, dark_ratio, stage_access and the depth tables are
not among its inputs, so no g should move.

Usage: python3 growth_check.py <repo_root> <library copy.yaml> [...]
"""
import contextlib
import io
import os
import sys
import warnings

ROOT = os.path.abspath(sys.argv[1])
COPIES = [os.path.abspath(p) for p in sys.argv[2:]]
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from lib.config.config_loader import load_project_config  # noqa: E402


def growth(lp):
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        res = load_project_config(os.path.join(ROOT, "mareld2.yaml"), library_path=lp, seed=1)
    fgs = res[0] if isinstance(res, tuple) else res
    if not isinstance(fgs, dict):
        fgs = {getattr(f, "id", getattr(f, "name", str(i))): f for i, f in enumerate(fgs)}
    out = {}
    for k, f in fgs.items():
        g = getattr(f, "params", {}).get("growth_rate", None) if hasattr(f, "params") else None
        if g is None:
            g = getattr(f, "growth_rate", None)
        out[str(k)] = g
    return out, [str(x.message) for x in w]


g_live, w_live = growth(os.path.join(ROOT, "fgconfig", "fg_library.yaml"))
for cp in COPIES:
    g_new, w_new = growth(cp)
    moved = [k for k in g_live if g_live[k] != g_new.get(k)]
    print(f"{os.path.basename(cp)}: {len(g_live)} groups, {len(moved)} derived growth_rate(s) moved"
          + (f" ({', '.join(moved)})" if moved else "") + f"; loader warnings live {len(w_live)}, copy {len(w_new)}")
    for k in g_live:
        print(f"  {k:26s} {g_live[k]} -> {g_new.get(k)}{'' if k not in moved else '  CHANGED'}")
