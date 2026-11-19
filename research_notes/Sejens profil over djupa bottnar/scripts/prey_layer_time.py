"""Share of saithe's search time inside each prey's layer (the prey's support, phi > 0), per bed class, day and
night, under the library profile (exp 15 m) and the depth-scaled profile (expHsz: Hs = 15 m x (bed + 10)/150).

Usage: python3 -I prey_layer_time.py <repo_root> <scratch_dir>

research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py exec'd unchanged up to its section 0 block; prey profiles
are its prey_prof (and the krill machinery's krill_profile), as in the pair derivations. Geometry only.
"""
import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
BP = os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "scripts", "band_pairs.py")
src = open(BP).read()
head = src[:src.index("# ================================================================== 0. overlap geometry")]
ARGV = list(sys.argv)
sys.argv = [BP, ROOT, SCR, os.path.join(SCR, "_unused_band.json")]
B = {"__name__": "band_pairs_head", "__file__": BP}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, BP, "exec"), B)
sys.argv = ARGV
K, qk0, BEDS, WCELL, CLS_IDX = B["K"], B["qk0"], B["BEDS"], B["WCELL"], B["CLS_IDX"]
expbed, uniform = K["expbed"], K["uniform"]


def saithe(kind):
    def f(bed, q, night):
        s = q["Hs_saithe"] * ((bed + 10.0) / 150.0 if kind == "expHsz" else 1.0)
        if not night:
            return expbed(bed, s)
        return q["f_up_saithe"] * uniform(0.0, min(30.0, bed - 1.0)) + (1 - q["f_up_saithe"]) * expbed(bed, s * q["Hn_ratio"])
    return f


PREYS = ["krill", "herring", "pout", "blue whiting", "juvenile benthivores", "whiting"]
for kind in ("exp15", "expHsz"):
    pf = saithe(kind)
    print(f"\n{kind}: saithe search-time share inside each prey's layer, by class < 100 / 100-150 / 150-250 / >= 250 m")
    for night in (False, True):
        print(f"  {'night' if night else 'day'}")
        for p in PREYS:
            vals = []
            for idx in CLS_IDX:
                acc = 0.0
                for b, w in zip(BEDS[idx], WCELL[idx]):
                    f = pf(b, qk0, night)
                    ph = K["krill_profile"](b, qk0, night) if p == "krill" else B["prey_prof"](p, qk0)(b, qk0, night)
                    acc += w * float(f[ph > 0].sum())
                vals.append(acc / WCELL[idx].sum())
            print(f"    {p:22s} " + " / ".join(f"{v:.3f}" for v in vals))
