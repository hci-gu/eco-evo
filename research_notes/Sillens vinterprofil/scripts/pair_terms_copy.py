"""Share of each in-layer-derived fish pair's attack rate a that comes from each (prey species, period) term,
under the library's current form (section 269: vision rule A on the band overlap, "A_band").

Usage: python3 -I pair_terms.py <repo_root> <scratch_dir> <out_json>

Nothing is fitted; no model output is read; nothing in the repository is written.

Method: research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py is exec'd up to its run block (it in turn
execs "En synregel for fiskparen"/scripts/one_rule_pairs.py), so every factor is the one the library a came from.
Its band_fish_pair() is copied with ONE textual change: the per-member contribution key (r.sp, r.st) becomes the
term key (t["prey"], t["night"]). The pair total (eq) is checked against the unmodified function, and the
light-weighted effective detection radius of each term (sqrt of the band-mean pi R^2 / pi) is reported, which is
the R_i the schooling ratio S needs (scripts/school_factor.py).
"""
import inspect
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = os.path.abspath(sys.argv[3])
os.makedirs(SCR, exist_ok=True)
BP = os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "scripts", "band_pairs.py")
src = open(BP).read()
MARK = "# ================================================================== 0. overlap geometry"
assert src.count(MARK) == 1
head = src[:src.index(MARK)]
save = list(sys.argv)
sys.argv = [BP, ROOT, SCR, os.path.join(SCR, "_unused_band.json")]
G = {"__name__": "band_pairs_head", "__file__": BP}
exec(compile(head, BP, "exec"), G)
sys.argv = save

fsrc = inspect.getsource(G["band_fish_pair"]) if False else None
# inspect cannot see exec'd source; cut the function text out of the file instead
start = head.index("def band_fish_pair(")
end = head.index("\n\n\n", start)
fsrc = head[start:end]
OLD = 'contrib[(r.sp, r.st)] = contrib.get((r.sp, r.st), 0.0) + float((w * k * val @ WCELL).mean()) * rows.mean()'
assert fsrc.count(OLD) == 1
NEW = ('key = (t["prey"], bool(t["night"]))\n'
       '                contrib[key] = contrib.get(key, 0.0) + float((w * k * val @ WCELL).mean()) * rows.mean()\n'
       '                if not clip or prey not in SLAB:\n'
       '                    o = (pp * ph).sum(axis=0)\n'
       '                    RW[key] = RW.get(key, 0.0) + float(((w * k * val).mean(axis=0) @ WCELL))\n'
       '                    OW[key] = OW.get(key, 0.0) + float(((w * k * np.pi * np.broadcast_to(o, val.shape)).mean(axis=0) @ WCELL))')
fsrc2 = fsrc.replace(OLD, NEW).replace("def band_fish_pair(", "def band_terms(")
fsrc2 = fsrc2.replace("members={f\"{k[0]}|{k[1]}\": v / tot for k, v in contrib.items() if v / tot >= 0.005})",
                      "members={f\"{k[0]}|{'night' if k[1] else 'day'}\": v / tot for k, v in contrib.items()},\n"
                      "                reff={f\"{k[0]}|{'night' if k[1] else 'day'}\": math.sqrt(RW[k] / OW[k]) for k in RW if OW[k] > 0})")
assert "band_terms" in fsrc2 and "reff=" in fsrc2
G["RW"], G["OW"] = {}, {}
exec(compile(fsrc2, "band_terms", "exec"), G)

PAIRS = ["predatory_fish->planktivorous_fish", "predatory_fish->benthivorous_fish", "predatory_fish->whiting",
         "predatory_fish->mackerel", "whiting->planktivorous_fish", "whiting->benthivorous_fish",
         "mackerel->planktivorous_fish"]
q = dict(G["qk0"])
# same state as band_pairs.py's "A_band" variant (its run block): krill machinery run once under each rule
K = G["K"]
G["RULE"].update(rule="lib", RD=1.2, Lk=None)
G["run_krill"](dict(K["level"](1)))
G["RULE"].update(rule="A", RD=1.2, Lk=None)
G["run_krill"](dict(K["level"](1), Ke_gad=0.02, k_deep=0.06))
BANDJ = json.load(open(os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "tables", "band_pairs.json")))
out = {}
print("Term shares of the A_band (library, section 269) equal-cell annual-mean a, by prey species and period")
print("(period = the sun-threshold day/night switch of the derivation). R_eff = light-weighted detection radius (m).")
for name in PAIRS:
    G["RW"].clear()
    G["OW"].clear()
    ref = G["band_fish_pair"](name, q, "A")
    res = G["band_terms"](name, q, "A")
    assert abs(ref["eq"] - res["eq"]) <= 1e-12 + 1e-9 * abs(ref["eq"]), (name, ref["eq"], res["eq"])
    shares = dict(sorted(res["members"].items(), key=lambda kv: -kv[1]))
    ref_eq = BANDJ["variants"]["A_band"]["fish"][name]["eq"]
    out[name] = dict(eq=res["eq"], eq_band_pairs_json=ref_eq, nd=res["nd"], shares=shares, reff=res["reff"])
    print(f"\n  {name}: a_eq {res['eq']:.5g} (band_pairs.json A_band {ref_eq:.5g}; section 220 base {res['a_lib']}),"
          f" night/day {res['nd']:.3f}")
    for k, v in shares.items():
        r = res["reff"].get(k)
        print(f"      {k:28s} share {v:6.3f}" + (f"   R_eff {r:5.2f} m" if r is not None else "   (bed slab term)"))
json.dump(out, open(OUTJ, "w"), indent=1)
print(f"\nwritten {OUTJ}")
