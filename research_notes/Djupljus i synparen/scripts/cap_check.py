"""Does the 2 L cap still bind on the krill pairs at K_deep 0.09? Rule A against the library rule (R_sat from
krill_pairs_wc, uncapped R0 on krill vs cap 2 L) at the SAME light, per predator group: a ratio of 1.000 means
R_sat = 2 L binds for every member (rule A cannot differ from the library rule), so the pair moves only through
the light. Executes research_notes/"En synregel for fiskparen"/scripts/one_rule_pairs.py up to its run block.

Usage: python3 cap_check.py <repo_root> <scratch_dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import run_patched  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
ORP = os.path.join(ROOT, "research_notes", "En synregel for fiskparen", "scripts", "one_rule_pairs.py")
O = run_patched(ORP, [], [ORP, ROOT, SCR, os.path.join(SCR, "_unused.json")], name="orp_head",
                upto="# =============================================================== run", quiet=True)
K = O["K"]
res = {}
for kd in (0.06, 0.09):
    for rule in ("lib", "A"):
        O["RULE"].update(rule=rule, RD=1.2, Lk=None)
        res[(kd, rule)] = O["run_krill"](dict(K["level"](1), Ke_gad=0.02, k_deep=kd))
print("krill pairs: equal-cell a (library units of the reproduction), rule A / library rule at the same K_deep")
print(f"  {'group':20s} {'lib 0.06':>9s} {'A 0.06':>9s} {'A/lib':>6s} {'lib 0.09':>9s} {'A 0.09':>9s} {'A/lib':>6s}"
      f" {'A 0.09 / A 0.06':>16s}")
for g in res[(0.06, "lib")]:
    l6, a6 = res[(0.06, "lib")][g]["eq"], res[(0.06, "A")][g]["eq"]
    l9, a9 = res[(0.09, "lib")][g]["eq"], res[(0.09, "A")][g]["eq"]
    print(f"  {g:20s} {l6:9.4g} {a6:9.4g} {a6 / l6:6.3f} {l9:9.4g} {a9:9.4g} {a9 / l9:6.3f} {a9 / a6:16.3f}")
