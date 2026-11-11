"""Does the krill : fish encounter ratio depend on saithe size in the library pairs and in one convention?

Usage: python3 -I member_size.py <repo_root> <wc_terms.json>

V_lib per member tonne: krill class a of the member (wc_terms krill_members) against the member's own in-layer
fish a's (wc_terms parts, per whole-group tonne, divided by the member's group weight W). V_one: meas_pairs.py
(exec'd, 262 layer patched, as gate_options.py) run with the predator group reduced to one member at W = 1.
Encounter shares at class 150-250 m and at the saithe spread, prey densities as gate_options.py.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
TERMS = json.load(open(sys.argv[2]))
ARGV0 = list(sys.argv)
MP = os.path.join(ROOT, "research_notes", "Vattenpelarens par i uppmätt ljus", "scripts", "meas_pairs.py")
src = open(MP).read()
head = src[:src.index("\n# ================================================================ central\nqc = level(1)")]
sys.argv = [MP, ROOT, os.devnull, "10"]
M = {"__name__": "meas_pairs_head", "__file__": MP}
exec(compile(head, MP, "exec"), M)
sys.argv = ARGV0
G = M["G"]
AB, AZ = [150.0, 160.0, 170.0, 250.0, 350.0], [95.0, 140.0, 140.0, 120.0, 145.0]


def new_layer(bed, q):
    old_zc = q["zc"] + q["zc_slope"] * np.clip(bed - 150.0, 0, 200) / 100.0
    zc = old_zc + float(np.interp(bed, AB, AZ)) - (95.0 + 25.0 * np.clip(bed - 150.0, 0, 200) / 100.0)
    H = q["Hk"] * (1.0 + (2.0 / 3.0) * np.clip((bed - 150.0) / 100.0, 0, 1))
    bot = min(zc + H / 2, bed - 1.0)
    top = max(bot - H, q["ztop_min"])
    if bot - top < 5.0:
        top = bot - 5.0
    return max(top, 0.0), bot


G["krill_day_layer"] = new_layer
BEDS, WCELL, CLS_IDX, MONTH = M["BEDS"], M["WCELL"], M["CLS_IDX"], M["MONTH"]
WT = np.where(np.isin(MONTH, [4, 5, 6, 7]), 0.05, 0.43)


def cls_a(a, krill):
    out = []
    for k, idx in enumerate(CLS_IDX):
        s = a[:, idx] @ WCELL[idx] / WCELL[idx].sum()
        out.append(float((s * WT).sum() / WT.sum()) if (k == 0 and krill) else float(s.mean()))
    return np.array(out)


B = {"krill": np.array([2.08, 5.02, 6.11, 7.12]), "planktivorous_fish": 1.776, "benthivorous_fish": 0.254,
     "whiting": 0.060}
SAI = np.array([0.043, 0.127, 0.782, 0.048])
NC = np.array([0.232, 0.299, 0.324, 0.146])


def shares(A):
    out = {}
    for lab, w in (("150-250 m", np.eye(4)[2] / NC[2] * NC), ("saithe spread", SAI)):
        e = {p: float((w * np.asarray(A[p]) * B[p]).sum()) for p in B}
        out[lab] = e["krill"] / sum(e.values())
    return out


fish = TERMS["fish"]
km = TERMS["krill_members"]
print("Krill share of the encounter (krill, planktivores, benthivores, whiting only) per saithe member")
for mname, key, W in (("saithe < 60 cm", "saithe|adult", 0.797 * 0.632), ("saithe >= 60 cm", "saithe|large", 0.797 * 0.368)):
    A = {"krill": [c["a"] for c in km[mname]["classes"]],
         "planktivorous_fish": np.full(4, fish["parts_plk"][key] / W * 1.0),
         "benthivorous_fish": np.full(4, fish["parts_ben"][key] / W),
         "whiting": np.full(4, fish["parts_whi"][key] / W)}
    # parts are per whole rest-group tonne; the rest group W_rest = 1 - mackerel group in pred_encounter weights,
    # the same divisor for every prey, so it cancels in the shares.
    s_lib = shares(A)
    mem = [m for m in G["GROUPS"]["predatory_fish"]["members"] if m["name"] == mname][0]
    keep = G["GROUPS"]["predatory_fish"]["members"]
    G["GROUPS"]["predatory_fish"]["members"] = [dict(mem, W=1.0)]
    fa, _ = M["prey_a"](M["level"](1), "meas")
    pa = M["pairs_from"](fa)
    G["GROUPS"]["predatory_fish"]["members"] = keep
    A1 = {"krill": cls_a(pa["krill"], True), "planktivorous_fish": cls_a(pa["planktivorous_fish"], False),
          "benthivorous_fish": cls_a(pa["benthivorous_fish"], False), "whiting": cls_a(pa["whiting"], False)}
    s_one = shares(A1)
    print(f"  {mname:16s} L {mem['L']:.2f} m: V_lib 150-250 {s_lib['150-250 m']:.3f}, saithe spread {s_lib['saithe spread']:.3f}"
          f" | V_one 150-250 {s_one['150-250 m']:.3f}, saithe spread {s_one['saithe spread']:.3f}")
    print(f"      V_one class a krill " + "/".join(f"{x:.4g}" for x in A1["krill"]) + "; planktivores " +
          "/".join(f"{x:.4g}" for x in A1["planktivorous_fish"]))
