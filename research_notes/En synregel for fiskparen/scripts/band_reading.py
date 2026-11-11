"""Cross-check for the water column only: the one rule in the BAND reading (the sibling note's V_one convention:
band overlap on every pair, research_notes/"Vattenpelarens par i uppmatt ljus" (a-umlaut) meas_pairs.py), with
the rule's contrast (|C0| krill 0.4, fish 0.7) and fish floor 0.3 m. This is NOT the literal library-overlap
reading of one_rule_pairs.py; it shows how much of the result rides on the overlap convention.

Usage: python3 -I band_reading.py <repo_root> <out_json>

Gate arithmetic as research_notes/"Vattenpelarens par per djupklass"/scripts/gate_options.py (60 x 60 depth map,
equal cells / saithe GOV spread / per class; B_vis and booked shares as there; the water column's bookings did
not change in section 268).
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
OUTJ = sys.argv[2]
RN = os.path.join(ROOT, "research_notes")
ARGV0 = list(sys.argv)
MP = os.path.join(RN, "Vattenpelarens par i uppmätt ljus", "scripts", "meas_pairs.py")
src = open(MP).read()
head = src[:src.index("\n# ================================================================ central\nqc = level(1)")]
sys.argv = [MP, ROOT, os.devnull, "10"]
M = {"__name__": "meas_pairs_head", "__file__": MP}
exec(compile(head, MP, "exec"), M)
sys.argv = ARGV0
G = M["G"]
ANCHOR_B = [150.0, 160.0, 170.0, 250.0, 350.0]
ANCHOR_Z = [95.0, 140.0, 140.0, 120.0, 145.0]


def new_layer(bed, q):
    old_zc = q["zc"] + q["zc_slope"] * np.clip(bed - 150.0, 0, 200) / 100.0
    shift = float(np.interp(bed, ANCHOR_B, ANCHOR_Z)) - (95.0 + 25.0 * np.clip(bed - 150.0, 0, 200) / 100.0)
    zc = old_zc + shift
    H = q["Hk"] * (1.0 + (2.0 / 3.0) * np.clip((bed - 150.0) / 100.0, 0, 1))
    bot = min(zc + H / 2, bed - 1.0)
    top = max(bot - H, q["ztop_min"])
    if bot - top < 5.0:
        top = bot - 5.0
    return max(top, 0.0), bot


G["krill_day_layer"] = new_layer
BEDS, WCELL, CLS_IDX, MONTH = M["BEDS"], M["WCELL"], M["CLS_IDX"], M["MONTH"]
WPRES = np.full(12, 0.43)
WPRES[[4, 5, 6, 7]] = 0.05
WT = WPRES[MONTH]
_r0 = M["r0"]
STATE = {"contrast": False}


def r0_rule(q, L):
    if not STATE["contrast"]:
        return _r0(q, L)
    C0 = 0.4 if abs(L - q["L_krill"]) < 1e-12 else 0.7
    r00 = math.sqrt(C0) * q["k_v"] * math.sqrt(0.75 * L * 0.2 * L)
    r = r00
    for _ in range(80):
        r = r00 * math.exp(-q["c_beam"] * r / 2.0)
    return r


M["r0"] = r0_rule


def cls_a(a_t_b, krill):
    out = []
    for k, idx in enumerate(CLS_IDX):
        s = a_t_b[:, idx] @ WCELL[idx] / WCELL[idx].sum()
        out.append(float((s * WT).sum() / WT.sum()) if (k == 0 and krill) else float(s.mean()))
    return np.array(out)


dep = np.load(os.path.join(ROOT, "maps", "depth.npz"))["depth"].ravel()
cls = np.digitize(dep, [100.0, 150.0, 250.0])
ncell = np.array([(cls == k).mean() for k in range(4)])
shape = np.interp(dep, [49.9, 50.0, 99.9, 100.0, 150.0, 250.0, 530.0], [0.025, 0.38, 0.38, 0.8, 1.0, 1.2, 1.4])
shape = shape / shape.mean()
B_KRILL = np.array([5.0 * shape[cls == k].mean() for k in range(4)])
SAI = np.array([0.043, 0.127, 0.782, 0.048])
WHI = np.array([0.769, 0.204, 0.027, 0.001])
BVIS = {"planktivores": 1.776, "benthivores": 0.254, "whiting": 0.060, "mackerel": 0.119, "zooplankton": 9.722}
BOOK = {"krill": 0.361, "planktivores": 0.498, "benthivores": 0.077, "whiting": 0.042, "mackerel": 0.011,
        "zooplankton": 0.010}
PMAP = {"krill": "krill", "planktivores": "planktivorous_fish", "benthivores": "benthivorous_fish",
        "whiting": "whiting", "mackerel": "mackerel", "zooplankton": "zooplankton"}


def B_cls(prey, spread):
    if prey == "krill":
        return B_KRILL
    b = BVIS[prey] * np.ones(4)
    if prey == "whiting" and spread:
        b = BVIS[prey] * WHI / ncell
    return b


def gate(A):
    out = {}
    for name, w, spread in (("equal", ncell, False), ("saithe", SAI, True)):
        e = {p: float((w * A[p] * B_cls(p, spread)).sum()) for p in BOOK}
        t = sum(e.values())
        out[name] = {p: e[p] / t for p in BOOK}
    for k in range(4):
        e = {p: float(A[p][k] * B_cls(p, False)[k]) for p in BOOK}
        t = sum(e.values())
        out[f"class{k}"] = {p: e[p] / t for p in BOOK}
    return out


RES = {}
print("Band reading (V_one convention) for the water column; krill share and other prey / booked")
print("variant              | equal  | saithe spread | 150-250 m | plk  ben  whi (saithe spread, ratio to booked)")
for lab, contrast, rnv, ke, prev in (("V_one (sibling)", False, 0.2, 0.02, False),
                                      ("V_one_p (sibling)", False, 0.2, 0.02, True),
                                      ("band, rule A", True, 0.3, 0.02, False),
                                      ("band, rule A, rev p", True, 0.3, 0.02, True),
                                      ("band, rule A, Ke 2e-4", True, 0.3, 2e-4, False),
                                      ("band, rule A, Ke 0.1", True, 0.3, 0.1, False),
                                      ("band, rule A, Ke 2e-4 rev", True, 0.3, 2e-4, True),
                                      ("band, rule A, Ke 0.1 rev", True, 0.3, 0.1, True)):
    STATE["contrast"] = contrast
    q = M["level"](1)
    q["R_fish_dark"] = rnv
    q["Ke_gad"] = ke
    fa, _ = M["prey_a"](q, "meas")
    pa = M["pairs_from"](fa)
    A = {p: cls_a(pa[PMAP[p]], p == "krill") for p in BOOK}
    if prev:
        A = {p: A[p] * (1.2 if p == "krill" else 1.0 if p == "zooplankton" else 2.25) for p in BOOK}
    g = gate(A)
    RES[lab] = dict(a={p: A[p].tolist() for p in A}, gate=g)
    s = g["saithe"]
    print(f"{lab:26s} | {g['equal']['krill']:.3f} | {s['krill']:.3f} | {g['class2']['krill']:.3f} | "
          f"{s['planktivores'] / BOOK['planktivores']:.2f} {s['benthivores'] / BOOK['benthivores']:.2f} "
          f"{s['whiting'] / BOOK['whiting']:.2f}")
json.dump(RES, open(OUTJ, "w"), indent=1)
