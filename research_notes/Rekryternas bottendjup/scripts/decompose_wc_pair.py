"""Why the J shape moves the water column -> benthivores gate ratio less than the class-level F of
research_notes/"Sejens hojd over botten" suggested: the pair's depth_access_table already weights the encounter
toward shallow beds. Class-level effective co-occurrence on the model's 60 x 60 map:

  F_eff(X) = sum_k w_k m_k f^X_k / sum_k w_k m_k,
  w_k = the predator's field class share per cell (FIELD_BED_SHARES share / cell share), m_k = the pair's
  depth_access_table class value, f^X_k = the prey stage shape (mean 1 over the cells; this note's GOV shapes).

Usage: python3 -I decompose_wc_pair.py <fg_library.yaml> <gov_juv_bed.json>
(F without m is the plain class co-occurrence; F_eff with m is what the field-spread gate applies.)
"""
import json
import sys

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("needs PyYAML")

LIB = yaml.safe_load(open(sys.argv[1], encoding="utf-8-sig"))
SH = json.load(open(sys.argv[2]))
CELL = np.array(SH["_cell_shares"])
INTER = LIB["interaction_definitions"]
PRED = {"predatory_fish": np.array([0.043, 0.127, 0.782, 0.048]), "whiting": np.array([0.769, 0.204, 0.027, 0.001])}
STAGES = {"J ann": "h|box|q|ann|all|J", "Js ann (< 19 cm)": "h|box|q|ann|all|Js", "J Q3": "h|box|q|Q3|all|J",
          "import shape": "h|box|q|rec_import", "A ann": "h|box|q|ann|all|A"}
for pid, shares in PRED.items():
    rows = INTER[f"{pid}_preys_on_benthivorous_fish"]["depth_access_table"]
    m = np.array([rows[0]["factor"], rows[1]["factor"], rows[3]["factor"], rows[5]["factor"]])
    m = m / (CELL * m).sum()
    w = shares / CELL
    sa = INTER[f"{pid}_preys_on_benthivorous_fish"]["stage_access"]
    print(f"{pid} -> benthivores: predator class shares {shares.tolist()}; depth table (mean 1) "
          f"{np.round(m, 2).tolist()}; stage_access {sa}")
    eff = CELL * w * m / (CELL * w * m).sum()
    print(f"  encounter weight by class (predator x depth table): {np.round(eff, 3).tolist()} "
          f"(predator alone {np.round(shares, 3).tolist()})")
    for lab, key in STAGES.items():
        f = np.array(SH[key]["central"])
        F = float((CELL * w * f).sum())
        Fe = float((CELL * w * m * f).sum() / (CELL * w * m).sum())
        print(f"  {lab:18s} f {'/'.join(f'{x:.2f}' for x in f)}  F (no depth table) {F:.3f}  F_eff {Fe:.3f}")
    js = 0.155  # library start_share annual mean (0.139-0.169)
    fj, fa = sa
    tJ = js * fj
    tA = (1 - js) * fa
    shJ = tJ / (tJ + tA)
    FJ = float((CELL * w * m * np.array(SH[STAGES["J ann"]]["central"])).sum() / (CELL * w * m).sum())
    FA = float((CELL * w * m * np.array(SH[STAGES["A ann"]]["central"])).sum() / (CELL * w * m).sum())
    print(f"  J share of the pair's encounter at uniform prey {shJ:.2f}; pair factor with J ann + A ann "
          f"{shJ * FJ + (1 - shJ) * FA:.3f} (J part {FJ:.3f}, A part {FA:.3f})")
