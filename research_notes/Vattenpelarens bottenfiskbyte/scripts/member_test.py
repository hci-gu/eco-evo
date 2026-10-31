"""Per-member clearance test: each member's own encounter a on benthivores vs planktivores (and whiting),
with its own stage access, against that member's field diet. No model output read.

Usage: python3 -I member_test.py <Arsungarnas_andel_dir>

Model per member m: a_m,ben f_m,ben / (a_m,plk f_m,plk), a from research_notes/"Rovfiskens uppdelning"
tables/split_encounter_mack_field_stdout.txt "Per-stage central a" (annual, per member tonne; stages
weighted by W), f by access_lib.py at the box J/B (benthivores 0.15, planktivores 0.38, whiting 0.165).
Field: mass ratio x B_plk / B_ben (B 3.33, 1.53, whiting 0.45).
Field diets (shares of the checked prey; field_diets.py): saithe CRR 219 / CRR 164 box-weighted; SMS
observed (saithe, ben/(pout+herring) 0.080); spurdog Ecopath 1991; hake ICES DB Skagerrak 2000+ (115
stomachs); anglerfish Ecopath 1991 and ICES DB Skagerrak 2000+ (8 stomachs, 97 g: ben 0.63, plk 0, own
0, sfish 0.14 - only bounds: plk 0 gives no ratio).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from access_lib import Access  # noqa: E402

ACC = Access(sys.argv[1])
F = json.load(open(os.path.join(HERE, "..", "tables", "field_diets.json")))
B = {"plk": 3.33, "ben": 1.53, "wht": 0.45}
JB = {"plk": 0.38, "ben": 0.15, "wht": 0.165}
# per-stage central a (W, plk, ben, wht_prey) from split_encounter_mack_field_stdout.txt
STAGES = {
    "saithe": [(0.2588, 0.007607, 0.08048, 0.0161), (0.1506, 0.002889, 0.03158, 0.006316)],
    "spurdog": [(0.0500, 0.003369, 0.06879, 0.01376)],
    "hake": [(0.0004, 0.1123, 1.062, 0.2124), (0.0245, 0.005702, 0.07298, 0.0146)],
    "anglerfish": [(0.0137, 0.0003812, 0.02685, 0.00537)],
}
IDENT = {"saithe": ACC.sa.GADOIDS, "spurdog": ACC.sa.FLAT | {"haddock"}, "hake": None, "anglerfish": None}
FIELD = {
    "saithe": [("CRR 219", F["saithe219"]), ("CRR 164", F["saithe164"]),
               ("SMS obs", {"ben": 0.080, "plk": 1.0, "wht": 0.7 / 51})],
    "spurdog": [("Ecopath 1991", F["others"]["spurdog"])],
    "hake": [("ICES Skagerrak 2000+", F["others"]["hake"])],
    "anglerfish": [("Ecopath 1991", F["others"]["anglerfish"])],
}


def feff(group, m, variant):
    out = []
    for stage in ("J", "A"):
        out.append(sum(ACC.member_f(q, group, m, IDENT[m] if group == "ben" else None, stage, variant) for q in (1, 2, 3, 4)) / 4)
    return JB[group] * out[0] + (1 - JB[group]) * out[1], out


print("Per member: model clearance ratio vs field (> 1 = model too high), benthivores and whiting relative to planktivores")
for m, st in STAGES.items():
    W = sum(s[0] for s in st)
    a = {k: sum(s[0] * s[i] for s in st) / W for k, i in (("plk", 1), ("ben", 2), ("wht", 3))}
    print(f"\n{m}: a per member tonne plk {a['plk']:.5f} ben {a['ben']:.5f} wht {a['wht']:.5f} (ben/plk {a['ben'] / a['plk']:.1f})")
    for variant in ("lib", "aw"):
        f = {g: feff(g, m, variant) for g in ("plk", "ben", "wht")}
        mod = {g: a[g] * f[g][0] for g in f}
        print(f"  {variant:3s} access: plk [{f['plk'][1][0]:.3f}, {f['plk'][1][1]:.3f}] ben [{f['ben'][1][0]:.3f}, {f['ben'][1][1]:.4f}]"
              f" wht [{f['wht'][1][0]:.3f}, {f['wht'][1][1]:.3f}] -> model c_ben/c_plk {mod['ben'] / mod['plk']:.2f},"
              f" c_wht/c_plk {mod['wht'] / mod['plk']:.2f}")
        for name, d in FIELD[m]:
            cb = d["ben"] / B["ben"] / (d["plk"] / B["plk"])
            cw = d["wht"] / B["wht"] / (d["plk"] / B["plk"])
            print(f"       field {name:22s} c_ben/c_plk {cb:.3f} -> model/field {mod['ben'] / mod['plk'] / cb:6.1f};"
                  f" c_wht/c_plk {cw:.3f} -> {mod['wht'] / mod['plk'] / cw if cw > 0 else float('nan'):6.1f}")
