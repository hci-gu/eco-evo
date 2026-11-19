"""Policy-free water-column (WC) diet and saturation PER BED CLASS, under the library and under the profile /
vision variants of class_terms.py: what a WC tonne standing over each bed class would eat if it attacked every
prey it met (section 265.1 fixed attack, shared handling, T = 1, interference left out).

Usage: python3 class_diet.py <repo_root> <class_terms.json>

The env is built exactly as tools/probes/budget_gate.py builds it (16 x 16 probe grid, mareld2.yaml, the
library on disk, read only). Prey sit at their manifest start density x the budget_gate field-spread shape
(open-pool depth shape for krill; benthivores by stage, J on the recruit shape and adults on GOV shares;
whiting on its GOV shares; every other prey uniform) x the pair's visible share. Per class c and WC pair j:
    enc_j(c) = mean over the class's cells of a_j m_j(cell) B_vis,j(cell)
    S(c) = sum_j enc_j(c) h_j;  share of the Holling ceiling taken = S / (1 + S);  diet share_j = enc_j / sum enc.
"library" uses the pair tables on disk. A variant replaces, for krill, planktivores, benthivores and whiting
only, the class multipliers m by the variant's class shape (equal-cell mean 1 on this grid) and scales a by
the variant's equal-cell a over the library derivation's (exp15|A); the section 270 capture-p x schooling
factors are constants on a and independent of the profile ("Sejens hojd over botten" M2), so they carry over.
Nothing is fitted; no model output is read.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
TJ = os.path.abspath(sys.argv[2])
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import io, contextlib  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    import train as train_mod  # noqa: E402
    import yaml  # noqa: E402
    from lib.diagnostics.field_diet import (FIELD_BED_SHARES, bed_class, field_prey_shape,  # noqa: E402
                                            field_stage_shapes, field_cell_weights)
    from lib.world import habitat  # noqa: E402
    PROJECT = "mareld2.yaml"
    H = W = 16
    env = train_mod._ProbeEnvBuilder(project_path=PROJECT, grid_size=(H, W), apply_natural_mortality=True,
                                     migration=True)()
    env._build_static_caches()
man = yaml.safe_load(open(PROJECT, encoding="utf-8-sig"))
meta = man["project_metadata"]
area = float(meta["reference_grid_width"]) * float(meta["reference_grid_height"])
density = {}
for entries in man.values():
    if isinstance(entries, list):
        for e in entries:
            if isinstance(e, dict) and "group_id" in e and e.get("inference_initial_biomass") is not None:
                density[e["group_id"]] = float(e["inference_initial_biomass"]) / area
depth = habitat.load_depth_map(man, PROJECT, (H, W))
cls = bed_class(depth)
CN = ["< 100", "100-150", "150-250", ">= 250"]


def visible_share(i, j, prey_id):
    if getattr(env, "_has_stages", False) and j in set(env.stage_rows.tolist()):
        js = float(env.stage_start[:, j].mean())
        f = js * float(env.stage_f_j[i, j]) + (1.0 - js) * float(env.stage_f_a[i, j])
    else:
        f = float(env.access_mat[i, j])
    phi = getattr(env.fgs[prey_id], "exposed_share", None)
    if phi is not None and not env.fgs[prey_id].is_decision_maker:
        f *= float(phi)
    return f


def prey_cells(i, j, prey_id):
    stage = field_stage_shapes(prey_id, env.fgs[prey_id], depth)
    if stage is not None and getattr(env, "_has_stages", False) and j in set(env.stage_rows.tolist()):
        js = float(env.stage_start[:, j].mean())
        juv, adult = stage
        return js * float(env.stage_f_j[i, j]) * juv + (1.0 - js) * float(env.stage_f_a[i, j]) * adult
    return field_prey_shape(prey_id, env.fgs[prey_id], depth) * visible_share(i, j, prey_id)


T = json.load(open(TJ))
RUNS = T["runs"]
PAIRMAP = {"krill": "krill", "planktivorous_fish": "predatory_fish->planktivorous_fish",
           "benthivorous_fish": "predatory_fish->benthivorous_fish", "whiting": "predatory_fish->whiting"}
i = env.dm_ids.index("predatory_fish")
menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
prey_ids = [env.global_fg_order[j] for j in menu]
saithe_w = field_cell_weights(depth, FIELD_BED_SHARES["predatory_fish"][0])


def run(variant):
    rows = {}
    for j, pid in zip(menu, prey_ids):
        if pid not in density:
            continue
        a = float(env.max_intake_mat[i, j])
        m = (np.asarray(env.pair_depth_mult[i, j], dtype=np.float64) if getattr(env, "_has_pair_depth", False)
             else np.ones((H, W)))
        if variant != "library" and pid in PAIRMAP:
            key = PAIRMAP[pid]
            v, c0 = RUNS[variant][key], RUNS["exp15|A"][key]
            share = np.array(T["share"])
            eqv = float(share @ np.array(v["classes"])) if key == "krill" else v["eq"]
            eq0 = float(share @ np.array(c0["classes"])) if key == "krill" else c0["eq"]
            rel = np.array(v["classes"]) / eqv
            m = rel[cls]
            m = m / m.mean()
            a *= eqv / eq0
        bv = density[pid] * prey_cells(i, j, pid)
        h = float(env.handling_time_mat[i, j])
        enc = np.array([float((a * m * bv)[cls == k].mean()) if (cls == k).any() else np.nan for k in range(4)])
        enc_field = float((saithe_w * a * m * bv).sum())
        rows[pid] = (enc, h, enc_field)
    return rows


out = {}
print(f"cells per class on the {H}x{W} probe grid: " + " / ".join(f"{(cls == k).mean():.3f}" for k in range(4)))
print("Densities (t/km2): " + ", ".join(f"{p} {density[p]:.3g}" for p in prey_ids if p in density))
for variant in ["library"] + list(RUNS):
    rows = run(variant)
    E = np.array([r[0] for r in rows.values()])          # (prey, class)
    Hh = np.array([r[1] for r in rows.values()])
    S = (E * Hh[:, None]).sum(0)
    sh = E / E.sum(0)
    fe = np.array([r[2] for r in rows.values()])
    fsh = fe / fe.sum()
    Sf = float((fe * Hh).sum())
    out[variant] = dict(prey=list(rows), enc=E.tolist(), share=sh.tolist(), S=S.tolist(), field_share=fsh.tolist(), S_field=Sf)
    print(f"\n== {variant}")
    print(f"  {'prey':20s} " + " ".join(f"{c:>9s}" for c in CN) + "   field spread")
    for k, p in enumerate(rows):
        print(f"  {p:20s} " + " ".join(f"{x:9.3f}" for x in sh[k]) + f"   {fsh[k]:9.3f}")
    print(f"  {'S = sum a h B':20s} " + " ".join(f"{x:9.2f}" for x in S) + f"   {Sf:9.2f}")
    print(f"  {'ceiling share S/(1+S)':20s} " + " ".join(f"{x / (1 + x):9.3f}" for x in S) + f"   {Sf / (1 + Sf):9.3f}")
    print(f"  {'intake vs 150-250 m':20s} " + " ".join(f"{(x / (1 + x)) / (S[2] / (1 + S[2])):9.2f}" for x in S))
json.dump(out, open(os.path.join(os.path.dirname(TJ), "class_diet.json"), "w"), indent=1)
