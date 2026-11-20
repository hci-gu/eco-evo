"""The factor on a that the BOOKED field flows imply, pair by pair, with the shared-handling denominator.
Policy-free: the library, the manifest and the 16 x 16 probe env (as tools/probes/budget_gate.py builds it).

Per predator g at the manifest's inference densities, equal cells, every prey visible as in budget_gate's diet
gate (accessible / stage / refuge shares; the krill pairs' depth table x the pool's outside shape), foraging the
whole tick (T = 1, so the factor includes any non-foraging time):
    R_j    booked flow per t predator per tick = M2_j B_j / B_g / 1460
    S      field feeding level on the ceiling  = sum_j h_j R_j
    enc_j  chain encounter per t predator per tick = a_j B_vis,j (budget_gate's 'enc')
    f_j    = R_j / ((1 - S) enc_j)       (the multi-prey Type II response R_j = f a_j B_j / (1 + sum_k f a_k h_k B_k)
                                          solved for f pair by pair: equal f on every pair <=> the gate's shares)
Equal f_j within a predator = the booked diet is the chain's encounter diet scaled by one factor; f_j is the
level the booked flow (field diet share x field Q/B) implies. It shares every number with the booked ration.
Usage: python3 booked_level.py [LIBRARY]   (default: the live library, read only)
"""
import os, sys
import numpy as np
ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import yaml
import train as train_mod
LIB = sys.argv[1] if len(sys.argv) > 1 else None
kw = dict(project_path='mareld2.yaml', grid_size=(16, 16), apply_natural_mortality=True, migration=True)
if LIB:
    kw['library_path'] = LIB
try:
    builder = train_mod._ProbeEnvBuilder(**kw)
except TypeError:
    kw.pop('library_path', None); builder = train_mod._ProbeEnvBuilder(**kw)
env = builder(); env._build_static_caches()
man = yaml.safe_load(open('mareld2.yaml', encoding='utf-8-sig'))
meta = man['project_metadata']; area = float(meta['reference_grid_width']) * float(meta['reference_grid_height'])
dens = {}
for entries in man.values():
    if isinstance(entries, list):
        for e in entries:
            if isinstance(e, dict) and 'group_id' in e and e.get('inference_initial_biomass') is not None:
                dens[e['group_id']] = float(e['inference_initial_biomass']) / area


def depth_factor(i, j, prey_id):
    if not getattr(env, '_has_pair_depth', False):
        return 1.0
    shape = getattr(env.fgs[prey_id], 'outside_weight_map', None)
    mult = env.pair_depth_mult[i, j]
    return float(mult.mean()) if shape is None else float((mult * shape).mean())


def visible_share(i, j, prey_id):
    if getattr(env, '_has_stages', False) and j in set(env.stage_rows.tolist()):
        js = float(env.stage_start[:, j].mean())
        f = js * float(env.stage_f_j[i, j]) + (1.0 - js) * float(env.stage_f_a[i, j])
    else:
        f = float(env.access_mat[i, j])
    phi = getattr(env.fgs[prey_id], 'exposed_share', None)
    if phi is not None and not env.fgs[prey_id].is_decision_maker:
        f *= float(phi)
    return f


print(f"library: {LIB or 'live fgconfig/fg_library.yaml'}; densities t/km2 at the manifest inference start")
for g in ['planktivorous_fish', 'whiting', 'predatory_fish', 'mackerel', 'benthivorous_fish']:
    i = env.dm_ids.index(g); Bg = dens[g]
    pairs = env.fgs[g].params.get('interaction', {}) or {}
    rows = []
    for j in range(env.N_all):
        if not env.eat_static_mask[i, j]:
            continue
        p = env.global_fg_order[j]
        b = dens.get(p)
        if b is None:
            continue
        m2 = float((pairs.get(f"{g}_preys_on_{p}") or {}).get('predation_mortality', 0.0) or 0.0)
        a = float(env.max_intake_mat[i, j]) * depth_factor(i, j, p)
        bv = b * visible_share(i, j, p)
        t3 = float(env._type3_pred_mask[i, j, 0, 0]) > 0.0
        enc = a * bv * (bv if t3 else 1.0)
        R = m2 * b / Bg / 1460.0
        rows.append((p, b, bv, enc, R, float(env.handling_time_mat[i, j]), m2))
    S = sum(r[4] * r[5] for r in rows)
    X = sum(r[3] * r[5] for r in rows)
    print(f"\n{g}: B {Bg:.3f}; booked field feeding level S {S:.3f} (X_field {S / (1 - S):.3f}); chain X at T 1, "
          f"equal cells {X:.2f}; common factor X_field / X_chain {S / (1 - S) / X:.4f}")
    print(f"  {'prey':22s} {'M2':>7s} {'B':>6s} {'B_vis':>7s} {'R %BW/d':>8s} {'h R':>6s} {'enc':>9s} {'f_j':>8s}")
    for p, b, bv, enc, R, h, m2 in sorted(rows, key=lambda r: -r[4]):
        f = R / ((1 - S) * enc) if enc > 0 else float('nan')
        print(f"  {p:22s} {m2:7.4f} {b:6.2f} {bv:7.3f} {R * 400:8.3f} {h * R:6.3f} {enc:9.5f} {f:8.4f}")
