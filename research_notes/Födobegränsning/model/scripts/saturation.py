"""Saturation of each fish predator over its whole menu at the start densities (shared handling,
section 242): X = sum_j h_j a_j B_vis_j (eat share 1 on every pair). Intake while foraging, as a
share of the ceiling, is X / (1 + X); the prey factor f that halves it solves f X / (1 + f X) =
X / (2 (1 + X)), f = 1 / (2 + X). Equal cells, box-mean densities, depth-access maps averaged
over the prey's outside shape (as budget_gate). Usage (repo root): python3 <this>
"""
import sys
sys.path.insert(0, '.')
import train as train_mod
import yaml

env = train_mod._ProbeEnvBuilder(project_path='mareld2.yaml', grid_size=(16, 16),
                                 apply_natural_mortality=True, migration=True)()
man = yaml.safe_load(open('mareld2.yaml', encoding='utf-8-sig'))
meta = man['project_metadata']; area = float(meta['reference_grid_width']) * float(meta['reference_grid_height'])
dens = {e['group_id']: float(e['inference_initial_biomass']) / area
        for v in man.values() if isinstance(v, list) for e in v
        if isinstance(e, dict) and e.get('inference_initial_biomass') is not None and not e.get('muted')}


def depth_factor(i, j, q):
    if not getattr(env, '_has_pair_depth', False):
        return 1.0
    shape = getattr(env.fgs[q], 'outside_weight_map', None)
    m = env.pair_depth_mult[i, j]
    return float(m.mean()) if shape is None else float((m * shape).mean())


def visible(i, j, q):
    if getattr(env, '_has_stages', False) and j in set(env.stage_rows.tolist()):
        js = float(env.stage_start[:, j].mean())
        f = js * float(env.stage_f_j[i, j]) + (1 - js) * float(env.stage_f_a[i, j])
    else:
        f = float(env.access_mat[i, j])
    phi = getattr(env.fgs[q], 'exposed_share', None)
    if phi is not None and not env.fgs[q].is_decision_maker:
        f *= float(phi)
    return f


for pid in ('benthivorous_fish', 'whiting', 'predatory_fish', 'planktivorous_fish', 'mackerel'):
    i = env.dm_ids.index(pid)
    rows = []
    for j in range(env.N_all):
        if not env.eat_static_mask[i, j]:
            continue
        q = env.global_fg_order[j]
        if q not in dens:
            continue
        a = float(env.max_intake_mat[i, j]) * depth_factor(i, j, q)
        h = float(env.handling_time_mat[i, j])
        bv = dens[q] * visible(i, j, q)
        t3 = float(env._type3_pred_mask[i, j, 0, 0]) > 0
        rows.append((q, h * a * bv * (bv if t3 else 1.0), 1 / (a * h) if a * h > 0 else float('inf'), bv))
    X = sum(r[1] for r in rows)
    print(f"\n{pid}: X = sum h a B_vis = {X:.1f}; share of ceiling while foraging {X / (1 + X):.3f}; "
          f"every prey x {0.9 / (1 + 0.1 * X):.3f} for a 10 % intake drop, x {1 / (2 + X):.3f} to halve it")
    for q, x, k, bv in sorted(rows, key=lambda r: -r[1]):
        print(f"  {q:>20} B_vis {bv:7.3f} t/km2  1/(a h) {k:8.3g}  h a B_vis {x:7.2f}")
