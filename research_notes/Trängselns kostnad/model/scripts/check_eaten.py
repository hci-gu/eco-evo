"""Check: does the post-room-cap demand ('eaten' in cf_field_diag.py) equal the realised gain after the harvest caps?"""
import sys, json, copy, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import predation
import inference
D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
BE = 'benthivorous_fish'
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
REC = []; _orig = predation._reserve_room_cap; _ap = predation.apply_predation
def _rec(env, demand, pb, ro):
    out = _orig(env, demand, pb, ro); i = env.dm_ids.index(BE)
    REC.append([np.asarray((out[i] * env.energy_gain_mat[i][:, None, None]).sum(axis=0), np.float64)]); return out
def _ap2(env, actions):
    r = _ap(env, actions); REC[-1].append(np.asarray(env.fgs[BE].temp_energy_gains, np.float64).copy())
    REC[-1].append(np.asarray(env.fgs[BE].biomass, np.float64).copy()); return r
predation._reserve_room_cap = _rec; predation.apply_predation = _ap2
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
REC.clear()
with evaluation_randomness(7):
    for t in range(40): env.step()
got = np.array([r[0].sum() for r in REC]); real = np.array([r[1].sum() for r in REC])
print(f"ticks {len(REC)}: total realised / post-room-cap {real.sum() / got.sum():.4f}")
k = np.concatenate([r[2] / r[2].mean() for r in REC]); B = np.concatenate([r[2] for r in REC])
for name, idx in (('post-room-cap', 0), ('realised', 1)):
    y = np.concatenate([r[idx] for r in REC]) / np.maximum(B, 1e-30); ok = (B > 1e-6) & (y > 0)
    X = np.column_stack([np.ones(ok.sum()), np.log(k[ok])]); sw = np.sqrt(B[ok] / B[ok].sum())
    print(f"  elasticity {name}: {np.linalg.lstsq(X * sw[:, None], np.log(y[ok]) * sw, rcond=None)[0][1]:+.4f}")
