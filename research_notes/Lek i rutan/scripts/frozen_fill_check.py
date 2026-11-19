"""Frozen fix104: fill and ration by month, live library vs the spawning copy."""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import interactions
import inference
D, seed, DAYS, lib = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = json.load(open(D + '/config.json'))
if lib != 'live':
    cfg['library'] = lib
FISH = ['benthivorous_fish', 'whiting', 'predatory_fish', 'planktivorous_fish']
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
print('spawning on:', getattr(env, '_has_spawning', False), [env.dm_ids[i] for i in np.flatnonzero(getattr(env, 'dm_spawning', []))])
acc = {f: np.zeros((12, 5)) for f in FISH}
B0 = {f: float(env.fgs[f].biomass.sum()) for f in FISH}
env.gamete_energy_shed = {}
with evaluation_randomness(seed * 31 + 7):
    for t in range(DAYS * 4):
        li = interactions.light_index(env)
        month = min(11, int((li / env.light_period) * 12))
        env.step()
        for f in FISH:
            fg = env.fgs[f]; i = env.dm_ids.index(f); B = float(fg.biomass.sum())
            fill = float(fg.energy_reserve.sum() / max(B * fg.max_energy_reserve, 1e-12))
            eat = float(env._tick_dm_intake[i].sum()) / max(B, 1e-12) * 1460
            top = float(env.set_point_table[li, i]) if getattr(env, '_has_set_point', False) else 1.0
            acc[f][month] += [fill, eat, top, 1, B / B0[f]]
print(f"seed {seed}, {DAYS} d, library {lib}")
for f in FISH:
    a = acc[f]; n = np.maximum(a[:, 3], 1)
    print(f"{f}: shed {env.gamete_energy_shed.get(f, 0.0) / B0[f]:.0f} MJ/t start B; B end/B0 {a[11 if a[11,3] else 0, 4]/n[11 if a[11,3] else 0]:.3f}")
    print("  fill  " + ' '.join(f'{x:5.2f}' for x in a[:, 0] / n))
    print("  top   " + ' '.join(f'{x:5.2f}' for x in a[:, 2] / n))
    print("  ration" + ' '.join(f'{x:6.2f}' for x in a[:, 1] / n))
