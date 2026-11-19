"""Feeding level (realised intake / ceiling at the ambient temperature) and reserve fill by month
for the fish groups, frozen policies, one probe year."""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import interactions, predation
import inference
D, seed, DAYS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = json.load(open(D + '/config.json'))
FISH = ['benthivorous_fish', 'whiting', 'predatory_fish', 'planktivorous_fish', 'mackerel']
capbook = {}
orig = predation._reserve_room_cap
def hooked(env, demand, pb, ro):
    out = orig(env, demand, pb, ro)
    capbook['pre'] = (demand * env.handling_time_mat[:, :, None, None]).sum(axis=(1, 2, 3))
    capbook['post'] = (out * env.handling_time_mat[:, :, None, None]).sum(axis=(1, 2, 3))
    return out
predation._reserve_room_cap = hooked
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
idx = {f: env.dm_ids.index(f) for f in FISH}
h = env.handling_time_mat
acc = {f: np.zeros((12, 5)) for f in FISH}  # sum feed, sum uncapped, sum fill, n, sum B
with evaluation_randomness(seed * 31 + 7):
    for t in range(DAYS * 4):
        li = interactions.light_index(env)
        month = min(11, int((li / env.light_period) * 12))
        tf = interactions.ingestion_temperature(env)
        B = {f: float(env.fgs[f].biomass.sum()) for f in FISH}
        env.step()
        rows = env._tick_dm_intake
        for f, i in idx.items():
            if B[f] <= 0: continue
            q = float(tf[i]) if tf is not None else 1.0
            feed = float((rows[i] * h[i]).sum()) / B[f] / q
            unc = float(capbook['pre'][i]) / B[f] / q
            fg = env.fgs[f]
            fill = float(fg.energy_reserve.sum() / max(fg.biomass.sum() * fg.max_energy_reserve, 1e-12))
            a = acc[f][month]; a += [feed, unc, fill, 1, B[f]]
print(f"seed {seed}, {DAYS} d, {D}. feed = realised intake / ceiling (ambient T); uncapped = demand before the reserve cap / ceiling;")
print("fill = reserve / (B ME), model by month vs the library's start_fill (field)")
mon = 'JFMAMJJASOND'
for f in FISH:
    a = acc[f]; n = np.maximum(a[:, 3], 1)
    sf = env.fgs[f].start_fill
    print(f"\n{f}: year feed {a[:,0].sum()/a[:,3].sum():.3f}, uncapped {a[:,1].sum()/a[:,3].sum():.3f}, fill {a[:,2].sum()/a[:,3].sum():.3f}")
    print("  month " + ' '.join(f'{c:>5s}' for c in mon))
    print("  feed  " + ' '.join(f'{x:5.2f}' for x in a[:, 0] / n))
    print("  fill  " + ' '.join(f'{x:5.2f}' for x in a[:, 2] / n))
    if sf: print("  field " + ' '.join(f'{x:5.2f}' for x in sf))
