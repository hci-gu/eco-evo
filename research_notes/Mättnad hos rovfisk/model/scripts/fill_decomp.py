"""Why are the water column and whiting full? Monthly decomposition of their feeding in a frozen world.
Per tick (hooks inside the predation step): T = foraging share (interactions.attack_choice), pre-cap demand and
post-cap (reserve room) demand per prey, predator biomass. Feeding level = sum_j demand_j h_j / B / q10 factor
(share of the ceiling at the ambient temperature, as Fodobegransning feeding_level_and_fill.py).
  T          biomass-weighted foraging share
  sat        uncapped feeding level / T  (share of the ceiling while foraging)
  unc, feed  feeding level before / after the reserve-room cap
  krill      krill's share of the uncapped handling time
  fill       reserve / (B ME)
Usage: fill_decomp.py POLICY_DIR SEED LIBRARY"""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import interactions, predation
import inference
D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
FISH = ['predatory_fish', 'whiting', 'benthivorous_fish']
G = {}
o_ac = interactions.attack_choice
def h_ac(eat):
    T, p = o_ac(eat); G['T'] = np.array(T, dtype=np.float64); return T, p
interactions.attack_choice = h_ac
o_cap = predation._reserve_room_cap
def h_cap(env, demand, pb, ro):
    out = o_cap(env, demand, pb, ro)
    G['pre'] = np.array(demand, dtype=np.float64); G['post'] = np.array(out, dtype=np.float64); G['B'] = np.array(pb, dtype=np.float64)
    return out
predation._reserve_room_cap = h_cap
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
order = list(env.global_fg_order); jk = order.index('krill')
h = env.handling_time_mat
idx = {f: env.dm_ids.index(f) for f in FISH}
acc = {f: np.zeros((12, 8)) for f in FISH}   # T, sat-num, unc, feed, krill-num, fill, n, B
with evaluation_randomness(seed * 13 + 5):
    for t in range(1460):
        li = interactions.light_index(env); month = min(11, int((li / env.light_period) * 12))
        tf = interactions.ingestion_temperature(env)
        env.step()
        for f, i in idx.items():
            B = G['B'][i]; Bt = B.sum()
            if Bt <= 0: continue
            q = float(tf[i]) if tf is not None else 1.0
            Tm = float((G['T'][i].reshape(B.shape) * B).sum() / Bt)
            pre_h = (G['pre'][i] * h[i][:, None, None]); post_h = (G['post'][i] * h[i][:, None, None])
            unc = float(pre_h.sum()) / Bt / q; feed = float(post_h.sum()) / Bt / q
            kr = float(pre_h[jk].sum()) / max(float(pre_h.sum()), 1e-30)
            fg = env.fgs[f]
            fill = float(fg.energy_reserve.sum() / max(fg.biomass.sum() * fg.max_energy_reserve, 1e-12))
            acc[f][month] += [Tm, unc / max(Tm, 1e-9), unc, feed, kr, fill, 1, Bt]
mon = 'JFMAMJJASOND'
print(f"seed {seed}; {D}; library {LIB}")
for f in FISH:
    a = acc[f]; n = np.maximum(a[:, 6], 1); tot = a[:, :6].sum(0) / a[:, 6].sum()
    sf = env.fgs[f].start_fill
    print(f"\n{f}: year T {tot[0]:.2f}, sat while foraging {tot[1]:.2f}, uncapped {tot[2]:.2f}, feed {tot[3]:.2f}, "
          f"krill share of uncapped {tot[4]:.2f}, fill {tot[5]:.3f}")
    print("  month  " + ' '.join(f'{c:>5s}' for c in mon))
    for k, lab in ((0, 'T'), (1, 'sat'), (2, 'unc'), (3, 'feed'), (4, 'krill'), (5, 'fill')):
        print(f"  {lab:6s} " + ' '.join(f'{x:5.2f}' for x in a[:, k] / n))
    if sf: print("  field  " + ' '.join(f'{x:5.2f}' for x in sf))
