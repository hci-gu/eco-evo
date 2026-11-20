"""One frozen year (after the run's warm-up) on a given library: bed-class placement, realised diets and prey-side M2.
Usage: frozen_compare.py POLICY_DIR SEED LIBRARY"""
import sys, json, numpy as np, yaml
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.world import habitat
from lib.diagnostics.field_diet import bed_class
import inference
D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    B0 = {g: float(env.fgs[g].biomass.sum()) for g in env.fgs}
    settle(env, cfg['warmup_ticks'], seed=seed)
man = yaml.safe_load(open(cfg['project']))
cls = bed_class(habitat.load_depth_map(man, cfg['project'], (env.H, env.W)))
G = ['whiting', 'predatory_fish', 'benthivorous_fish', 'planktivorous_fish']
PRED = ['benthivorous_fish', 'whiting', 'predatory_fish']
sh = {g: [] for g in G}; Bsum = {g: 0.0 for g in env.fgs}
prev = {p: {q: float(x) for q, x in env.intake_by_pred_prey.get(p, {}).items()} for p in env.fgs}
with evaluation_randomness(seed * 13 + 5):
    for t in range(1460):
        for g in Bsum: Bsum[g] += float(env.fgs[g].biomass.sum())
        env.step()
        if t % 4 == 0:
            for g in G:
                x = env.fgs[g].biomass.astype(np.float64); sh[g].append([x[cls == k].sum() / x.sum() for k in range(4)])
now = {p: {q: float(x) for q, x in env.intake_by_pred_prey.get(p, {}).items()} for p in env.fgs}
d = {p: {q: now[p].get(q, 0) - prev[p].get(q, 0) for q in now[p]} for p in now}
Bm = {g: Bsum[g] / 1460 for g in Bsum}
fmt = lambda s: '/'.join(f'{x:.2f}' for x in s)
print(f"seed {seed}; library {LIB}")
print("bed-class shares (< 100 / 100-150 / 150-250 / >= 250 m):")
for g in G: print(f"  {g:20s} {fmt(np.mean(sh[g], 0))}")
print("realised diets (share of tonnes) and ration t/t/yr:")
for p in PRED:
    tot = sum(d[p].values())
    print(f"  {p:20s} ration {tot / Bm[p]:.2f}: " + ", ".join(f"{q} {x / tot:.3f}" for q, x in sorted(d[p].items(), key=lambda z: -z[1]) if x / tot > 0.002))
print("prey-side M2 (/yr) by predator:")
for q in ['whiting', 'planktivorous_fish', 'benthivorous_fish', 'krill']:
    m2 = {p: d[p].get(q, 0) / Bm[q] for p in d if d[p].get(q, 0) > 0}
    print(f"  {q:20s} total {sum(m2.values()):.3f}: " + ", ".join(f"{p} {x:.3f}" for p, x in sorted(m2.items(), key=lambda z: -z[1]) if x > 1e-3))
print("year-end biomass / start: " + ", ".join(f"{g} {float(env.fgs[g].biomass.sum()) / B0[g]:.2f}" for g in
      ['whiting', 'benthivorous_fish', 'predatory_fish', 'planktivorous_fish', 'mackerel', 'seals', 'porpoises', 'krill', 'mobile_epifauna']))
