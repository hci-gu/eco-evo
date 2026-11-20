"""Is whiting's segregation from the benthivores food competition? Epifauna/K and krill met by the mean tonne of
each group vs the grid mean; whiting's overlap with benthivores by the benthivores' age in a cell is not needed.
Frozen, 1 year, daily. Usage: POLICY_DIR SEED LIBRARY"""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
import inference
D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
epi = env.fgs['mobile_epifauna']; K = np.asarray(epi.capacity_map, dtype=np.float64)
G = ['whiting', 'benthivorous_fish', 'planktivorous_fish', 'predatory_fish']
r = {g: {'epi': [], 'krill': []} for g in G}; corr = []
with evaluation_randomness(seed * 13 + 5):
    for t in range(365 * 4):
        env.step()
        if t % 4: continue
        e = epi.biomass.astype(np.float64) / np.maximum(K, 1e-12); ok = K > 0
        kr = env.fgs['krill'].biomass.astype(np.float64)
        for g in G:
            B = env.fgs[g].biomass.astype(np.float64)
            r[g]['epi'].append((B[ok] * e[ok]).sum() / B[ok].sum() / e[ok].mean())
            r[g]['krill'].append((B * kr).sum() / B.sum() / kr.mean())
        W = env.fgs['whiting'].biomass.astype(np.float64); Bb = env.fgs['benthivorous_fish'].biomass.astype(np.float64)
        hot = Bb > 8 * Bb.mean()
        corr.append(((W[hot].sum() / W.sum()) / max(hot.mean(), 1e-9), e[hot & ok].mean() / e[ok].mean() if (hot & ok).any() else np.nan))
print(f"seed {seed}: resource met by the mean tonne / grid mean (epifauna as /K)")
for g in G:
    print(f"  {g:20s} epifauna {np.mean(r[g]['epi']):.2f}  krill {np.mean(r[g]['krill']):.2f}")
c = np.array(corr)
print(f"  cells with benthivores > 8x mean: whiting density there / mean {np.nanmean(c[:, 0]):.2f}; epifauna/K there / mean {np.nanmean(c[:, 1]):.2f}")
