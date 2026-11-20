"""Do the fish prey avoid every predator, or the clumped benthivores most? Cell overlap of whiting and planktivores
with each fish predator and that predator's Lloyd crowding. Frozen, 1 year, daily. Usage: POLICY_DIR SEED LIBRARY"""
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
PRED = ['benthivorous_fish', 'predatory_fish', 'whiting', 'mackerel', 'seals', 'porpoises']
PREY = ['whiting', 'planktivorous_fish']
ov = {(p, q): [] for p in PRED for q in PREY if p != q}; ll = {p: [] for p in PRED}
lloyd = lambda x: x.size * (x ** 2).sum() / x.sum() ** 2
with evaluation_randomness(seed * 13 + 5):
    for t in range(365 * 4):
        env.step()
        if t % 4: continue
        for p in PRED:
            P = env.fgs[p].biomass.astype(np.float64)
            if P.sum() <= 0: continue
            ll[p].append(lloyd(P))
            for q in PREY:
                if p == q: continue
                Q = env.fgs[q].biomass.astype(np.float64)
                ov[(p, q)].append(Q.size * (P * Q).sum() / (P.sum() * Q.sum()))
print(f"seed {seed}: cell overlap (1 = no association) of prey with predator, and the predator's Lloyd")
for p in PRED:
    print(f"  {p:18s} Lloyd {np.mean(ll[p]):6.1f}  " + "  ".join(
        f"{q}: {np.mean(ov[(p, q)]):.2f}" for q in PREY if (p, q) in ov and ov[(p, q)]))
