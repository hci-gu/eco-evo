"""Benthivore realised diet (tonnes, per year) under a given infauna refuge. Frozen policies, 3 years,
same random context as refuge_clump.py. Usage: refuge_diet.py POLICY_DIR SEED LIBRARY"""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
import inference
D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
BE = 'benthivorous_fish'
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
inter = env.fgs[BE].params['interaction']
booked = {k.split('_preys_on_')[1]: float(v.get('predation_mortality', 0) or 0) for k, v in inter.items()
          if k.startswith(BE + '_preys_on_') and v.get('preys_on', True)}
inf = env.fgs['benthic_infauna']
print(f"seed {seed}; phi {inf.exposed_share}, k {inf.refuge_exchange_rate * 1460:.2f}")
prev = {q: float(x) for q, x in env.intake_by_pred_prey.get(BE, {}).items()}
with evaluation_randomness(seed * 13 + 5):
    for y in range(3):
        Bsum = {g: 0.0 for g in env.fgs}
        for t in range(1460):
            for g in Bsum: Bsum[g] += float(env.fgs[g].biomass.sum())
            env.step()
        now = {q: float(x) for q, x in env.intake_by_pred_prey.get(BE, {}).items()}
        d = {q: now.get(q, 0) - prev.get(q, 0) for q in now}; prev = now
        tot = sum(d.values()); Bb = Bsum[BE] / 1460
        print(f"year {y+1}: ration {tot / Bb:.2f} t/t/yr; " + ", ".join(
            f"{q} {d[q] / tot:.3f} (M2 {d[q] / (Bsum[q] / 1460):.4f} vs booked {booked.get(q, 0):.4f})"
            for q in sorted(d, key=lambda q: -d[q])))
