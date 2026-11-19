import sys, json, copy, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
sys.argv_saved = list(sys.argv)
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
import inference

D, seed = sys.argv[1], int(sys.argv[2])
G, P = 'benthivorous_fish', 'predatory_fish'
SNAP_DAYS = [0, 91, 182, 273]; W = 4
cfg = json.load(open(D + '/config.json'))
src = open(sys.argv[0].replace('cf_wc.py', 'cf_spread.py')).read()
ns = {'np': np, 'G': G}
exec("from scipy.ndimage import gaussian_filter\nlloyd = lambda B: float((B**2).sum() * B.size / max(B.sum(), 1e-12)**2)\n" + src[src.index("def spread"):src.index("def books")], ns)
spread = ns['spread']; lloyd = ns['lloyd']
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    B0 = {g: float(env.fgs[g].biomass.sum()) for g in env.fgs}
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4
lib = env.fgs[P].params.get('interaction', {})
booked = {k.split('_preys_on_')[1]: float(v.get('predation_mortality', 0) or 0) for k, v in lib.items()
          if k.startswith(P + '_preys_on_') and v.get('preys_on')}
# booked flows at start densities (t/yr) and the booked ration / diet share of the benthivores
bflow = {q: booked[q] * B0[q] for q in booked if q in B0}
ration_booked = sum(bflow.values()) / B0[P]; share_booked = bflow[G] / sum(bflow.values())
print(f"booked: WC ration {ration_booked:.2f} t/t/yr, benthivore share of WC diet {share_booked:.3f}, M2 {booked[G]}", flush=True)
day = 0
for k, sd in enumerate(SNAP_DAYS):
    with evaluation_randomness(seed * 7 + k):
        while day < sd:
            env.step(); day += 0.25
    fg = env.fgs[G]
    print(f"\n=== seed {seed} day {sd} (doy {int((doy0 + sd) % 365)}): benth B/B0={fg.biomass.sum()/B0[G]:.2f} "
          f"J/B={fg.juvenile.sum()/fg.biomass.sum():.3f}  WC B/B0={env.fgs[P].biomass.sum()/B0[P]:.2f}  "
          f"WC fill={float((env.fgs[P].energy_reserve.sum())/(env.fgs[P].biomass.sum()*env.fgs[P].max_energy_reserve)):.2f}", flush=True)
    for mode in ('clumped', '3.0', 'uniform'):
        e = copy.deepcopy(env)
        if mode != 'clumped': spread(e, mode)
        b0 = {q: float(x) for q, x in e.intake_by_pred_prey.get(P, {}).items()}
        Bp = Bb = 0.0
        with evaluation_randomness(1000 * seed + k):
            for t in range(W):
                Bp += float(e.fgs[P].biomass.sum()); Bb += float(e.fgs[G].biomass.sum()); e.step()
        Bp /= W; Bb /= W
        eat = {q: float(x) - b0.get(q, 0.0) for q, x in e.intake_by_pred_prey.get(P, {}).items()}
        tot = sum(eat.values()); yr = 1460 / W
        ration = tot / Bp * yr; share = eat.get(G, 0) / tot
        m2 = eat.get(G, 0) / Bb * yr
        f_r = ration / ration_booked; f_s = share / share_booked
        f_b = (Bp / B0[P]) / (Bb / B0[G])
        print(f"  {mode:8s} M2={m2:.3f} ({m2/booked[G]:.1f}x) = ration {f_r:.2f} x share {f_s:.2f} x B_WC/B_ben {f_b:.2f} | "
              f"WC ration {ration:.2f}, diet " + ", ".join(f"{q[:5]} {x/tot:.2f}" for q, x in sorted(eat.items(), key=lambda kv: -kv[1])[:5]), flush=True)
