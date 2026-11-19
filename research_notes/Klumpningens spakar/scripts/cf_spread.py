import sys, json, copy, numpy as np, torch
from scipy.ndimage import gaussian_filter
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
import inference
D, seed = sys.argv[1], int(sys.argv[2])
G = sys.argv[3] if len(sys.argv) > 3 else 'benthivorous_fish'
SNAP_DAYS = [0, 91, 182, 273]; W = 120; MARKS = (4, 40, 120)
cfg = json.load(open(D + '/config.json'))
lloyd = lambda B: float((B**2).sum() * B.size / max(B.sum(), 1e-12)**2)

def spread(env, mode):
    fg = env.fgs[G]
    arrs = {'biomass': fg.biomass, 'energy_reserve': fg.energy_reserve}
    if getattr(fg, 'juvenile', None) is not None: arrs['juvenile'] = fg.juvenile
    tot = {k: float(v.sum()) for k, v in arrs.items()}
    if mode == 'uniform':
        new = {k: np.full_like(v, tot[k] / v.size) for k, v in arrs.items()}
    else:
        target = float(mode)
        for sig in np.arange(0.25, 15, 0.25):
            nb = gaussian_filter(fg.biomass.astype(np.float64), sig, mode='reflect')
            if lloyd(nb) <= target: break
        new = {k: gaussian_filter(v.astype(np.float64), sig, mode='reflect') for k, v in arrs.items()}
    for k, v in new.items():
        v = np.maximum(v, 0); v *= tot[k] / max(v.sum(), 1e-30)
        setattr(fg, k, v.astype(arrs[k].dtype))
    if 'juvenile' in new:
        fg.juvenile = np.minimum(fg.juvenile, fg.biomass)

def books(env):
    fg = env.fgs[G]; ec = float(fg.params.get('energy_content', 0.0))
    eat = env.intake_by_pred_prey.get(G, {})
    pred = {p: float(env.intake_by_pred_prey.get(p, {}).get(G, 0.0)) for p in env.dm_ids if p != G}
    return dict(B=float(fg.biomass.sum()), E=float((fg.biomass * ec).sum() + fg.energy_reserve.sum()),
                eat=float(sum(eat.values())), eat_by={k: float(x) for k, x in eat.items()},
                pred=pred, starv=float(env.loss_starvation.get(G, 0.0)),
                seal_eat=float(sum(env.intake_by_pred_prey.get('seals', {}).values())),
                seal_B=float(env.fgs['seals'].biomass.sum()))

def run_arm(env0, mode, k):
    env = copy.deepcopy(env0)
    if mode != 'clumped': spread(env, mode)
    L0 = lloyd(env.fgs[G].biomass)
    out = {}; b0 = books(env); logE = 0.0; Bsum = 0.0; Ssum = 0.0
    prev = b0
    with evaluation_randomness(1000 * seed + k):
        for t in range(1, W + 1):
            env.step()
            cur = books(env)
            X = float(getattr(env, 'tick_exchange_energy', {}).get(G, 0.0))
            logE += np.log(max(cur['E'] - X, 1e-12) / max(prev['E'], 1e-12))
            Bsum += 0.5 * (cur['B'] + prev['B']); Ssum += 0.5 * (cur['seal_B'] + prev['seal_B']); prev = cur
            if t in MARKS:
                days = t / 4; Bm = Bsum / t; Sm = Ssum / t
                out[t] = dict(lloyd=lloyd(env.fgs[G].biomass), dlogE=logE,
                              eat=(cur['eat'] - b0['eat']) / Bm / days,
                              pred=sum(cur['pred'].values()) - sum(b0['pred'].values()),
                              pred_by={p: (cur['pred'][p] - b0['pred'][p]) / Bm / days * 365 for p in cur['pred']},
                              starv=(cur['starv'] - b0['starv']) / Bm / days * 365,
                              seal_eat=(cur['seal_eat'] - b0['seal_eat']) / Sm / days * 365,
                              Bm=Bm, days=days)
                out[t]['pred'] = out[t]['pred'] / Bm / days * 365
    return L0, out

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4
day = 0
for k, sd in enumerate(SNAP_DAYS):
    with evaluation_randomness(seed * 7 + k):
        while day < sd:
            env.step(); day += 0.25
    print(f"\n=== {G} seed {seed} snapshot day {sd} (doy {int((doy0 + sd) % 365)})", flush=True)
    for mode in ('clumped', '3.0', 'uniform'):
        L0, out = run_arm(env, mode, k)
        for t in MARKS:
            o = out[t]
            top = sorted(o['pred_by'].items(), key=lambda kv: -kv[1])[:3]
            print(f"{mode:8s} L0={L0:5.1f} d{o['days']:5.1f}: lloyd={o['lloyd']:5.1f} eat={o['eat']:.4f} t/t/d  "
                  f"pred={o['pred']:.3f}/yr ({', '.join(f'{p[:5]} {x:.3f}' for p, x in top)})  starv={o['starv']:.3f}/yr  "
                  f"dlogE={o['dlogE']:+.4f}  seal_ration={o['seal_eat']:.1f}", flush=True)
