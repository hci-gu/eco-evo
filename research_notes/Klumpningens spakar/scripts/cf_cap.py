"""Scratch (278.6): which cap binds for the predators in the benthivores' cells? Reserve room or Holling ceiling.

Frozen policies, settled snapshots (as cf_lever.py), clumped and Lloyd-3 benthivores, 4 ticks each.
Per predator (water column, seals, porpoises, whiting): in the cells weighted by benthivore biomass and by the
predator's own biomass, (a) the room-cap scale (1 = not capped), (b) the pre-cap potential energy per predator
tonne / its ceiling energy (max_j E_j / h_j), (c) fill.
"""
import sys, json, copy, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from scipy.ndimage import gaussian_filter
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import predation
import inference

D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SNAP = [0, 91, 182, 273]; W = 4
BE = 'benthivorous_fish'; PREDS = ('predatory_fish', 'seals', 'porpoises', 'whiting')
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
lloyd = lambda B: float((B ** 2).sum() * B.size / max(B.sum(), 1e-12) ** 2)
REC = []
_orig = predation._reserve_room_cap

def _rec(env, demand, predator_biomass, reserve_offset):
    out = _orig(env, demand, predator_biomass, reserve_offset)
    pot = (demand * env.energy_gain_mat[:, :, None, None]).sum(axis=1)
    got = (out * env.energy_gain_mat[:, :, None, None]).sum(axis=1)
    REC.append((np.asarray(pot, np.float64), np.asarray(got, np.float64), np.asarray(predator_biomass, np.float64)))
    return out
predation._reserve_room_cap = _rec

def spread3(env):
    fg = env.fgs[BE]
    arrs = {'biomass': fg.biomass, 'energy_reserve': fg.energy_reserve, 'juvenile': fg.juvenile}
    tot = {k: float(v.sum()) for k, v in arrs.items()}
    for sig in np.arange(0.25, 15, 0.25):
        if lloyd(gaussian_filter(fg.biomass.astype(np.float64), sig, mode='reflect')) <= 3.0: break
    for k, v in arrs.items():
        x = gaussian_filter(v.astype(np.float64), sig, mode='reflect'); x *= tot[k] / max(x.sum(), 1e-30)
        setattr(fg, k, x.astype(v.dtype))
    fg.juvenile = np.minimum(fg.juvenile, fg.biomass)

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4
gain = np.asarray(env.energy_gain_mat, np.float64); h = np.asarray(env.handling_time_mat, np.float64)
ceil = {p: float(np.nanmax(np.where(h[env.dm_ids.index(p)] > 0, gain[env.dm_ids.index(p)] / np.where(h[env.dm_ids.index(p)] > 0, h[env.dm_ids.index(p)], 1), 0)))
        for p in PREDS}
print('ceiling energy MJ/t/tick', {p[:5]: round(c, 2) for p, c in ceil.items()}, flush=True)
day = 0
for k, sd in enumerate(SNAP):
    with evaluation_randomness(seed * 7 + k):
        while day < sd:
            env.step(); day += 0.25
    print(f"\n=== seed {seed} day {sd} (doy {int((doy0 + sd) % 365)})", flush=True)
    for mode in ('clumped', '3.0'):
        e = copy.deepcopy(env)
        if mode == '3.0': spread3(e)
        REC.clear(); wts = []
        with evaluation_randomness(1000 * seed + k):
            for t in range(W):
                wts.append(np.asarray(e.fgs[BE].biomass, np.float64).copy()); e.step()
        rows = []
        for p in PREDS:
            i = e.dm_ids.index(p)
            num_b = den_b = 0.0; scl_b = []; pot_b = []; scl_p = []; pot_p = []
            for (pot, got, Bp), wb in zip(REC, wts):
                B = Bp[i]; ok = B > 1e-9
                if not ok.any(): continue
                scale = np.where(pot[i] > 0, got[i] / np.maximum(pot[i], 1e-30), 1.0)
                sat = pot[i] / np.maximum(B, 1e-30) / ceil[p]
                wbB = wb * ok * B   # benthivore cells, weighted by predator tonnes there
                if wbB.sum() > 0:
                    scl_b.append((scale * wbB).sum() / wbB.sum()); pot_b.append((sat * wbB).sum() / wbB.sum())
                scl_p.append((scale * B).sum() / B.sum()); pot_p.append((sat * B).sum() / B.sum())
            fg = e.fgs[p]; fill = float(fg.energy_reserve.sum() / max(fg.biomass.sum() * fg.max_energy_reserve, 1e-30))
            rows.append(f"{p[:5]}: benth cells room {np.mean(scl_b):.2f} demand/ceil {np.mean(pot_b):.2f} | "
                        f"own cells room {np.mean(scl_p):.2f} demand/ceil {np.mean(pot_p):.2f} | fill {fill:.2f}")
        print(f"  {mode:7s} " + "\n          ".join(rows), flush=True)
