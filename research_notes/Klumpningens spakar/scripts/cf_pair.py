"""Scratch (278.9): how does the water column's excess on spread benthivores split, and how much of it would
fixing the pair remove once the reserve-room cap refills the meal?

Frozen fix104 u1267, pre-279 library, snapshots and one-day windows as cf_lever.py (WC trained placement).
Arms:
  pair scale  : 1 | 0.5 | 0.42 (x on the WC -> benthivores max_intake_mat; 0.42 ~ 1/2.4, the pre-279 pair at its
                booking by the gate's equal-cell ratio) - a counterfactual, never a library value
  planktivores: trained | uniform (B, reserve, J even; the WC's missing prey back, as wc_overlap.py)
  benthivores : clumped | 3.0 | field (as cf_lever.py)
Readouts: WC M2 on benthivores (x booked), all-predator M2, the saving of clumping (spread - clumped, all predators),
the WC's ration in tonnes and in assimilated MJ (post room cap) per tonne per year against the booked diet, and the
benthivores' share of the WC's tonnes.
"""
import sys, json, copy, numpy as np, yaml
sys.path.insert(0, '/home/simon/eco-evo')
from scipy.ndimage import gaussian_filter
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.world import habitat
from lib.diagnostics.field_diet import field_cell_weights, FIELD_STAGE_BED_SHARES
from lib.environments.ecosystem_env import predation
import inference

D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SNAP = [0, 91, 182, 273]; W = 4
WC, BE, PL = 'predatory_fish', 'benthivorous_fish', 'planktivorous_fish'
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
lloyd = lambda B: float((B ** 2).sum() * B.size / max(B.sum(), 1e-12) ** 2)
MJ = [0.0]
_orig = predation._reserve_room_cap
def _rec(env, demand, pb, ro):
    out = _orig(env, demand, pb, ro)
    i = env.dm_ids.index(WC)
    MJ[0] += float((out[i] * env.energy_gain_mat[i][:, None, None]).sum())
    return out
predation._reserve_room_cap = _rec

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    B0 = {g: float(env.fgs[g].biomass.sum()) for g in env.fgs}
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4
man = yaml.safe_load(open(cfg['project']))
depth = habitat.load_depth_map(man, cfg['project'], (env.H, env.W))
iw = env.dm_ids.index(WC); jb = env.global_fg_order.index(BE)
gain = {q: float(env.energy_gain_mat[iw][env.global_fg_order.index(q)]) for q in env.global_fg_order}
inter = env.fgs[WC].params['interaction']
booked = {k.split('_preys_on_')[1]: float(v.get('predation_mortality', 0) or 0) for k, v in inter.items()
          if k.startswith(WC + '_preys_on_') and v.get('preys_on')}
flow = {q: booked[q] * B0[q] for q in booked if q in B0}
r_t = sum(flow.values()) / B0[WC]; r_mj = sum(flow[q] * gain[q] for q in flow) / B0[WC]
print(f"booked WC: ration {r_t:.2f} t/t/yr, {r_mj:.0f} MJ/t/yr assimilated; benth share {flow[BE] / sum(flow.values()):.3f}; "
      f"M2 {booked[BE]}; gain MJ/t " + ", ".join(f"{q[:5]} {gain[q]:.0f}" for q in flow), flush=True)
ad_field = field_cell_weights(depth, FIELD_STAGE_BED_SHARES[BE][0])
juv_field = np.asarray(env.fgs[BE].recruit_weight_map, np.float64); juv_field /= juv_field.sum()

def set_arrays(fg, new):
    for k, val in new.items():
        old = getattr(fg, k); setattr(fg, k, np.maximum(val, 0).astype(old.dtype))
    if getattr(fg, 'juvenile', None) is not None:
        fg.juvenile = np.minimum(fg.juvenile, fg.biomass)

def place_benth(e, mode):
    fg = e.fgs[BE]
    arrs = {'biomass': fg.biomass, 'energy_reserve': fg.energy_reserve, 'juvenile': fg.juvenile}
    tot = {k: float(v.sum()) for k, v in arrs.items()}
    if mode == 'field':
        J = tot['juvenile'] * juv_field; B = J + (tot['biomass'] - tot['juvenile']) * ad_field
        new = {'biomass': B, 'juvenile': J, 'energy_reserve': tot['energy_reserve'] * B / B.sum()}
    else:
        for sig in np.arange(0.25, 15, 0.25):
            if lloyd(gaussian_filter(fg.biomass.astype(np.float64), sig, mode='reflect')) <= float(mode): break
        new = {k: gaussian_filter(v.astype(np.float64), sig, mode='reflect') for k, v in arrs.items()}
        new = {k: x * tot[k] / max(x.sum(), 1e-30) for k, x in new.items()}
    set_arrays(fg, new)

def plank_uniform(e):
    fg = e.fgs[PL]
    new = {}
    for k in ('biomass', 'energy_reserve', 'juvenile'):
        a = getattr(fg, k, None)
        if a is not None: new[k] = np.full(a.shape, float(a.sum()) / a.size)
    set_arrays(fg, new)

day = 0
for kq, sd in enumerate(SNAP):
    with evaluation_randomness(seed * 7 + kq):
        while day < sd:
            env.step(); day += 0.25
    print(f"\n=== seed {seed} day {sd} (doy {int((doy0 + sd) % 365)}): WC B/B0 {env.fgs[WC].biomass.sum() / B0[WC]:.2f}, "
          f"benth Lloyd {lloyd(env.fgs[BE].biomass):.0f}, plank Lloyd {lloyd(env.fgs[PL].biomass):.1f}", flush=True)
    for pl in ('trained', 'uniform'):
        for s in (1.0, 0.5, 0.42):
            base = None; parts = []
            for bmode in ('clumped', '3.0', 'field'):
                e = copy.deepcopy(env)
                e.max_intake_mat = e.max_intake_mat.copy(); e.max_intake_mat[iw, jb] *= s
                if pl == 'uniform': plank_uniform(e)
                if bmode != 'clumped': place_benth(e, bmode)
                pre = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
                pw = {q: float(x) for q, x in e.intake_by_pred_prey.get(WC, {}).items()}
                Bb = Bw = 0.0; MJ[0] = 0.0
                with evaluation_randomness(1000 * seed + kq):
                    for t in range(W):
                        Bb += float(e.fgs[BE].biomass.sum()); Bw += float(e.fgs[WC].biomass.sum()); e.step()
                Bb /= W; Bw /= W; yr = 1460 / W
                post = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
                eat = {q: float(x) - pw.get(q, 0.0) for q, x in e.intake_by_pred_prey.get(WC, {}).items()}
                tot = sum(eat.values())
                m_wc = (post[WC] - pre[WC]) / Bb * yr; m_all = sum(post[p] - pre[p] for p in post) / Bb * yr
                if base is None: base = m_all
                parts.append(f"{bmode} WC {m_wc:.3f} ({m_wc / booked[BE]:4.1f}x) all {m_all:.3f} save {m_all - base:+.3f} "
                             f"rat {tot / Bw * yr:.2f}t {MJ[0] / Bw * yr / r_mj:.2f}MJ sh {eat.get(BE, 0) / max(tot, 1e-30):.2f} "
                             f"pl {eat.get(PL, 0) / max(tot, 1e-30):.2f}")
            print(f"  plank {pl:7s} pair x{s:4.2f}: " + " | ".join(parts), flush=True)
