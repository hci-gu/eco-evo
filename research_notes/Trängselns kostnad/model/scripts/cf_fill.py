"""Scratch (278.8): what would closing the demersal fill gap do to the swamping payoff and the crowding cost?

Frozen fix104 u1267, pre-279 library, snapshots as cf_lever.py, one-day windows (a payoff readout: the
policies never saw these fills). At each snapshot the reserves are set to the month's field start_fill
(each cell's fill, not the eat share; capping T would be predator_share again):
  fill arm  : 'as trained' | 'predators' (water column 0.71, whiting its month) | 'benthivores' (its month) | 'both'
  benthivores: clumped | 3.0 | field (as cf_lever.py)
Readouts: all-predator and water-column M2 on the benthivores, the saving of clumping, the water column's
room scale in the benthivore cells, and the benthivores' intake elasticity on own density (post-room-cap,
biomass-weighted, clumped and Lloyd-3 arms) and their own room scale.
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
WC, BE, WH = 'predatory_fish', 'benthivorous_fish', 'whiting'
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
lloyd = lambda B: float((B ** 2).sum() * B.size / max(B.sum(), 1e-12) ** 2)
REC = []
_orig = predation._reserve_room_cap
def _rec(env, demand, pb, ro):
    out = _orig(env, demand, pb, ro)
    rows = {}
    for g in (WC, BE):
        i = env.dm_ids.index(g)
        pot = (demand[i] * env.energy_gain_mat[i][:, None, None]).sum(axis=0)
        got = (out[i] * env.energy_gain_mat[i][:, None, None]).sum(axis=0)
        rows[g] = (np.asarray(pot, np.float64), np.asarray(got, np.float64))
    REC.append(rows)
    return out
predation._reserve_room_cap = _rec

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4
iw = env.dm_ids.index(WC)
from lib.diagnostics.field_diet import field_cell_weights, FIELD_STAGE_BED_SHARES
import yaml
from lib.world import habitat
man = yaml.safe_load(open(cfg['project']))
depth = habitat.load_depth_map(man, cfg['project'], (env.H, env.W))
ad_field = field_cell_weights(depth, FIELD_STAGE_BED_SHARES[BE][0])
juv_field = np.asarray(env.fgs[BE].recruit_weight_map, np.float64); juv_field /= juv_field.sum()

def set_fill(e, g, month):
    fg = e.fgs[g]; f = float(fg.params['start_fill'][month])
    fg.energy_reserve = (f * fg.biomass.astype(np.float64) * fg.max_energy_reserve).astype(fg.energy_reserve.dtype)

def place(e, mode):
    fg = e.fgs[BE]
    arrs = {'biomass': fg.biomass, 'energy_reserve': fg.energy_reserve, 'juvenile': fg.juvenile}
    tot = {k: float(v.sum()) for k, v in arrs.items()}
    if mode == 'field':
        J = tot['juvenile'] * juv_field; B = J + (tot['biomass'] - tot['juvenile']) * ad_field
        new = {'biomass': B, 'juvenile': J, 'energy_reserve': tot['energy_reserve'] * B / B.sum()}
    else:
        for sig in np.arange(0.25, 15, 0.25):
            if lloyd(gaussian_filter(fg.biomass.astype(np.float64), sig, mode='reflect')) <= 3.0: break
        new = {k: gaussian_filter(v.astype(np.float64), sig, mode='reflect') for k, v in arrs.items()}
        new = {k: x * tot[k] / max(x.sum(), 1e-30) for k, x in new.items()}
    for k, x in new.items():
        setattr(fg, k, np.maximum(x, 0).astype(arrs[k].dtype))
    fg.juvenile = np.minimum(fg.juvenile, fg.biomass)

def elast(k, y, w):
    ok = (k > 0) & (y > 0) & (w > 0)
    X = np.column_stack([np.ones(ok.sum()), np.log(k[ok])]); sw = np.sqrt(w[ok] / w[ok].sum())
    return float(np.linalg.lstsq(X * sw[:, None], np.log(y[ok]) * sw, rcond=None)[0][1])

ARMS = {'as trained': (), 'predators': (WC, WH), 'benthivores': (BE,), 'both': (WC, WH, BE)}
day = 0
for kq, sd in enumerate(SNAP):
    with evaluation_randomness(seed * 7 + kq):
        while day < sd:
            env.step(); day += 0.25
    doy = int((doy0 + sd) % 365); month = min(11, int(doy / 30.42))
    print(f"\n=== seed {seed} day {sd} (doy {doy}, month {month + 1}); field fill benth "
          f"{env.fgs[BE].params['start_fill'][month]}, whiting {env.fgs[WH].params['start_fill'][month]}, WC 0.71", flush=True)
    for arm, groups in ARMS.items():
        base = None; parts = []
        for bmode in ('clumped', '3.0', 'field'):
            e = copy.deepcopy(env)
            for g in groups: set_fill(e, g, month)
            if bmode != 'clumped': place(e, bmode)
            pre = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
            REC.clear(); Bs = []; Bsum = 0.0
            with evaluation_randomness(1000 * seed + kq):
                for t in range(W):
                    B = np.asarray(e.fgs[BE].biomass, np.float64); Bs.append(B.copy()); Bsum += B.sum(); e.step()
            Bm = Bsum / W; yr = 1460 / W
            post = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
            m_all = sum(post[p] - pre[p] for p in post) / Bm * yr; m_wc = (post[WC] - pre[WC]) / Bm * yr
            # WC room scale in benthivore cells (weighted by benthivores x WC there); benthivore room scale and elasticity
            num = den = 0.0; ks = []; ys = []; ws = []; bnum = bden = 0.0
            for rows, B in zip(REC, Bs):
                pw, gw = rows[WC]; pb, gb = rows[BE]
                wcB = np.asarray(e.fgs[WC].biomass, np.float64)
                wt = B * (pw > 0)
                num += (np.where(pw > 0, gw / np.maximum(pw, 1e-30), 1) * wt).sum(); den += wt.sum()
                bnum += gb.sum(); bden += pb.sum()
                ks.append(B / B.mean()); ys.append(gb / np.maximum(B, 1e-30)); ws.append(B)
            el = elast(np.concatenate(ks), np.concatenate(ys), np.concatenate(ws))
            if base is None: base = m_all
            parts.append(f"{bmode} all {m_all:.3f} (save {m_all - base:+.3f}) WC {m_wc:.3f} | WCroom {num / max(den, 1e-30):.2f} "
                         f"Broom {bnum / max(bden, 1e-30):.2f} el {el:+.3f}")
        print(f"  {arm:12s}: " + " || ".join(parts), flush=True)
