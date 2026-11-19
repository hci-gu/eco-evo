"""Scratch (278.6): which lever removes the benthivores' swamping payoff against the water column?

Frozen policies, one-day windows from settled snapshots (as cf_wc.py). Arms:
  WC placement : 'trained' | 'field' (each bed class rescaled to GOV saithe shares, within-class pattern kept)
  pair         : 'lib' | 'dark' (WC -> benthivores at the no-daylight bound: depth mult 1, no diel swing, a x 0.31)
  benthivores  : 'clumped' | '3.0' (Gaussian spread to Lloyd 3) | 'uniform' | 'field' (adults at GOV class
                 shares, J at the recruit shape, each even within its class)
Readouts: WC -> benthivore M2 (x booked), all-predator M2, payoff = M2(arm)/M2(clumped) in the same WC/pair cell,
and the WC density met by the mean benthivore tonne (B, J) against the class-only expectation.
"""
import sys, json, copy, numpy as np, yaml
sys.path.insert(0, '/home/simon/eco-evo')
from scipy.ndimage import gaussian_filter
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.world import habitat
from lib.diagnostics.field_diet import bed_class, field_cell_weights, FIELD_BED_SHARES, FIELD_STAGE_BED_SHARES
import inference

D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SNAP = [0, 91, 182, 273]; W = 4; DARK_A = 0.31
WC, BE = 'predatory_fish', 'benthivorous_fish'
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
lloyd = lambda B: float((B ** 2).sum() * B.size / max(B.sum(), 1e-12) ** 2)

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    B0 = {g: float(env.fgs[g].biomass.sum()) for g in env.fgs}
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4
man = yaml.safe_load(open(cfg['project']))
depth = habitat.load_depth_map(man, cfg['project'], (env.H, env.W))
cls = bed_class(depth)
iw = env.dm_ids.index(WC); jb = env.global_fg_order.index(BE)
booked = float(env.fgs[WC].params['interaction'][f'{WC}_preys_on_{BE}']['predation_mortality'])
wc_field_sh = FIELD_BED_SHARES[WC][0]
ad_field = field_cell_weights(depth, FIELD_STAGE_BED_SHARES[BE][0])
juv_field = np.asarray(env.fgs[BE].recruit_weight_map, dtype=np.float64); juv_field = juv_field / juv_field.sum()
print(f"library {LIB}\nbooked WC->benth M2 {booked}; cells per class {[round(float((cls == k).mean()), 3) for k in range(4)]}; "
      f"pair depth mult per class {[round(float(np.asarray(env.pair_depth_mult[iw, jb])[cls == k].mean()), 2) for k in range(4)]}",
      flush=True)

def shares(x):
    s = x.sum(); return '/'.join(f"{x[cls == k].sum() / s:.2f}" for k in range(4))

def set_arrays(fg, new):
    for k, val in new.items():
        old = getattr(fg, k); setattr(fg, k, np.maximum(val, 0).astype(old.dtype))
    if getattr(fg, 'juvenile', None) is not None:
        fg.juvenile = np.minimum(fg.juvenile, fg.biomass)

def place_benth(env, mode):
    fg = env.fgs[BE]
    arrs = {'biomass': fg.biomass, 'energy_reserve': fg.energy_reserve, 'juvenile': fg.juvenile}
    tot = {k: float(v.sum()) for k, v in arrs.items()}
    if mode == 'uniform':
        new = {k: np.full(v.shape, tot[k] / v.size) for k, v in arrs.items()}
    elif mode == 'field':
        J = tot['juvenile'] * juv_field; A = (tot['biomass'] - tot['juvenile']) * ad_field
        B = J + A
        new = {'biomass': B, 'juvenile': J, 'energy_reserve': tot['energy_reserve'] * B / B.sum()}
    else:
        target = float(mode)
        for sig in np.arange(0.25, 15, 0.25):
            if lloyd(gaussian_filter(fg.biomass.astype(np.float64), sig, mode='reflect')) <= target: break
        new = {k: gaussian_filter(v.astype(np.float64), sig, mode='reflect') for k, v in arrs.items()}
        new = {k: x * tot[k] / max(x.sum(), 1e-30) for k, x in new.items()}
    set_arrays(fg, new)

def place_wc_field(env):
    fg = env.fgs[WC]
    B = fg.biomass.astype(np.float64); R = fg.energy_reserve.astype(np.float64)
    tot = B.sum(); scale = np.zeros_like(B); newB = np.zeros_like(B); newR = np.zeros_like(R)
    fill = R.sum() / max(B.sum(), 1e-30)
    for k in range(4):
        cells = cls == k
        if not cells.any(): continue
        target = wc_field_sh[k] * tot
        bk = B[cells].sum()
        if bk > 1e-9 * tot:
            newB[cells] = B[cells] * target / bk; newR[cells] = R[cells] * target / bk
        else:
            newB[cells] = target / cells.sum(); newR[cells] = newB[cells] * fill
    s = tot / newB.sum(); set_arrays(fg, {'biomass': newB * s, 'energy_reserve': newR * s})

def dark_pair(env):
    env.max_intake_mat = env.max_intake_mat.copy(); env.max_intake_mat[iw, jb] *= DARK_A
    if env.light_mult_table is not None:
        env.light_mult_table = env.light_mult_table.copy(); env.light_mult_table[:, iw, jb] = 1.0
    env.pair_depth_mult = env.pair_depth_mult.copy(); env.pair_depth_mult[iw, jb] = 1.0

def met(wc, prey):   # WC density met by the mean prey tonne / box mean WC
    return float((wc * prey).sum() / prey.sum() / wc.mean())

def met_class(wc, prey):   # the same from the class shares alone (even within class)
    dens = np.zeros_like(wc)
    for k in range(4):
        c = cls == k
        if c.any(): dens[c] = wc[c].sum() / c.sum()
    return met(dens, prey)

day = 0
for k, sd in enumerate(SNAP):
    with evaluation_randomness(seed * 7 + k):
        while day < sd:
            env.step(); day += 0.25
    wc = env.fgs[WC].biomass.astype(np.float64); be = env.fgs[BE]
    Bb = be.biomass.astype(np.float64); J = be.juvenile.astype(np.float64)
    print(f"\n=== seed {seed} day {sd} (doy {int((doy0 + sd) % 365)}): benth B/B0 {Bb.sum() / B0[BE]:.2f} J/B {J.sum() / Bb.sum():.3f} "
          f"Lloyd {lloyd(Bb):.0f}; WC B/B0 {wc.sum() / B0[WC]:.2f} Lloyd {lloyd(wc):.1f}", flush=True)
    print(f"  classes WC {shares(wc)} (field {'/'.join(str(x) for x in wc_field_sh)})  benth B {shares(Bb)}  J {shares(J)}")
    print(f"  predator response: WC met per benth tonne {met(wc, Bb):.2f} (class-only {met_class(wc, Bb):.2f}), "
          f"per J tonne {met(wc, J):.2f} (class-only {met_class(wc, J):.2f}); 1 = no association", flush=True)
    for wmode in ('trained', 'field'):
        for pmode in ('lib', 'dark'):
            base = None; row = []
            for bmode in ('clumped', '3.0', 'uniform', 'field'):
                e = copy.deepcopy(env)
                if wmode == 'field': place_wc_field(e)
                if pmode == 'dark': dark_pair(e)
                if bmode != 'clumped': place_benth(e, bmode)
                L = lloyd(e.fgs[BE].biomass)
                fw = met(e.fgs[WC].biomass.astype(np.float64), e.fgs[BE].biomass.astype(np.float64))
                pre = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
                Bsum = 0.0
                with evaluation_randomness(1000 * seed + k):
                    for t in range(W):
                        Bsum += float(e.fgs[BE].biomass.sum()); e.step()
                Bm = Bsum / W; yr = 1460 / W
                post = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
                m_wc = (post[WC] - pre[WC]) / Bm * yr
                m_all = sum(post[p] - pre[p] for p in post) / Bm * yr
                if base is None: base = (m_wc, m_all)
                row.append(f"{bmode} L{L:4.1f} met{fw:4.2f} WC {m_wc:.3f} ({m_wc / booked:4.1f}x, pay {m_wc / max(base[0], 1e-9):4.1f}) "
                           f"all {m_all:.3f} (pay {m_all / max(base[1], 1e-9):4.1f})")
            print(f"  WC {wmode:7s} pair {pmode:4s}: " + " | ".join(row), flush=True)
