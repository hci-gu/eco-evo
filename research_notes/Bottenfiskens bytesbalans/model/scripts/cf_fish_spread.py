"""Cost vs benefit of clumping at a given infauna refuge (resume 282 follow-up). Frozen policies.

Run the world 2 years after the warm-up (same random context as refuge_clump.py, so the state is that
script's year-3 start), then at 4 snapshots in year 3 copy the world and place the benthivores
  clumped (as the policies left them) | 3.0 (Gaussian spread to Lloyd 3, 278.6) | field (adults at GOV
  bed-class shares, J at the recruit shape, 278.6)
and run 4 days. Readouts per arm, over day 1 and days 1-4:
  M2_all  = tonnes of benthivores eaten by every predator / mean benthivore B, /yr  (benefit side, 278.6)
  eat     = tonnes the benthivores eat per tonne, /yr (realised book, after both caps)
  MJ      = net assimilated MJ per tonne per yr (book x energy_gain_mat at the reference fill)
  infauna share of the tonnes eaten, Lloyd at the end of the window
Usage: cf_refuge_spread.py POLICY_DIR SEED LIBRARY
"""
import sys, json, copy, numpy as np, yaml
sys.path.insert(0, '/home/simon/eco-evo')
from scipy.ndimage import gaussian_filter
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.world import habitat
from lib.diagnostics.field_diet import bed_class, field_cell_weights, FIELD_STAGE_BED_SHARES
import inference

D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
BE = 'benthivorous_fish'
PRE_DAYS = 730
SNAP = [0, 91, 182, 273]          # days into year 3
W1, W4 = 4, 16                    # ticks: one day, four days
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
lloyd = lambda B: float((B ** 2).sum() * B.size / max(B.sum(), 1e-12) ** 2)

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    B0 = float(env.fgs[BE].biomass.sum())
    settle(env, cfg['warmup_ticks'], seed=seed)
man = yaml.safe_load(open(cfg['project']))
depth = habitat.load_depth_map(man, cfg['project'], (env.H, env.W))
ad_field = field_cell_weights(depth, FIELD_STAGE_BED_SHARES[BE][0])
juv_field = np.asarray(env.fgs[BE].recruit_weight_map, dtype=np.float64); juv_field = juv_field / juv_field.sum()
order = list(env.global_fg_order); i_be = env.dm_ids.index(BE)
inf = env.fgs['benthic_infauna']
print(f"seed {seed}; library {LIB}: exposed_share {inf.exposed_share}, k {inf.refuge_exchange_rate * 1460:.2f} /yr", flush=True)

def set_arrays(fg, new):
    for k, val in new.items():
        old = getattr(fg, k); setattr(fg, k, np.maximum(val, 0).astype(old.dtype))
    fg.juvenile = np.minimum(fg.juvenile, fg.biomass)

def place(e, mode):
    fg = e.fgs[BE]
    arrs = {'biomass': fg.biomass, 'energy_reserve': fg.energy_reserve, 'juvenile': fg.juvenile}
    tot = {k: float(v.sum()) for k, v in arrs.items()}
    if mode == 'field':
        J = tot['juvenile'] * juv_field; A = (tot['biomass'] - tot['juvenile']) * ad_field; B = J + A
        new = {'biomass': B, 'juvenile': J, 'energy_reserve': tot['energy_reserve'] * B / B.sum()}
    else:
        target = float(mode)
        for sig in np.arange(0.25, 15, 0.25):
            if lloyd(gaussian_filter(fg.biomass.astype(np.float64), sig, mode='reflect')) <= target: break
        new = {k: gaussian_filter(v.astype(np.float64), sig, mode='reflect') for k, v in arrs.items()}
        new = {k: x * tot[k] / max(x.sum(), 1e-30) for k, x in new.items()}
    set_arrays(fg, new)

BEDS = ['benthic_infauna', 'mobile_epifauna', 'norway_lobster']
def fresh_beds(e):
    """Every bed cell at the grid-mean fill of its K (a clump that has just arrived on unvisited beds)."""
    for b in BEDS:
        f = e.fgs[b]; K = np.asarray(f.capacity_map, dtype=np.float64); ok = K > 0
        if getattr(f, 'exposed_share', None) is not None:
            KA = K * f.exposed_share
            A = np.clip(f.energy_reserve, 0, f.biomass).astype(np.float64)
            r = A[ok].sum() / KA[ok].sum()
            f.energy_reserve = np.minimum(r * KA, f.biomass).astype(f.energy_reserve.dtype)
        else:
            x = f.biomass.astype(np.float64); r = x[ok].sum() / K[ok].sum()
            f.biomass = (r * K).astype(f.biomass.dtype)

def book(e):
    eaten_on = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
    own = e.intake_by_pred_prey.get(BE, {})
    t = {q: float(x) for q, x in own.items()}
    return eaten_on, t

def diff(a, b):
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}

def readout(pre, post, Bm, ticks):
    yr = 1460 / ticks
    m2 = sum(diff(pre[0], post[0]).values()) / Bm * yr
    t = diff(pre[1], post[1])
    eat = sum(t.values()) / Bm * yr
    mj = sum(x * float(env.energy_gain_mat[i_be, order.index(q)]) for q, x in t.items()) / Bm * yr
    sh = t.get('benthic_infauna', 0.0) / max(sum(t.values()), 1e-30)
    tt = max(sum(t.values()), 1e-30)
    return m2, eat, mj, sh, t.get('planktivorous_fish', 0.0) / tt, t.get('whiting', 0.0) / tt

rows = []
day = 0.0
with evaluation_randomness(seed * 13 + 5):
    for t in range(PRE_DAYS * 4):
        env.step()
    for k, sd in enumerate(SNAP):
        while day < sd:
            env.step(); day += 0.25
        fg = env.fgs[BE]
        A = np.clip(inf.energy_reserve, 0, inf.biomass) / np.maximum(inf.capacity_map * inf.exposed_share, 1e-12)
        print(f"\n=== year 3 day {sd}: benth B/B0 {fg.biomass.sum() / B0:.2f}, Lloyd {lloyd(fg.biomass):.0f}, "
              f"grid A/K_A {A.mean():.3f}, A/K_A met by the mean benthivore tonne "
              f"{(A * fg.biomass).sum() / fg.biomass.sum():.3f}", flush=True)
        base = None
        for mode in ('clumped', '3.0', 'field', 'clumped_fresh', '3.0_fresh'):
            e = copy.deepcopy(env)
            if mode.startswith('3.0'): place(e, '3.0')
            elif mode == 'field': place(e, 'field')
            if mode.endswith('_fresh'): fresh_beds(e)
            efg = e.fgs[BE]
            L0 = lloyd(efg.biomass)
            Ae = np.clip(e.fgs['benthic_infauna'].energy_reserve, 0, e.fgs['benthic_infauna'].biomass) / \
                np.maximum(inf.capacity_map * inf.exposed_share, 1e-12)
            met = float((Ae * efg.biomass).sum() / efg.biomass.sum())
            pre = book(e); Bs = 0.0; out = {}
            with evaluation_randomness(1000 * seed + k):
                for tt in range(1, W4 + 1):
                    Bs += float(efg.biomass.sum()); e.step()
                    if tt in (W1, W4):
                        out[tt] = readout(pre, book(e), Bs / tt, tt)
            L1 = lloyd(efg.biomass)
            r = dict(snap=sd, mode=mode, L0=L0, L1=L1, met=met, d1=out[W1], d4=out[W4])
            rows.append(r)
            print(f"  {mode:7s} Lloyd {L0:5.1f} -> {L1:5.1f} (4 d); A/K_A met {met:.3f} | "
                  + " | ".join(f"{lab}: M2_all {o[0]:.3f} eat {o[1]:.2f} t/t/yr MJ {o[2]:6.0f} inf {o[3]:.2f} plk {o[4]:.4f} wht {o[5]:.4f}"
                               for lab, o in (('day 1', out[W1]), ('days 1-4', out[W4]))), flush=True)

print("\n=== means over the 4 snapshots (spread minus clumped; benefit = extra M2 avoided by clumping,"
      " cost = intake given up by clumping; cost in /yr = 0.3 x d eat, the 278.6 growth-ration rule)")
for mode, ref in (('3.0', 'clumped'), ('field', 'clumped'), ('3.0_fresh', 'clumped_fresh')):
    for w, lab in (('d1', 'day 1'), ('d4', 'days 1-4')):
        mm = lambda md, i: np.mean([r[w][i] for r in rows if r['mode'] == md])
        dm2 = mm(mode, 0) - mm(ref, 0); de = mm(mode, 1) - mm(ref, 1); dmj = mm(mode, 2) - mm(ref, 2)
        ec = mm(ref, 1); mjc = mm(ref, 2)
        print(f"  {mode:9s} vs {ref:13s} {lab:8s}: benefit of clumping (M2 saved) {dm2:+.3f} /yr | cost of clumping: eat {de:+.2f} t/t/yr "
              f"({de / ec:+.0%} of clumped {ec:.2f}), MJ {dmj:+.0f} ({dmj / mjc:+.0%}) -> 0.3 x d eat = {0.3 * de:+.3f} /yr")
