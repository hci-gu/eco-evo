"""Scratch (278.7): the two field diagnostics of research_notes/"Trängselns kostnad" on frozen fix104.

(1) Upper tail: the benthivores' biomass share in cells above 20 and 50 t/km2 (a cell's tonnes are t/km2 at the
    reference grid's density scale), max cell / mean, Lloyd; daily over one probe year after the warm-up.
    Field (NS-IBTS GOV, box rectangles, 1991-2025): share above 20 t/km2 0 % (Q1) / 11 % (Q3), above 50 0 %,
    max haul / survey mean 9.5 (Q1) / 13.3 (Q3).
(2) Intake elasticity on own density at field-like crowding: at 4 snapshots the benthivores are spread to Lloyd 3
    (as cf_lever.py) and run 4 ticks; per cell the realised and pre-cap energy intake per tonne, regressed on
    ln(own density / mean) with ln(accessible bed prey density) as the spatial control, weighted by biomass.
    Also in the clumped (trained) state. Field: -0.02 to -0.2 (haddock -0.02..-0.07, large cod / flounder -0.2).
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
BE = 'benthivorous_fish'; BED = ('benthic_infauna', 'mobile_epifauna', 'norway_lobster')
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
lloyd = lambda B: float((B ** 2).sum() * B.size / max(B.sum(), 1e-12) ** 2)
REC = []
_orig = predation._reserve_room_cap
def _rec(env, demand, predator_biomass, reserve_offset):
    out = _orig(env, demand, predator_biomass, reserve_offset)
    i = env.dm_ids.index(BE)
    pot = (demand[i] * env.energy_gain_mat[i][:, None, None]).sum(axis=0)
    got = (out[i] * env.energy_gain_mat[i][:, None, None]).sum(axis=0)
    REC.append((np.asarray(pot, np.float64), np.asarray(got, np.float64)))
    return out
predation._reserve_room_cap = _rec

def bed_prey(env):
    tot = 0.0
    for g in BED:
        fg = env.fgs[g]
        if g == 'benthic_infauna' and fg.params.get('exposed_share'):
            tot = tot + np.asarray(fg.energy_reserve, np.float64)   # refuge: the accessible part A rides here
        else:
            tot = tot + np.asarray(fg.biomass, np.float64)
    return tot

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

def elasticity(rows):
    k = np.concatenate([r[0] for r in rows]); y = np.concatenate([r[1] for r in rows])
    prey = np.concatenate([r[2] for r in rows]); w = np.concatenate([r[3] for r in rows])
    ok = (k > 0) & (y > 0) & (prey > 0) & (w > 0)
    X = np.column_stack([np.ones(ok.sum()), np.log(k[ok]), np.log(prey[ok])])
    sw = np.sqrt(w[ok] / w[ok].sum())
    b0 = np.linalg.lstsq(X[:, :2] * sw[:, None], np.log(y[ok]) * sw, rcond=None)[0][1]
    b1 = np.linalg.lstsq(X * sw[:, None], np.log(y[ok]) * sw, rcond=None)[0][1]
    return b0, b1, float(np.exp(np.average(np.log(k[ok]), weights=w[ok])))

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4

# (1) one probe year, daily
base = copy.deepcopy(env)
tail = []
with evaluation_randomness(seed * 31 + 5):
    for t in range(1, 365 * 4 + 1):
        base.step()
        if t % 4 == 0:
            B = np.asarray(base.fgs[BE].biomass, np.float64); s = B.sum()
            tail.append((B[B > 20].sum() / s, B[B > 50].sum() / s, B.max() / B.mean(), lloyd(B), B.max(), B.mean()))
T = np.array(tail)
print(f"seed {seed}: benthivore mean density {T[:, 5].mean():.2f} t/km2 (cell tonnes at the reference scale)")
for name, sl in (('year', slice(None)), ('Q1 (days 0-89 of doy)', None), ('Q3', None)):
    pass
doys = (doy0 + np.arange(1, len(T) + 1)) % 365
for name, mask in (('year', np.ones(len(T), bool)), ('Q1 Jan-Mar', doys < 90), ('Q3 Jul-Sep', (doys >= 181) & (doys < 273))):
    X = T[mask]
    print(f"  {name:11s}: share > 20 t/km2 {X[:, 0].mean():.2f} (range {X[:, 0].min():.2f}-{X[:, 0].max():.2f}), "
          f"> 50 {X[:, 1].mean():.2f}, max cell {X[:, 4].mean():.0f} t/km2 = {X[:, 2].mean():.0f}x mean, Lloyd {X[:, 3].mean():.0f}",
          flush=True)

# (2) elasticity at Lloyd 3 and as trained
day = 0
rows = {'clumped': [], '3.0': []}
for kq, sd in enumerate(SNAP):
    with evaluation_randomness(seed * 7 + kq):
        while day < sd:
            env.step(); day += 0.25
    for mode in rows:
        e = copy.deepcopy(env)
        if mode == '3.0': spread3(e)
        REC.clear(); state = []
        with evaluation_randomness(1000 * seed + kq):
            for t in range(W):
                B = np.asarray(e.fgs[BE].biomass, np.float64)
                state.append((B.copy(), bed_prey(e))); e.step()
        for (B, prey), (pot, got) in zip(state, REC):
            k = B / B.mean()
            rows[mode].append((k, got / np.maximum(B, 1e-30), prey, B, pot / np.maximum(B, 1e-30)))
for mode, r in rows.items():
    eg0, eg1, kg = elasticity([(a, b, c, d) for a, b, c, d, _ in r])
    ep0, ep1, _ = elasticity([(a, p, c, d) for a, _, c, d, p in r])
    print(f"  elasticity {mode:7s} (biomass-weighted geometric mean k {kg:.1f}): eaten {eg0:+.3f} (with bed-prey control "
          f"{eg1:+.3f}); pre-cap {ep0:+.3f} ({ep1:+.3f})", flush=True)
