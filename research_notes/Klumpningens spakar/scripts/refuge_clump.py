"""Does a smaller / slower infauna refuge make packing cost the benthivores food? Frozen policies, 3 years.

Variant of research_notes/"Interferens hos bottenfisk" scripts/bed_depletion_under_clumps.py (278.3) with
the library as an argument and per-year readouts. Daily samples at the predation step.
  k     = own density / grid-mean density
  pot   = demand per tonne before the reserve-room cap, t/t/d; eat = after it
  elasticity = biomass-weighted slope of log(per tonne) on log(k) across occupied cells, mean over days
Usage: refuge_clump.py POLICY_DIR SEED YEARS LIBRARY
"""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import predation, interactions
import inference

D, seed, YEARS, LIB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
G = 'benthivorous_fish'
BEDS = ['benthic_infauna', 'mobile_epifauna', 'norway_lobster']
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB

cap = {}
orig_cap = predation._reserve_room_cap
def hooked(env, demand, predator_biomass, reserve_offset):
    out = orig_cap(env, demand, predator_biomass, reserve_offset)
    if cap.get('on'):
        i = env.dm_ids.index(G)
        cap['pre'] = demand[i].astype(np.float64).copy()
        cap['post'] = out[i].astype(np.float64).copy()
        cap['B'] = predator_biomass[i].astype(np.float64).copy()
    return out
predation._reserve_room_cap = hooked

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    inf0 = float(env.fgs['benthic_infauna'].biomass.sum())
    ben0 = float(env.fgs[G].biomass.sum())
    settle(env, cfg['warmup_ticks'], seed=seed)

order = list(env.global_fg_order)
bed_j = {b: order.index(b) for b in BEDS if b in order}
benthic_cols = list(bed_j.values())
j_inf = bed_j['benthic_infauna']
fg = env.fgs[G]; inf = env.fgs['benthic_infauna']
phi = float(inf.exposed_share)
tpd = 4

def ratio(b):
    f = env.fgs[b]; K = f.capacity_map
    x = f.biomass.astype(np.float64)
    if getattr(f, 'exposed_share', None) is not None:
        x = np.clip(f.energy_reserve, 0, f.biomass).astype(np.float64); K = K * f.exposed_share
    return x / np.maximum(K, 1e-12)

BINS = [0, 0.5, 2, 8, 32, 1e9]; LAB = ['<0.5', '0.5-2', '2-8', '8-32', '>32']
def new_year():
    return dict(acc={l: dict(w=0.0, pot=0.0, eat=0.0, fill=0.0, inf=0.0, epi=0.0, infA=0.0, infshare=0.0)
                     for l in LAB},
                se=[], sp=[], lloyd=[], infK=[], infB=[], benB=[], infshare=[], runs=[], fill=[],
                A_hot_min=[])
Y = [new_year() for _ in range(YEARS)]
visit = None
i_g = env.dm_ids.index(G)
a_now = interactions.attack_rate(env)
hs_inf = 1.0 / (float(a_now[i_g, j_inf]) * float(env.handling_time_mat[i_g, j_inf]))

with evaluation_randomness(seed * 13 + 5):
    for t in range(YEARS * 365 * tpd):
        sample = (t % tpd == 0)
        cap['on'] = sample
        if sample:
            r_inf = ratio('benthic_infauna'); r_epi = ratio('mobile_epifauna')
            A_inf = np.clip(inf.energy_reserve, 0, inf.biomass).astype(np.float64)
        env.step()
        if not sample:
            continue
        y = Y[min(t // (365 * tpd), YEARS - 1)]
        B = cap['B']; tot = B.sum()
        if tot <= 0:
            continue
        k = B / (tot / B.size); occ = B > 1e-6 * tot
        pot = cap['pre'].sum(0) / np.maximum(B, 1e-12) * tpd
        eat = cap['post'].sum(0) / np.maximum(B, 1e-12) * tpd
        eat_inf = cap['post'][j_inf]
        shr = eat_inf / np.maximum(cap['post'].sum(0), 1e-30)
        s = np.divide(fg.energy_reserve, fg.biomass, out=np.zeros_like(fg.biomass, dtype=np.float64),
                      where=fg.biomass > 1e-9) / float(fg.max_energy_reserve)
        y['fill'].append(float((s * fg.biomass).sum() / max(fg.biomass.sum(), 1e-12)))
        for lo, hi, l in zip(BINS[:-1], BINS[1:], LAB):
            msk = occ & (k >= lo) & (k < hi); w = B[msk]; a = y['acc'][l]
            a['w'] += w.sum(); a['pot'] += (pot[msk] * w).sum(); a['eat'] += (eat[msk] * w).sum()
            a['fill'] += (s[msk] * w).sum(); a['inf'] += (r_inf[msk] * w).sum(); a['epi'] += (r_epi[msk] * w).sum()
            a['infA'] += (A_inf[msk] * w).sum(); a['infshare'] += (shr[msk] * w).sum()
        x = np.log(k[occ]); w = B[occ]
        for key, yy0 in (('se', eat), ('sp', pot)):
            yy = np.log(np.maximum(yy0[occ], 1e-12))
            xm = (w * x).sum() / w.sum(); ym = (w * yy).sum() / w.sum()
            y[key].append(float((w * (x - xm) * (yy - ym)).sum() / max((w * (x - xm) ** 2).sum(), 1e-12)))
        y['lloyd'].append(float((B ** 2).sum() * B.size / tot ** 2))
        y['infK'].append(float(r_inf.mean())); y['infB'].append(float(inf.biomass.sum()) / inf0)
        y['benB'].append(float(fg.biomass.sum()) / ben0)
        y['infshare'].append(float(cap['post'][j_inf].sum() / max(cap['post'].sum(), 1e-30)))
        hot = k > 8
        if visit is None:
            visit = np.where(hot, 1, 0)
        else:
            ended = (visit > 0) & ~hot
            y['runs'].extend(visit[ended].tolist())
            visit = np.where(hot, visit + 1, 0)
        if hot.any():
            y['A_hot_min'].append(float(np.percentile(A_inf[hot], 10)))

print(f"seed {seed}, {YEARS} yr, frozen {D}\nlibrary {LIB}: exposed_share {phi}, refuge k "
      f"{float(inf.refuge_exchange_rate) * 1460:.2f} /yr; infauna half-sat 1/(a h) {hs_inf:.3f} t/km2 accessible")
for n, y in enumerate(Y, 1):
    W = sum(a['w'] for a in y['acc'].values())
    print(f"--- year {n}: Lloyd {np.mean(y['lloyd']):.1f}; benthivores B/B0 {np.mean(y['benB']):.2f}; "
          f"infauna B/B0 {np.mean(y['infB']):.3f}, grid A/K_A {np.mean(y['infK']):.3f}; fill {np.mean(y['fill']):.3f}; "
          f"infauna share of eaten tonnes {np.mean(y['infshare']):.3f}")
    se, sp = np.array(y['se']), np.array(y['sp'])
    print(f"    elasticity on k: eaten {se.mean():+.3f} (days p10 {np.percentile(se,10):+.3f}, p90 {np.percentile(se,90):+.3f}); "
          f"pre-cap {sp.mean():+.3f} (p10 {np.percentile(sp,10):+.3f}, p90 {np.percentile(sp,90):+.3f})")
    runs = np.array(y['runs']) if y['runs'] else np.array([0])
    print(f"    stays at k > 8 (days): mean {runs.mean():.2f}, p90 {np.percentile(runs,90):.0f}, max {runs.max()}; "
          f"accessible infauna in k > 8 cells p10 (mean over days) {np.mean(y['A_hot_min']) if y['A_hot_min'] else float('nan'):.2f} t/km2")
    print(f"    {'k':>6s} {'share B':>8s} {'pot t/t/d':>9s} {'eat t/t/d':>9s} {'fill':>6s} {'infA/K_A':>9s} {'infA t/km2':>10s} "
          f"{'epi/K':>6s} {'inf share':>9s}")
    for l in LAB:
        a = y['acc'][l]
        if a['w'] <= 0: continue
        w = a['w']
        print(f"    {l:>6s} {w/W:8.3f} {a['pot']/w:9.4f} {a['eat']/w:9.4f} {a['fill']/w:6.3f} {a['inf']/w:9.3f} "
              f"{a['infA']/w:10.2f} {a['epi']/w:6.3f} {a['infshare']/w:9.3f}")
