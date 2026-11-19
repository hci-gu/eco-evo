"""Does packing cost the benthivores food? Frozen policies, one probe year.

Per daily sample (the tick's predation step), per cell:
  k        = own density / grid-mean density
  pot      = demand per tonne before the reserve-room cap (what the food allows), t/t/d
  eat      = demand per tonne after the cap (what is eaten, before the harvest cap), t/t/d
  fill     = reserve fill s of the benthivores in the cell
  bed/K    = each benthic prey's (accessible) density over its K in the cell
Field (Interferens hos bottenfisk Q2): intake ~ k^-m, m 0.19-0.21 large cod / flounder
(depletion included), 0 small cod and cod on capelin.
"""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import predation
import inference

D, seed, DAYS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
G = 'benthivorous_fish'
BEDS = ['benthic_infauna', 'mobile_epifauna', 'norway_lobster']
cfg = json.load(open(D + '/config.json'))

cap = {}
orig_cap = predation._reserve_room_cap
def hooked(env, demand, predator_biomass, reserve_offset):
    out = orig_cap(env, demand, predator_biomass, reserve_offset)
    if cap.get('on'):
        i = env.dm_ids.index(G)
        cap['pre'] = demand[i].astype(np.float64).copy()
        cap['post'] = out[i].astype(np.float64).copy()
        cap['B'] = predator_biomass[i].astype(np.float64).copy()
        cap['H'] = (demand[i] * env.handling_time_mat[i][:, None, None]).sum(0).astype(np.float64)
    return out
predation._reserve_room_cap = hooked

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)

order = list(env.global_fg_order)
bed_j = {b: order.index(b) for b in BEDS if b in order}
benthic_cols = list(bed_j.values())
fg = env.fgs[G]
ME = float(fg.max_energy_reserve)
tpd = 4  # ticks per day at 6 h

def bed_ratio(b):
    f = env.fgs[b]
    K = f.capacity_map
    if K is None:
        return None
    x = f.biomass.astype(np.float64)
    if getattr(f, 'exposed_share', None) is not None:
        x = np.clip(f.energy_reserve, 0, f.biomass).astype(np.float64)
        K = K * f.exposed_share
    return x / np.maximum(K, 1e-12)

BINS = [0, 0.5, 2, 8, 32, 1e9]
LAB = ['<0.5', '0.5-2', '2-8', '8-32', '>32']
acc = {l: dict(w=0.0, hshare=0.0, pot=0.0, potb=0.0, eat=0.0, eatb=0.0, fill=0.0,
               **{b: 0.0 for b in bed_j}) for l in LAB}
slopes = {'pot': [], 'eat': [], 'potb': []}
seen_vs_area = {b: [] for b in bed_j}
overlap = []
fill_mean = []
# depletion during a visit: cell occupied at k > 8, days since arrival
visit = None
dep = {}
prevB = None
with evaluation_randomness(seed * 13 + 5):
    for t in range(DAYS * tpd):
        sample = (t % tpd == 0)
        cap['on'] = sample
        if sample:
            beds_now = {b: bed_ratio(b) for b in bed_j}
        env.step()
        if not sample:
            continue
        B = cap['B']; tot = B.sum()
        if tot <= 0:
            continue
        k = B / (tot / B.size)
        occ = B > 1e-6 * tot
        pot = cap['pre'].sum(0) / np.maximum(B, 1e-12) * tpd
        potb = cap['pre'][benthic_cols].sum(0) / np.maximum(B, 1e-12) * tpd
        eat = cap['post'].sum(0) / np.maximum(B, 1e-12) * tpd
        eatb = cap['post'][benthic_cols].sum(0) / np.maximum(B, 1e-12) * tpd
        # fill at sample end (after growth); use the post-step state: close enough for bins
        s = np.divide(fg.energy_reserve, fg.biomass, out=np.zeros_like(fg.biomass, dtype=np.float64),
                      where=fg.biomass > 1e-9) / ME
        fill_mean.append(float((s * fg.biomass).sum() / max(fg.biomass.sum(), 1e-12)))
        for lo, hi, l in zip(BINS[:-1], BINS[1:], LAB):
            msk = occ & (k >= lo) & (k < hi)
            w = B[msk]
            a = acc[l]; a['w'] += w.sum()
            a['pot'] += (pot[msk] * w).sum(); a['potb'] += (potb[msk] * w).sum()
            a['eat'] += (eat[msk] * w).sum(); a['eatb'] += (eatb[msk] * w).sum()
            a['fill'] += (s[msk] * w).sum()
            a['hshare'] += (cap['H'][msk] / np.maximum(B[msk], 1e-12) * w).sum()
            for b, r in beds_now.items():
                if r is not None:
                    a[b] += (r[msk] * w).sum()
        for b, r in beds_now.items():
            if r is not None:
                seen_vs_area[b].append(float((r * B).sum() / tot / r.mean()))
        x = np.log(k[occ]); w = B[occ]
        for key, y in (('pot', pot), ('eat', eat), ('potb', potb)):
            yy = np.log(np.maximum(y[occ], 1e-12))
            xm = (w * x).sum() / w.sum(); ym = (w * yy).sum() / w.sum()
            slopes[key].append(float((w * (x - xm) * (yy - ym)).sum() / max((w * (x - xm) ** 2).sum(), 1e-12)))
        if prevB is not None:
            overlap.append(float(np.minimum(prevB, B).sum() / tot))
        prevB = B
        # visits: per cell days since k first exceeded 8 (continuous)
        hot = k > 8
        if visit is None:
            visit = np.where(hot, 0, -1)
        else:
            visit = np.where(hot, np.where(visit >= 0, visit + 1, 0), -1)
        r = beds_now.get('mobile_epifauna')
        ri = beds_now.get('benthic_infauna')
        for d in (0, 1, 2, 4, 8, 16, 32):
            msk = (visit >= d) & (visit < (d * 2 if d else 1))
            if msk.any():
                e = dep.setdefault(d, [0, 0.0, 0.0, 0.0])
                e[0] += int(msk.sum()); e[1] += float(r[msk].sum()) if r is not None else 0.0
                e[2] += float(ri[msk].sum()) if ri is not None else 0.0
                e[3] += float(potb[msk].sum())

print(f"seed {seed}, {DAYS} d, frozen {D}")
print(f"mean reserve fill {np.mean(fill_mean):.3f}; day-to-day overlap of the benthivore field "
      f"{np.mean(overlap):.3f} (share of biomass in the same cell one day later, min-overlap)")
print("elasticity d log(per tonne) / d log(k), biomass-weighted across cells, mean over days "
      "(field: -0.19..-0.21 large cod/flounder, 0 small cod):")
for key, lab in (('pot', 'potential intake (all prey)'), ('potb', 'potential intake (bed prey)'),
                 ('eat', 'intake after reserve cap')):
    a = np.array(slopes[key]); print(f"  {lab:32s} {a.mean():+.3f}  (days p10 {np.percentile(a,10):+.3f}, p90 {np.percentile(a,90):+.3f})")
print("bed prey state seen by the average benthivore tonne over the grid mean (prey/K):")
for b, v in seen_vs_area.items():
    if v: print(f"  {b:18s} {np.mean(v):.3f}")
print("by own-density class k (biomass-weighted):")
print(f"{'k':>6s} {'share B':>8s} {'pot t/t/d':>9s} {'pot bed':>8s} {'eat t/t/d':>9s} {'eat bed':>8s} {'fill':>6s} "
      + ' '.join(f'{b[:10]:>10s}' for b in bed_j) + '  handling/tick')
W = sum(a['w'] for a in acc.values())
for l in LAB:
    a = acc[l]
    if a['w'] <= 0: continue
    w = a['w']
    print(f"{l:>6s} {w/W:8.3f} {a['pot']/w:9.4f} {a['potb']/w:8.4f} {a['eat']/w:9.4f} {a['eatb']/w:8.4f} {a['fill']/w:6.3f} "
          + ' '.join(f"{a[b]/w:10.3f}" for b in bed_j) + f"  {a['hshare']/w:8.3f}")
print("cells held at k > 8: bed state by days since arrival (epifauna/K, infauna A/K_A, pot bed t/t/d):")
for d in sorted(dep):
    n, e, i, p = dep[d]
    print(f"  day {d:>2d}+ : cells {n:5d}  epifauna {e/n:.3f}  infauna {i/n:.3f}  pot bed {p/n:.4f}")

from lib.environments.ecosystem_env import interactions
i = env.dm_ids.index(G)
a_now = interactions.attack_rate(env)
print("bed pairs: half-saturation 1/(a h) (t/km2 accessible) vs the accessible density under k > 32 cells now:")
B = fg.biomass; k = B / B.mean()
for b, j in bed_j.items():
    a = float(a_now[i, j]); h = float(env.handling_time_mat[i, j]); f = env.fgs[b]
    x = np.clip(f.energy_reserve, 0, f.biomass) if getattr(f, 'exposed_share', None) is not None else f.biomass
    hs = 1 / (a * h) if a * h > 0 else float('nan')
    print(f"  {b:18s} a {a:.4g} h {h:.4g} half-sat {hs:.3g}   accessible density k>32 {x[k > 32].mean() if (k > 32).any() else float('nan'):.3g}, grid mean {x.mean():.3g}")
