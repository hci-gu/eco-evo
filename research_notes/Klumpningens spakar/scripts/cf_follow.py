"""Does clumping still pay against hungry predators that follow their prey? (resume 288). Frozen world.

Snapshots as cf_fill.py (278.8): days 0/91/182/273 after the warm-up, one-day windows (4 ticks). Arms:
  fill      : trained | hungry (water column 0.71, whiting its month's field start_fill; mammals unchanged, 278.8)
  predators : trained | wX (each cell (1 - X) trained + X following; X 0.05/0.1/0.2, half = 0.5) | follow (every predator of the benthivores -
              water column, whiting, seals, porpoises - redistributed in proportion to the benthivores' biomass in
              that arm, its total and fill kept)
  benthivores: clumped | 3.0 (Gaussian spread to Lloyd 3) | field (GOV adults by class, J at the recruit shape)
Readout: all-predator M2 on the benthivores (/yr) and the saving of clumping = M2(spread) - M2(clumped) under the
same fill and predator rule (the predators follow the benthivores in each arm).
Usage: cf_follow.py POLICY_DIR SEED LIBRARY
"""
import sys, json, copy, numpy as np, yaml
sys.path.insert(0, '/home/simon/eco-evo')
from scipy.ndimage import gaussian_filter
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.world import habitat
from lib.diagnostics.field_diet import field_cell_weights, FIELD_STAGE_BED_SHARES
import inference

D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SNAP = [0, 91, 182, 273]; W = 4
WC, BE, WH = 'predatory_fish', 'benthivorous_fish', 'whiting'
PREDS = [WC, WH, 'seals', 'porpoises']
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
lloyd = lambda B: float((B ** 2).sum() * B.size / max(B.sum(), 1e-12) ** 2)
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4
man = yaml.safe_load(open(cfg['project']))
depth = habitat.load_depth_map(man, cfg['project'], (env.H, env.W))
ad_field = field_cell_weights(depth, FIELD_STAGE_BED_SHARES[BE][0])
juv_field = np.asarray(env.fgs[BE].recruit_weight_map, np.float64); juv_field /= juv_field.sum()
preds = [p for p in PREDS if p in env.fgs]

def set_fill(e, g, f):
    fg = e.fgs[g]
    fg.energy_reserve = (f * fg.biomass.astype(np.float64) * fg.max_energy_reserve).astype(fg.energy_reserve.dtype)

def place_benth(e, mode):
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

def place_pred(e, mode):
    if mode == 'trained': return
    Bb = e.fgs[BE].biomass.astype(np.float64); shape = Bb / Bb.sum()
    for p in preds:
        fg = e.fgs[p]
        B = fg.biomass.astype(np.float64); R = fg.energy_reserve.astype(np.float64); tot = B.sum()
        if tot <= 0: continue
        fill = np.divide(R, B, out=np.zeros_like(B), where=B > 1e-12)
        mean_fill = R.sum() / tot
        target = tot * shape
        w = 1.0 if mode == 'follow' else (0.5 if mode == 'half' else float(mode[1:]))
        newB = (1 - w) * B + w * target
        # each cell keeps the reserve per tonne it had; new tonnes in an empty cell get the group mean
        newR = np.where(B > 1e-12, fill, mean_fill) * newB
        fg.biomass = newB.astype(fg.biomass.dtype); fg.energy_reserve = newR.astype(fg.energy_reserve.dtype)
        if getattr(fg, 'juvenile', None) is not None:
            js = float(fg.juvenile.sum()) / tot
            fg.juvenile = np.minimum(newB * js, newB).astype(fg.juvenile.dtype)

day = 0.0
res = {}
with evaluation_randomness(seed * 7):
    for kq, sd in enumerate(SNAP):
        while day < sd:
            env.step(); day += 0.25
        month = min(11, int(((doy0 + sd) % 365) / 30.42))
        print(f"\n=== seed {seed} day {sd} (month {month + 1}); benthivore Lloyd {lloyd(env.fgs[BE].biomass):.0f}", flush=True)
        for fill in ('trained', 'hungry'):
            for pm in ('trained', 'w0.05', 'w0.1', 'w0.2', 'half', 'follow'):
                row = {}
                for bm in ('clumped', '3.0', 'field'):
                    e = copy.deepcopy(env)
                    if fill == 'hungry':
                        set_fill(e, WC, 0.71); set_fill(e, WH, float(e.fgs[WH].params['start_fill'][month]))
                    if bm != 'clumped': place_benth(e, bm)
                    place_pred(e, pm)
                    Bb_ = e.fgs[BE].biomass.astype(np.float64); Wc_ = e.fgs[WC].biomass.astype(np.float64)
                    assoc = float((Wc_ * Bb_).sum() / Bb_.sum() / Wc_.mean())
                    pre = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
                    Bsum = 0.0
                    with evaluation_randomness(1000 * seed + kq):
                        for t in range(W):
                            Bsum += float(e.fgs[BE].biomass.sum()); e.step()
                    post = {p: float(e.intake_by_pred_prey.get(p, {}).get(BE, 0.0)) for p in e.dm_ids}
                    yr = 1460 / W; Bm = Bsum / W
                    row[bm] = (sum(post[p] - pre[p] for p in post) / Bm * yr, (post[WC] - pre[WC]) / Bm * yr, assoc)
                res.setdefault((fill, pm), []).append(row)
                print(f"  fill {fill:7s} pred {pm:7s}: " + " | ".join(
                    f"{bm} M2 {row[bm][0]:.3f} (WC {row[bm][1]:.3f})" for bm in row)
                    + f" | saving L3 {row['3.0'][0] - row['clumped'][0]:+.3f} field {row['field'][0] - row['clumped'][0]:+.3f}",
                    flush=True)

print("\n=== means over the 4 snapshots: all-predator M2 on benthivores (clumped / Lloyd 3 / field) and the saving of clumping")
for (fill, pm), rows in res.items():
    mm = {bm: np.mean([r[bm][0] for r in rows]) for bm in ('clumped', '3.0', 'field')}
    am = np.mean([r['clumped'][2] for r in rows]); al = np.mean([r['3.0'][2] for r in rows])
    print(f"  fill {fill:7s} pred {pm:7s}: WC met per benth tonne / mean clumped {am:5.1f} L3 {al:4.2f} | M2 {mm['clumped']:.3f} / {mm['3.0']:.3f} / {mm['field']:.3f}; saving L3 "
          f"{mm['3.0'] - mm['clumped']:+.3f}, field {mm['field'] - mm['clumped']:+.3f}; ratio L3/clumped "
          f"{mm['3.0'] / max(mm['clumped'], 1e-9):.2f}")
