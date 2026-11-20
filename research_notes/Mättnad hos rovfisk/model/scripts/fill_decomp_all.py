"""Every fish group's feeding in a frozen world, with the prey behind its spare capacity (extends fill_decomp.py).
Per tick (hooks inside the predation step): T = foraging share (interactions.attack_choice), pre-cap and post-cap
(reserve room, set point) demand per prey, predator biomass. Per cell, with shared handling
    S_j = demand_j h_j / (B T q) = X_j / (1 + X + w B)        (q = ingestion Q10 factor; X_j = a_j B_vis,j h_j)
so the share of the ceiling while foraging is S = sum_j S_j, and with prey j removed it is exactly
(S - S_j) / (1 - S_j) (no need to know X or w B).
Readouts per group (biomass-weighted over cells and ticks; feeding levels on the ceiling at ambient temperature):
  T        foraging share
  unc      uncapped feeding level  sum B T q S / (B q)
  feed     realised (post-cap) feeding level; at fill ~1 this is the room's refill = the need
  share_j  prey j's share of the uncapped handling time
  spare_j  prey j's share of the cut (pre - post)
  wo_j     uncapped feeding level with prey j removed
  fill     reserve / (B ME)
Usage: fill_decomp_all.py POLICY_DIR SEED LIBRARY"""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import interactions, predation
import inference
D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
FISH = ['planktivorous_fish', 'benthivorous_fish', 'whiting', 'predatory_fish', 'mackerel']
G = {}
o_ac = interactions.attack_choice
def h_ac(eat):
    T, p = o_ac(eat); G['T'] = np.array(T, dtype=np.float64); return T, p
interactions.attack_choice = h_ac
o_cap = predation._reserve_room_cap
def h_cap(env, demand, pb, ro):
    out = o_cap(env, demand, pb, ro)
    G['pre'] = np.array(demand, dtype=np.float64); G['post'] = np.array(out, dtype=np.float64)
    G['B'] = np.array(pb, dtype=np.float64)
    return out
predation._reserve_room_cap = h_cap
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
order = list(env.global_fg_order)
h = np.asarray(env.handling_time_mat, dtype=np.float64)
idx = {f: env.dm_ids.index(f) for f in FISH if f in env.dm_ids}
prey = {f: [j for j in range(len(order)) if h[i][j] > 0] for f, i in idx.items()}
# per group, per month: T, unc, feed, fill, wsum(B), then per prey: pre_h, cut_h, wo
acc = {f: dict(base=np.zeros((12, 5)), pre=np.zeros((12, len(prey[f]))), cut=np.zeros((12, len(prey[f]))),
               wo=np.zeros((12, len(prey[f])))) for f in idx}
with evaluation_randomness(seed * 13 + 5):
    for t in range(1460):
        li = interactions.light_index(env); month = min(11, int((li / env.light_period) * 12))
        tf = interactions.ingestion_temperature(env)
        env.step()
        for f, i in idx.items():
            B = G['B'][i]; Bt = B.sum()
            if Bt <= 0:
                continue
            q = float(tf[i]) if tf is not None else 1.0
            T = G['T'][i].reshape(B.shape)
            pre_h = G['pre'][i][prey[f]] * h[i][prey[f]][:, None, None]      # [J, H, W]
            post_h = G['post'][i][prey[f]] * h[i][prey[f]][:, None, None]
            den = B * T * q
            ok = den > 1e-12
            S_j = np.where(ok[None], pre_h / np.where(ok, den, 1.0)[None], 0.0)
            S = S_j.sum(0)
            wo = np.where(ok[None], (S[None] - S_j) / np.maximum(1.0 - S_j, 1e-9), 0.0)
            fg = env.fgs[f]
            fill = float(fg.energy_reserve.sum() / max(fg.biomass.sum() * fg.max_energy_reserve, 1e-12))
            a = acc[f]
            a['base'][month] += [float((B * T).sum()), float(pre_h.sum()) / q, float(post_h.sum()) / q,
                                 fill * Bt, Bt]
            a['pre'][month] += pre_h.reshape(len(prey[f]), -1).sum(1) / q
            a['cut'][month] += (pre_h - post_h).reshape(len(prey[f]), -1).sum(1) / q
            a['wo'][month] += (wo * (B * T)[None]).reshape(len(prey[f]), -1).sum(1)
mon = 'JFMAMJJASOND'
print(f"seed {seed}; {D}; library {LIB}")
out = {}
for f in idx:
    a = acc[f]; b = a['base']; Bm = np.maximum(b[:, 4], 1e-12)
    T_m, unc_m, feed_m, fill_m = b[:, 0] / Bm, b[:, 1] / Bm, b[:, 2] / Bm, b[:, 3] / Bm
    tot = b.sum(0); BT = tot[4]
    T_y, unc_y, feed_y, fill_y = tot[0] / BT, tot[1] / BT, tot[2] / BT, tot[3] / BT
    pre_y = a['pre'].sum(0); cut_y = a['cut'].sum(0); wo_y = a['wo'].sum(0) / BT
    names = [order[j] for j in prey[f]]
    print(f"\n{f}: year T {T_y:.2f}, uncapped {unc_y:.3f}, realised {feed_y:.3f} (cut {unc_y - feed_y:.3f}, "
          f"uncapped / realised {unc_y / max(feed_y, 1e-9):.2f}), fill {fill_y:.3f}")
    print(f"  {'prey':26s} {'share':>7s} {'spare':>7s} {'unc w/o':>8s}")
    for k, n in sorted(enumerate(names), key=lambda kv: -pre_y[kv[0]]):
        print(f"  {n:26s} {pre_y[k] / max(pre_y.sum(), 1e-30):7.3f} {cut_y[k] / max(cut_y.sum(), 1e-30):7.3f} "
              f"{wo_y[k]:8.3f}")
    print("  month   " + ' '.join(f'{c:>5s}' for c in mon))
    for lab, arr in (('T', T_m), ('unc', unc_m), ('feed', feed_m), ('fill', fill_m)):
        print(f"  {lab:7s} " + ' '.join(f'{x:5.2f}' for x in arr))
    sf = getattr(env.fgs[f], 'start_fill', None)
    if sf is not None and len(np.atleast_1d(sf)) == 12:
        print("  field   " + ' '.join(f'{x:5.2f}' for x in sf))
    out[f] = dict(T=T_y, unc=unc_y, feed=feed_y, fill=fill_y, prey=names, share=(pre_y / pre_y.sum()).tolist(),
                  spare=(cut_y / max(cut_y.sum(), 1e-30)).tolist(), wo=wo_y.tolist(),
                  month=dict(T=T_m.tolist(), unc=unc_m.tolist(), feed=feed_m.tolist(), fill=fill_m.tolist()))
json.dump(out, open(f"fill_decomp_all_s{seed}.json", "w"), indent=1)
