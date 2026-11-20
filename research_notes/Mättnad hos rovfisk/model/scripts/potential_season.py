"""Why each fish group's uncapped feeding potential has the seasonal shape it has (follows fill_decomp_all.py).
Frozen world, one year after the warm-up. Hooks interactions.multi_prey_rate (a, B_vis, attack) and
predation._reserve_room_cap (pre/post demand, predator biomass), interactions.attack_choice (T).
Per month, per group, weighted by the foraging tonnes B T of each cell:
  q        ingestion Q10 factor (whole response; ceiling-scale levels below are divided by it)
  qm       metabolism Q10 factor (resting cost)
  unc      uncapped feeding level on the reference ceiling = sum demand h / (q B)
  feed     realised feeding level (post cap), same scale; at fill ~1 the room's refill = need
  E_unc    unc x q = potential intake in units of the reference ceiling (what the fish can take in energy)
  E_feed   feed x q
and per prey j:
  a_j      the engine's per-tick a in the predator's cells (library a x light multiplier x depth access)
  Bv_j     visible prey biomass the foraging tonnes meet (t per cell)
  X_j      a_j Bv_j h_j (handling-time demand while foraging, before saturation)
  S_j      share of the ceiling taken from prey j while foraging (uncapped)
X = sum X_j and S = X / (1 + X + I) tie the prey drivers to unc = T S.
Usage: potential_season.py POLICY_DIR SEED LIBRARY"""
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
o_mpr = interactions.multi_prey_rate
def h_mpr(a, h, Bp, m3, attack, interference=0.0):
    shp = np.broadcast_shapes(np.shape(a), np.shape(Bp))
    G['a'] = np.array(np.broadcast_to(a, shp), dtype=np.float64)
    G['Bv'] = np.array(np.broadcast_to(Bp, shp), dtype=np.float64)
    G['I'] = np.array(np.broadcast_to(interference, (shp[0], 1) + shp[2:]), dtype=np.float64)[:, 0]
    return o_mpr(a, h, Bp, m3, attack, interference)
interactions.multi_prey_rate = h_mpr
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
acc = {f: dict(base=np.zeros((12, 7)), a=np.zeros((12, len(prey[f]))), Bv=np.zeros((12, len(prey[f]))),
               X=np.zeros((12, len(prey[f]))), S=np.zeros((12, len(prey[f]))), Btot=np.zeros((12, len(prey[f]))),
               n=np.zeros(12)) for f in idx}
with evaluation_randomness(seed * 13 + 5):
    for t in range(1460):
        li = interactions.light_index(env); month = min(11, int((li / env.light_period) * 12))
        tf = interactions.ingestion_temperature(env)
        mt = env.metabolism_temp_table[li] if getattr(env, 'metabolism_temp_table', None) is not None else None
        prey_tot = {j: float(env.fgs[order[j]].biomass.sum()) if order[j] in env.fgs else 0.0
                    for j in range(len(order))}
        env.step()
        for f, i in idx.items():
            B = G['B'][i]; Bt = B.sum()
            if Bt <= 0:
                continue
            q = float(tf[i]) if tf is not None else 1.0
            qm = float(mt[i]) if mt is not None else 1.0
            T = G['T'][i].reshape(B.shape)
            w = B * T; W = w.sum()
            J = prey[f]
            a_j = G['a'][i][J]; Bv_j = G['Bv'][i][J]
            X_j = a_j * Bv_j * h[i][J][:, None, None]
            I = G['I'][i]
            S_j = X_j / (1.0 + X_j.sum(0) + I)[None]
            pre_h = (G['pre'][i][J] * h[i][J][:, None, None]).sum() / q
            post_h = (G['post'][i][J] * h[i][J][:, None, None]).sum() / q
            A = acc[f]
            A['base'][month] += [q * Bt, qm * Bt, pre_h, post_h, W, Bt, 0.0]
            if W > 0:
                A['a'][month] += (a_j * w[None]).reshape(len(J), -1).sum(1)
                A['Bv'][month] += (Bv_j * w[None]).reshape(len(J), -1).sum(1)
                A['X'][month] += (X_j * w[None]).reshape(len(J), -1).sum(1)
                A['S'][month] += (S_j * w[None]).reshape(len(J), -1).sum(1)
            A['Btot'][month] += np.array([prey_tot[j] for j in J])
            A['n'][month] += 1
mon = 'JFMAMJJASOND'
print(f"seed {seed}; {D}; library {LIB}")
out = {}
row = lambda lab, arr, fmt='5.2f': print(f"  {lab:12s} " + ' '.join(f'{x:{fmt}}' if abs(x) < 1e4 else f'{x:5.0e}'
                                                                 for x in arr))
for f in idx:
    A = acc[f]; b = A['base']; Bm = np.maximum(b[:, 5], 1e-12); Wm = np.maximum(b[:, 4], 1e-12)
    q_m, qm_m, unc_m, feed_m, T_m = b[:, 0] / Bm, b[:, 1] / Bm, b[:, 2] / Bm, b[:, 3] / Bm, b[:, 4] / Bm
    names = [order[j] for j in prey[f]]
    print(f"\n{f}")
    print("  month        " + ' '.join(f'{c:>5s}' for c in mon))
    for lab, arr in (('T', T_m), ('q ingest', q_m), ('q metab', qm_m), ('unc', unc_m), ('feed', feed_m),
                     ('E_unc', unc_m * q_m), ('E_feed', feed_m * q_m)):
        row(lab, arr)
    Xtot = A['X'].sum(1) / Wm
    row('X total', Xtot)
    pj = {}
    for k, n in enumerate(names):
        a_k = A['a'][:, k] / Wm; Bv_k = A['Bv'][:, k] / Wm; X_k = A['X'][:, k] / Wm; S_k = A['S'][:, k] / Wm
        Bg = A['Btot'][:, k] / np.maximum(A['n'], 1)
        if X_k.mean() < 0.01 * Xtot.mean():
            continue
        print(f"  -- {n}")
        row('a/mean', a_k / max(a_k.mean(), 1e-30)); row('Bv met', Bv_k, '5.2f'); row('B grid/mean', Bg / max(Bg.mean(), 1e-30))
        row('X_j', X_k); row('S_j', S_k)
        pj[n] = dict(a=a_k.tolist(), Bv=Bv_k.tolist(), Bgrid=Bg.tolist(), X=X_k.tolist(), S=S_k.tolist())
    out[f] = dict(T=T_m.tolist(), q=q_m.tolist(), qm=qm_m.tolist(), unc=unc_m.tolist(), feed=feed_m.tolist(),
                  X=Xtot.tolist(), prey=pj)
json.dump(out, open(f"potential_season_s{seed}.json", "w"), indent=1)
