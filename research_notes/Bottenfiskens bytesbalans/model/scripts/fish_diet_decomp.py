"""Why do the benthivores eat so little fish? Encounter decomposition in a frozen world.

Per tick, hooked inside interactions.multi_prey_rate (the engine's own a(t) x depth mult and per-pair
visible prey, stage access included), for the benthivore row and each prey j:
  lambda_j(c) = a_j(t, c) x Bvis_j(c)        (all benthivore pairs are Type II)
  M_j  = sum_c B_be(c) lambda_j(c) / sum_c B_be(c)       encounter met by the mean benthivore tonne
  gate = a_lib_j x Bvis_gate_j                            budget_gate's equal-cell encounter at start
  M_j / gate = L x P x V x D x O with
    L = a_j(t) / a_lib (light / season), P = grid-mean B_j / manifest density,
    V = (sum Bvis_j / sum B_j) / (Bvis_gate / B_gate)  (visible + stage-access share),
    D = mean(mult x Bvis) / mean(Bvis)                 (prey placed on the pair's favoured beds),
    O = M_j / (a_j(t) mean(mult x Bvis))               (benthivores' association with that prey)
Encounter share S_j = M_j / sum_k M_k vs the gate's share. Realised tonnes from the intake book alongside.
Usage: fish_diet_decomp.py POLICY_DIR SEED DAYS LIBRARY
"""
import sys, json, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import interactions
import inference

D, seed, DAYS, LIB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
BE = 'benthivorous_fish'
PREY = ['benthic_infauna', 'krill', 'mobile_epifauna', 'norway_lobster', 'planktivorous_fish', 'whiting']
# budget_gate's equal-cell start densities (manifest) and visible tonnes per pair (diet gate block)
GATE_B = {'benthic_infauna': 93.61, 'krill': 5.00, 'mobile_epifauna': 3.78, 'norway_lobster': 4.58,
          'planktivorous_fish': 4.72, 'whiting': 0.45}
GATE_VIS = {'benthic_infauna': 18.722, 'krill': 5.000, 'mobile_epifauna': 2.342, 'norway_lobster': 0.092,
            'planktivorous_fish': 1.717, 'whiting': 0.139}
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB

grab = {}
orig = interactions.multi_prey_rate
def hooked(intake_rate, handling_time, prey_visible, *a, **k):
    if grab.get('on'):
        grab['a'] = np.array(intake_rate, dtype=np.float64, copy=True)
        grab['vis'] = np.array(prey_visible, dtype=np.float64, copy=True)
    return orig(intake_rate, handling_time, prey_visible, *a, **k)
interactions.multi_prey_rate = hooked
from lib.environments.ecosystem_env import predation
orig_cap = predation._reserve_room_cap
def hooked_cap(env, demand, predator_biomass, reserve_offset):
    out = orig_cap(env, demand, predator_biomass, reserve_offset)
    if grab.get('on'):
        ii = env.dm_ids.index(BE)
        grab['B'] = np.array(predator_biomass[ii], dtype=np.float64, copy=True)
        grab['pre'] = np.array(demand[ii], dtype=np.float64, copy=True)
        grab['post'] = np.array(out[ii], dtype=np.float64, copy=True)
    return out
predation._reserve_room_cap = hooked_cap

with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
order = list(env.global_fg_order); i = env.dm_ids.index(BE)
J = {p: order.index(p) for p in PREY}
a_lib = {p: float(env.max_intake_mat[i, J[p]]) for p in PREY}
gate = {p: a_lib[p] * GATE_VIS[p] for p in PREY}
gate_share = {p: gate[p] / sum(gate.values()) for p in PREY}

acc = {p: {k: [] for k in ('M', 'L', 'P', 'V', 'D', 'O')} for p in PREY}
shares = {p: [] for p in PREY}; pre_sh = {p: [] for p in PREY}; post_sh = {p: [] for p in PREY}
prev = {q: float(x) for q, x in env.intake_by_pred_prey.get(BE, {}).items()}
with evaluation_randomness(seed * 13 + 5):
    for t in range(DAYS * 4):
        grab['on'] = True                    # every tick (all four light phases)
        env.step()
        if not grab.get('on'):
            continue
        grab['on'] = False
        Bb = grab['B']; wt = Bb / Bb.sum()          # benthivores at the predation step
        pre_tot = grab['pre'].sum(); post_tot = grab['post'].sum()
        Ms = {}
        for p in PREY:
            j = J[p]
            a = grab['a'][i, j] if grab['a'].ndim == 4 else np.broadcast_to(grab['a'][i, j], Bb.shape)
            a = np.broadcast_to(a, Bb.shape)
            vis = grab['vis'][i, j] if grab['vis'].shape[0] == env.N_dm else grab['vis'][0, j]
            at = float(a.mean())
            # a(t, c) = a(t) x mult(c); recover mult with the pair's depth table
            mult = a / at if at > 0 else np.zeros_like(a)
            lam = a * vis
            M = float((wt * lam).sum())
            Bj = env.fgs[p].biomass.astype(np.float64)
            L = at / a_lib[p] if a_lib[p] > 0 else 0.0
            P = Bj.mean() / GATE_B[p]
            V = (vis.sum() / max(Bj.sum(), 1e-30)) / (GATE_VIS[p] / GATE_B[p])
            Dd = (mult * vis).mean() / max(vis.mean(), 1e-30)
            O = M / max(at * (mult * vis).mean(), 1e-30)
            for k, x in (('M', M), ('L', L), ('P', P), ('V', V), ('D', Dd), ('O', O)):
                acc[p][k].append(x)
            Ms[p] = M
        tot = sum(Ms.values())
        for p in PREY:
            shares[p].append(Ms[p] / tot)
            pre_sh[p].append(grab['pre'][J[p]].sum() / pre_tot); post_sh[p].append(grab['post'][J[p]].sum() / post_tot)

now = {q: float(x) for q, x in env.intake_by_pred_prey.get(BE, {}).items()}
d = {q: now.get(q, 0) - prev.get(q, 0) for q in now}; dt = sum(d.values())
print(f"seed {seed}, {DAYS} d, frozen {D}\nlibrary {LIB}")
print("per prey (means over every tick; a(t) = cell mean of a (depth tables have equal-cell mean 1), mult = a/a(t)):")
print(f"{'prey':>20s} {'gate share':>10s} {'enc share':>10s} {'pre-cap':>8s} {'post-cap':>8s} {'realised':>9s} {'M/gate':>7s} {'L light':>8s} {'P level':>8s} "
      f"{'V vis':>6s} {'D beds':>7s} {'O overlap':>9s}")
for p in PREY:
    A = {k: np.mean(v) for k, v in acc[p].items()}
    print(f"{p:>20s} {gate_share[p]:10.3f} {np.mean(shares[p]):10.3f} {np.mean(pre_sh[p]):8.3f} {np.mean(post_sh[p]):8.3f} {d.get(p, 0) / dt:9.3f} {A['M'] / gate[p]:7.3f} "
          f"{A['L']:8.3f} {A['P']:8.3f} {A['V']:6.3f} {A['D']:7.3f} {A['O']:9.3f}")
tot_world = sum(np.mean(acc[p]['M']) for p in PREY); tot_gate = sum(gate.values())
print(f"total encounter met per benthivore tonne / gate total: {tot_world / tot_gate:.3f}")
print("note: L x P x V x D x O = M/gate up to the covariance of the factors over days")
