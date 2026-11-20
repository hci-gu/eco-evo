"""Does rest pay in the model today? Frozen world, one year after the warm-up: every fish group's eat shares are
scaled by k after the policy's decision and the removed share is added to rest (the moving share and its foraging,
eat + (1 - tau) M q with the policy's own prey choice q, are kept). k = 1 is the control. Same randomness per arm.
Readouts per fish group over the year (biomass-weighted per tick):
  T        foraging share actually used (interactions.attack_choice)
  feed     realised feeding level (post reserve-room cap; ceiling share at ambient temperature)
  fill     reserve / (B ME), year mean and minimum month
  B end    biomass at the end / at the start of the year
  R end    reserve at the end / at the start of the year
  M2       all-predator mortality suffered (/yr; intake_by_pred_prey over the mean biomass)
Usage: forced_rest.py POLICY_DIR SEED LIBRARY K [K ...]"""
import sys, json, copy, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.environments.ecosystem_env import interactions, predation, decisions
import inference
D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
KS = [float(x) for x in sys.argv[4:]]
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
    G['post'] = np.array(out, dtype=np.float64); G['B'] = np.array(pb, dtype=np.float64)
    return out
predation._reserve_room_cap = h_cap
with evaluation_randomness(seed):
    env0 = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env0, D, verbose=False)
    env0.policies = pol; env0.rebuild_batched_weights(); env0.obs_mean, env0.obs_var = m, v
    settle(env0, cfg['warmup_ticks'], seed=seed)
idx = {f: env0.dm_ids.index(f) for f in FISH if f in env0.dm_ids}
fi = np.array(list(idx.values()))
h = np.asarray(env0.handling_time_mat, dtype=np.float64)


def run(k):
    env = copy.deepcopy(env0)
    orig = env.calculate_decisions

    def calc():
        a = orig()
        if k == 1.0:
            return a
        eat = a.eat.copy(); rest = a.rest.copy()
        q = decisions.eat_preference(a.eat)
        removed = (1.0 - k) * eat[fi].sum(axis=1)
        eat[fi] = eat[fi] * np.float32(k); rest[fi] = rest[fi] + removed.astype(rest.dtype)
        forage = a.forage
        if forage is not None:
            forage = decisions.forage_with(env, eat, a.move, q)
        b = type(a)(move=a.move, rest=rest, eat=eat, forage=forage)
        env.publish_action_probabilities(b)
        return b
    env.calculate_decisions = calc
    acc = {f: np.zeros((12, 4)) for f in idx}          # B T, feed B, fill B, B
    B0 = {f: float(env.fgs[f].biomass.sum()) for f in idx}
    R0 = {f: float(env.fgs[f].energy_reserve.sum()) for f in idx}
    pre = {f: sum(float(env.intake_by_pred_prey.get(p, {}).get(f, 0.0)) for p in env.dm_ids) for f in idx}
    Bsum = {f: 0.0 for f in idx}
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
                feed = float((G['post'][i] * h[i][:, None, None]).sum()) / q
                fg = env.fgs[f]
                fill = float(fg.energy_reserve.sum() / max(fg.biomass.sum() * fg.max_energy_reserve, 1e-12))
                acc[f][month] += [float((B * T).sum()), feed, fill * Bt, Bt]
                Bsum[f] += float(fg.biomass.sum())
    out = {}
    for f in idx:
        a = acc[f]; tot = a.sum(0); Bm = np.maximum(a[:, 3], 1e-12)
        post = sum(float(env.intake_by_pred_prey.get(p, {}).get(f, 0.0)) for p in env.dm_ids)
        out[f] = dict(T=tot[0] / tot[3], feed=tot[1] / tot[3], fill=tot[2] / tot[3], fill_min=float((a[:, 2] / Bm).min()),
                      B_end=float(env.fgs[f].biomass.sum()) / B0[f], R_end=float(env.fgs[f].energy_reserve.sum()) / R0[f],
                      M2=(post - pre[f]) / (Bsum[f] / 1460))
    return out


res = {k: run(k) for k in KS}
print(f"seed {seed}; {D}; library {LIB}; eat shares x k, the rest to rest, all five fish groups")
for f in idx:
    print(f"\n{f}")
    print(f"  {'k':>5s} {'T':>6s} {'feed':>6s} {'fill':>6s} {'fillmin':>7s} {'B end':>6s} {'R end':>6s} {'M2':>7s}")
    for k in KS:
        r = res[k][f]
        print(f"  {k:5.2f} {r['T']:6.3f} {r['feed']:6.3f} {r['fill']:6.3f} {r['fill_min']:7.3f} {r['B_end']:6.3f} "
              f"{r['R_end']:6.3f} {r['M2']:7.3f}")
json.dump({str(k): v for k, v in res.items()}, open(f"forced_rest_s{seed}.json", "w"), indent=1)
