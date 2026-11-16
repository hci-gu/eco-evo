"""Porpoise budget and co-location with planktivores, frozen fix101 policies.

Probe world on 30x30 (as the training probe). Env:
  CKPT   checkpoint dir (frozen copy)
  TICKS  ticks (default 2920)
  PLK_STILL=1  planktivores cannot move (dm_v = 0): their moves are masked,
               the policy's remaining shares renormalised (counterfactual)
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import train as train_mod  # noqa: E402
from inference import load_policies_and_stats  # noqa: E402
from lib.environments.ecosystem_env import predation  # noqa: E402
from lib.environments.ecosystem_env.currents import CurrentConfig  # noqa: E402
from lib.world import daylight  # noqa: E402

CKPT = os.environ['CKPT']
TICKS = int(os.environ.get('TICKS', 2920))

builder = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(30, 30),
    apply_natural_mortality=True, migration=True, mass_balance=True,
    currents=CurrentConfig(0.1, 20, 0, 12.0))
env = builder()
policies, mean, var = load_policies_and_stats(env, CKPT, verbose=False)
env.policies, env.obs_mean, env.obs_var = policies, mean, var
env.build_static_caches()

P, F = 'porpoises', 'planktivorous_fish'
ip = env.dm_ids.index(P)
iff = env.dm_ids.index(F)
if os.environ.get('PLK_STILL'):
    env.dm_v[iff] = 0.0
porp, plk = env.fgs[P], env.fgs[F]
u = float(porp.maintenance_level)
c_eat, c_move, c_rest = (float(env.dm_cost_eat[ip]), float(env.dm_cost_move[ip]),
                         float(env.dm_cost_rest[ip]))
rm = float(env.dm_resting_metabolism[ip])
book = {}
_apply = predation.apply_predation


def booking(e, actions):
    b_plk_before = plk.biomass.copy()
    _apply(e, actions)
    tot = max(float(porp.biomass.sum()), 1e-30)
    w = porp.biomass / tot
    book['B'] = tot
    book['gain'] = float(porp.temp_energy_gains.sum()) / tot
    book['eat'] = float((actions.eat[ip].sum(axis=0) * w).sum())
    book['move'] = float((actions.move[ip].sum(axis=0) * w).sum())
    book['rest'] = float((actions.rest[ip] * w).sum())
    book['s'] = float((porp.energy_level * w).sum())
    book['starv'] = float((w * (porp.energy_level < u)).sum())
    book['occ'] = int((porp.biomass > 0).sum())
    book['sub'] = int(((porp.biomass > 0) & (porp.biomass < float(e.dm_min_split[ip]))).sum())
    # planktivore density where the porpoises are vs the grid mean
    book['plk_loc'] = float((b_plk_before * w).sum())
    book['plk_mean'] = float(b_plk_before.mean())
    pw = plk.biomass / max(float(plk.biomass.sum()), 1e-30)
    book['plk_s'] = float((plk.energy_level * pw).sum())
    book['plk_occ'] = int((plk.biomass > 1e-3 * plk.biomass.max()).sum())
    book['plk_move'] = float((actions.move[iff].sum(axis=0) * pw).sum())


predation.apply_predation = booking

if os.environ.get('BLIND'):
    from lib.environments.ecosystem_env import observations, impacts
    _orig = observations.build_observation_batch
    d = iff
    chan = [int(x) for x in env.obs_others_idx[d]]
    jv = getattr(env, 'obs_juvenile_idx', None)
    if jv is not None:
        chan += [int(x) for x in jv[d]]
    jp = env.global_fg_order.index(P)
    pos = [k for k, c in enumerate(chan) if c == jp]
    n_o, n_i = len(chan), len(impacts.observable_impact_layers(env))
    cd, nd = 2 + n_o + n_i, 1 + n_o + n_i
    idx = [2 + k for k in pos] + [cd + q * nd + 1 + k for q in range(4) for k in pos]
    print('blind slots', idx, [env.global_fg_order[c][:6] for c in chan], flush=True)

    def blind(e):
        out = _orig(e)
        out[d][:, idx] = 0.0
        return out
    observations.build_observation_batch = blind

if os.environ.get('UNITS_OFF'):
    env.unit_splits = False
if os.environ.get('STOCH'):
    from lib.environments.ecosystem_env import decisions as _dec
    _rng = np.random.default_rng(7)
    _orig_merge = _dec.merge_subunit_fractions

    def stoch_merge(e, probs):
        """Stochastic rounding of each move fraction to whole animals (E = continuous)."""
        out = probs.copy()
        unit = float(e.dm_min_split[ip])
        B = porp.biomass
        act = B >= unit
        if not np.any(act):
            return _orig_merge(e, probs)
        n = B / unit
        mv = probs[ip, _dec.MOVE_SLICE]                       # [4, H, W]
        units = mv * n[None]
        k = np.floor(units) + (_rng.random(units.shape) < (units - np.floor(units)))
        newmv = np.where(act[None], k / np.maximum(n[None], 1e-30), mv)
        tot = newmv.sum(axis=0)
        newmv = np.where(tot[None] > 1, newmv / np.maximum(tot[None], 1e-30), newmv)
        stay_old = probs[ip, _dec.REST_INDEX:].sum(axis=0)
        stay_new = 1.0 - newmv.sum(axis=0)
        scale = np.where(stay_old > 1e-30, stay_new / np.maximum(stay_old, 1e-30), 0.0)
        out[ip, _dec.MOVE_SLICE] = newmv
        out[ip, _dec.REST_INDEX:] = probs[ip, _dec.REST_INDEX:] * scale[None]
        # other DMs: the library rule
        rest = _orig_merge(e, probs)
        rest[ip] = out[ip]
        return rest.astype(probs.dtype, copy=False)
    _dec.merge_subunit_fractions = stoch_merge

if os.environ.get('MULTI'):
    from lib.environments.ecosystem_env import decisions as _dec
    _rngm = np.random.default_rng(7)
    _orig_merge2 = _dec.merge_subunit_fractions

    def multi_merge(e, probs):
        """Whole animals: floor(n) animals drawn multinomially over the five
        fractions (4 moves + stay) on the policy's shares; the sub-unit remainder
        rides with stay, or with the largest move if stay drew none. E = continuous."""
        res = _orig_merge2(e, probs)
        unit = float(e.dm_min_split[ip])
        B = porp.biomass
        act = B >= unit
        if not np.any(act):
            return res
        mv = probs[ip, _dec.MOVE_SLICE]
        stay = probs[ip, _dec.REST_INDEX:].sum(axis=0)
        fr = np.concatenate([mv, stay[None]], axis=0)          # [5, H, W]
        out_mv = mv.copy()
        out_stay = stay.copy()
        for h, w in zip(*np.where(act)):
            n = B[h, w] / unit
            nf = int(np.floor(n))
            pv = np.clip(fr[:, h, w].astype(np.float64), 0, None)
            pv = pv / pv.sum()
            k = _rngm.multinomial(nf, pv).astype(np.float64)
            rem = n - nf
            if k[4] > 0:
                k[4] += rem
            else:
                k[int(np.argmax(k[:4]))] += rem
            q = k / n
            out_mv[:, h, w] = q[:4]
            out_stay[h, w] = q[4]
        scale = np.where(stay > 1e-30, out_stay / np.maximum(stay, 1e-30), 0.0)
        res = res.copy()
        res[ip, _dec.MOVE_SLICE] = out_mv
        res[ip, _dec.REST_INDEX:] = probs[ip, _dec.REST_INDEX:] * scale[None]
        return res.astype(probs.dtype, copy=False)
    _dec.merge_subunit_fractions = multi_merge
np.random.seed(20260530)
torch.manual_seed(20260530)

months = defaultdict(lambda: defaultdict(list))
order = []
B0 = float(porp.biomass.sum())
for t in range(TICKS):
    cal = daylight.calendar_at(env.daylight, t)
    key = (t // 1460, cal['month_name'])
    if not order or order[-1] != key:
        order.append(key)
    env.step()
    m = months[key]
    for k, v in book.items():
        m[k].append(v)
    m['cost'].append(rm * (book['rest'] * c_rest + book['eat'] * c_eat
                           + book['move'] * c_move))
    m['Bend'].append(float(porp.biomass.sum()))

print(f"CKPT={CKPT} PLK_STILL={bool(os.environ.get('PLK_STILL'))} BLIND={bool(os.environ.get('BLIND'))} UNITS_OFF={bool(os.environ.get('UNITS_OFF'))} STOCH={bool(os.environ.get('STOCH'))} MULTI={bool(os.environ.get('MULTI'))} "
      f"rm={rm} c_eat={c_eat} c_move={c_move} u={u} B0={B0:.1f}")
print(f"{'yr':>2} {'mon':>4} {'B/B0':>6} {'s':>5} {'starv%':>6} {'occ':>4} "
      f"{'eat':>5} {'mov':>5} {'gain/t':>7} {'cost/t':>7} {'plk@P':>7} "
      f"{'plkMean':>7} {'ratio':>6} {'plk_s':>5} {'plkOcc':>6} {'plkMov':>6} {'sub':>5}")
for key in order:
    m = months[key]
    a = {k: float(np.mean(v)) for k, v in m.items()}
    print(f"{key[0]:>2} {key[1]:>4} {m['Bend'][-1] / B0:>6.3f} {a['s']:>5.2f} "
          f"{100 * a['starv']:>6.1f} {a['occ']:>4.0f} {a['eat']:>5.2f} "
          f"{a['move']:>5.3f} {a['gain']:>7.1f} {a['cost']:>7.1f} "
          f"{a['plk_loc']:>7.2f} {a['plk_mean']:>7.2f} "
          f"{a['plk_loc'] / max(a['plk_mean'], 1e-9):>6.2f} {a['plk_s']:>5.2f} "
          f"{a['plk_occ']:>6.0f} {a['plk_move']:>6.3f} {a['sub']:>5.1f}")
