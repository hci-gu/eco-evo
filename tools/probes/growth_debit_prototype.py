"""Prototype: make the growth term PAY for the biomass it adds.

Option A of the design note - energy as currency, not only as a gate.
The growth wish is unchanged (B*g*q), but it is capped by the reserve
above the maintenance line and, crucially, the reserve is DEBITED:

    E_avail = max(0, R - u*ME*B)              reserve above maintenance
    dB      = min(B*g*q, E_avail / ec)
    R      -= dB * ec

Monkey-patched for every DM; nothing in lib/ is modified.
"""
import sys, argparse
import numpy as np
sys.path.insert(0, '.')
from inference import build_env
from lib.diagnostics import viability
from lib.environments.ecosystem_env import population_change as pc

_orig = pc._apply_decision_maker_population_change


def _debited(env, fg_id, fg):
    nm = float(getattr(fg, 'natural_mortality', 0.0) or 0.0) * float(
        getattr(env, 'mortality_multiplier', 1.0))
    if nm > 0.0 and env.apply_natural_mortality:
        keep = np.float32(max(0.0, 1.0 - nm))
        fg.energy_reserve = (fg.energy_reserve * keep).astype(env.dtype, copy=False)
        fg.biomass = (fg.biomass * keep).astype(env.dtype, copy=False)

    q = fg.energy_level - fg.maintenance_level
    ec = float(fg.params.get('energy_content', 0.0) or 0.0)
    ME = float(fg.max_energy_reserve)
    u = float(fg.maintenance_level)
    grow = q >= 0.0

    # --- growth side: capped AND charged -----------------------------
    wish = fg.biomass * np.float32(fg.growth_rate) * np.maximum(0.0, q)
    if ec > 0.0:
        avail = np.maximum(0.0, fg.energy_reserve - u * ME * fg.biomass)
        dB = np.minimum(wish, avail / ec)
    else:
        dB = wish
    dB = np.where(grow, dB, 0.0)
    fg.energy_reserve = np.maximum(
        0.0, fg.energy_reserve - dB * ec).astype(env.dtype, copy=False)

    # --- starvation side: unchanged from the engine -------------------
    starve = np.float32(fg.starve_rate) if fg.starve_rate > 0.0 else np.float32(fg.growth_rate)
    loss = np.where(grow, 0.0, -fg.biomass * starve * q)
    loss = np.minimum(loss, fg.biomass)
    env.loss_starvation[fg_id] = (float(env.loss_starvation.get(fg_id, 0.0))
                                 + float(loss.sum()))
    red = np.ones_like(fg.biomass)
    m = loss > 0
    red[m] = (fg.biomass[m] - loss[m]) / (fg.biomass[m] + 1e-9)
    fg.energy_reserve = (fg.energy_reserve * np.clip(red, 0.0, 1.0)).astype(
        env.dtype, copy=False)
    fg.biomass = np.maximum(0.0, fg.biomass + dB - loss).astype(env.dtype, copy=False)


p = argparse.ArgumentParser()
p.add_argument('--debit', action='store_true')
p.add_argument('--phyto-growth', type=float, default=None)
p.add_argument('--ticks', type=int, default=3000)
p.add_argument('--window', type=int, default=300)
a = p.parse_args()

pc._apply_decision_maker_population_change = _debited if a.debit else _orig
env = build_env('mareld2.yaml', (60, 60), seed=0, verbose=False,
                apply_natural_mortality=True, migration=False, tick_hours=6)
zoo, phyto = env.fgs['zooplankton'], env.fgs['phytoplankton']
zoo.growth_rate = 0.0913
if a.phyto_growth is not None:
    phyto.growth_rate = a.phyto_growth
prov = viability.install_behaviour(env, 'greedy', seed=0)
viability.colocate_spawn(env)

acc = dict(gr=0.0, zb=0.0, pb=0.0, gb=0.0, st=0.0, n=0)
prev = [0.0, 0.0]
start = a.ticks - a.window


def on_tick(k, total):
    pl = float(env.loss_predation.get('phytoplankton', 0.0))
    zs = float(env.loss_starvation.get('zooplankton', 0.0))
    if k > start:
        acc['gr'] += pl - prev[0]; acc['st'] += zs - prev[1]
        acc['zb'] += float(zoo.biomass.sum()); acc['pb'] += float(phyto.biomass.sum())
        acc['gb'] += float(env.fgs['gadoids'].biomass.sum()); acc['n'] += 1
    prev[0], prev[1] = pl, zs


res = viability.run_rollout(env, a.ticks, prov, on_tick)
n = acc['n']
zb, pb, gb = acc['zb']/n, acc['pb']/n, acc['gb']/n
nmloss = zb * float(zoo.natural_mortality)
prod = acc['st']/n + nmloss
print(f"{'DEBITED' if a.debit else 'baseline'}  phyto g="
      f"{phyto.growth_rate:g}  window last {n} of {a.ticks}")
print(f"  zoo {zb:>9.0f} t ({zb/res.reference['zooplankton']:.2f}x spawn)   "
      f"phyto {pb:>8.0f} t ({pb/res.reference['phytoplankton']:.2f}x)   "
      f"zoo:phyto {zb/pb:>5.2f}:1")
print(f"  gadoids {gb:>8.0f} t ({gb/res.reference['gadoids']:.2f}x spawn)")
print(f"  phyto grazed {acc['gr']/n:>8.1f} t/tick   zoo production (= losses) "
      f"{prod:>8.1f} t/tick   -> efficiency {100*prod/max(acc['gr']/n,1e-9):>6.1f} %"
      f"   (ceiling 28.9 %)")
print(f"  of which starvation {acc['st']/n:>8.1f} t/tick")
