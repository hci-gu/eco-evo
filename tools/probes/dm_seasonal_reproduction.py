"""Does seasonal REPRODUCTION for the decision makers change anything?

`_seasonal_population_rate` is only called from the NDM branch, so zoo,
fish, mammals and birds reproduce at a flat rate all year. Real copepods
diapause: the spring bloom escapes grazing because the grazers have not
built up yet. This monkey-patches the DM branch to use the same seasonal
rate the NDMs get, in memory only - nothing in lib/ is modified.
"""
import sys, argparse
import numpy as np
sys.path.insert(0, '.')

from inference import build_env
from lib.diagnostics import viability
from lib.environments.ecosystem_env import population_change as pc
from lib.world.energy_balance import resolve_satiation_scale

_orig = pc._apply_decision_maker_population_change


def _seasonal_dm(env, fg_id, fg):
    saved = fg.growth_rate
    fg.growth_rate = pc._seasonal_population_rate(env, fg_id, fg)
    try:
        return _orig(env, fg_id, fg)
    finally:
        fg.growth_rate = saved


def run(g, seed, ticks, grid, amp, phase_shift, seasonal_dm):
    if seasonal_dm:
        pc._apply_decision_maker_population_change = _seasonal_dm
    else:
        pc._apply_decision_maker_population_change = _orig
    env = build_env('mareld2.yaml', grid, seed=seed, verbose=False,
                    apply_natural_mortality=True, migration=False, tick_hours=6)
    zoo = env.fgs['zooplankton']
    zoo.growth_rate = float(g)
    u = float(zoo.params.get('maintenance_level', 0) or 0)
    if seasonal_dm:
        zoo.seasonal_amplitude = float(amp)
        zoo.seasonal_period = 1460.0
        # Put the grazer's peak `phase_shift` ticks after phytoplankton's,
        # which is what a diapausing copepod population does.
        env._season_phase['zooplankton'] = (
            env._season_phase.get('phytoplankton', 0.0) - float(phase_shift))
    provider = viability.install_behaviour(env, 'greedy', seed=seed)
    viability.colocate_spawn(env)
    surplus = []

    def on_tick(k, total):
        b = np.asarray(zoo.biomass, dtype=np.float64); t = b.sum()
        s = np.asarray(zoo.energy_level, dtype=np.float64)
        surplus.append(float((b * np.maximum(0.0, s - u)).sum() / t) if t > 0 else 0.0)

    res = viability.run_rollout(env, ticks, provider, on_tick)
    w = max(1, ticks // 10)
    out = {}
    for fid in ('phytoplankton', 'zooplankton', 'gadoids'):
        v = res.series[fid]; ref = res.reference[fid]
        out[fid] = (float(np.mean(v[-w:])) / ref, float(v.min()) / ref)
    return out, g * float(np.mean(surplus)) * 1460


p = argparse.ArgumentParser()
p.add_argument('--g', type=float, default=0.0913)
p.add_argument('--ticks', type=int, default=3000)
p.add_argument('--seeds', type=int, default=2)
p.add_argument('--amps', type=float, nargs='+', default=[0.0, 0.7, 0.9])
p.add_argument('--phase', type=float, default=365.0)
a = p.parse_args()
print(f"{'arm':>28} {'seed':>4} {'realPB':>7} {'phy_end':>8} {'phy_min':>8} "
      f"{'zoo_end':>8} {'gad_end':>8}")
print('-' * 80)
for amp in a.amps:
    for seed in range(a.seeds):
        on = amp > 0.0
        o, pb = run(a.g, seed, a.ticks, (60, 60), amp, a.phase, on)
        lbl = f'DM season amp={amp}' if on else 'flat (baseline)'
        print(f"{lbl:>28} {seed:>4} {pb:>7.1f} {o['phytoplankton'][0]:>8.3f} "
              f"{o['phytoplankton'][1]:>8.3f} {o['zooplankton'][0]:>8.3f} "
              f"{o['gadoids'][0]:>8.3f}", flush=True)
