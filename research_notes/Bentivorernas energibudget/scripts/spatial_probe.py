"""Per-cell diagnostic of benthivore starvation in the frozen probe world.

    python3 spatial_probe.py CKPT_DIR SEED YEARS OUT_DIR [LIBRARY]

Builds the probe world exactly as tools/probes/frozen_years.py does
(train._ProbeEnvBuilder with PROBE_SEED = SEED, mareld2.yaml, 60x60, M1
on, migration, currents CurrentConfig(0.1, 20, 0, 12.0)), loads the
frozen policies and runs YEARS years of 1460 ticks. Nothing in the repo
is written; the engine functions are wrapped in memory only.

Per year it writes OUT_DIR/year_YY.npz with per-cell (60x60) sums over
the year for benthivorous_fish:
  B          sum of biomass at tick start (t x ticks)       -> mean = B/1460
  Bs         sum of B * reserve fill
  Blow       sum of B where fill < maintenance_level
  occ        ticks with B > 0
  gains      assimilated energy gained (MJ)
  cost       action costs charged (MJ)
  eatB       sum of B * eat share; restB, moveB likewise
  starve     starvation loss from the energy branch (t)
  sweep      biomass removed by the extinction sweep (t)
  inf_eat    infauna removed by predation (t; only benthivores eat it)
  inf_B      sum of infauna biomass (t x ticks)
  epi_B      sum of epifauna biomass
and OUT_DIR/monthly.json with grid totals per calendar month and year.
Static maps (depth, bottom-trawl effort, infauna K) go to static.npz.
"""
import json
import os
import sys

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TPY = 1460
BEN = 'benthivorous_fish'
INF = 'benthic_infauna'
EPI = 'mobile_epifauna'


def main(ckpt, seed, years, out_dir, library=None):
    import numpy as np
    import torch
    import train as train_mod
    from inference import load_policies_and_stats
    from lib.environments.ecosystem_env.currents import CurrentConfig
    from lib.environments.ecosystem_env import (predation, population_change,
                                                interactions, impacts)

    os.makedirs(out_dir, exist_ok=True)

    class Builder(train_mod._ProbeEnvBuilder):
        PROBE_SEED = seed

    env = Builder(project_path='mareld2.yaml', grid_size=(60, 60),
                  apply_natural_mortality=True, migration=True,
                  currents=CurrentConfig(0.1, 20, 0, 12.0),
                  library_path=library)()
    env.policies, env.obs_mean, env.obs_var = load_policies_and_stats(
        env, ckpt, verbose=False)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ben = env.fgs[BEN]
    i_ben = env.dm_ids.index(BEN)
    u = float(ben.maintenance_level)
    rm = float(env.dm_resting_metabolism[i_ben])
    c_rest = float(env.dm_cost_rest[i_ben])
    c_eat = float(env.dm_cost_eat[i_ben])
    c_move = float(env.dm_cost_move[i_ben])

    inf_fg = env.fgs[INF]
    static = {
        'depth': np.load('maps/depth.npz')['depth'],
        'trawl': np.load('maps/fishing_bottom_otter_trawls.npz')['bottom_trawling'],
        'inf_K': np.asarray(inf_fg.capacity_map if inf_fg.capacity_map is not None
                            else np.full((60, 60), np.nan)),
        'epi_K': np.asarray(env.fgs[EPI].capacity_map
                            if env.fgs[EPI].capacity_map is not None
                            else np.full((60, 60), np.nan)),
        'inf_B0': np.asarray(inf_fg.biomass, dtype=np.float64).copy(),
        'ben_B0': np.asarray(ben.biomass, dtype=np.float64).copy(),
    }
    np.savez_compressed(os.path.join(out_dir, 'static.npz'), **static)

    keys = ('B', 'Bs', 'Blow', 'occ', 'gains', 'cost', 'eatB', 'restB',
            'moveB', 'starve', 'sweep', 'inf_eat', 'inf_B', 'epi_B')
    acc = {}
    mkeys = ('B', 'Bs', 'Blow', 'gains', 'cost', 'starve', 'sweep',
             'inf_eat', 'tmult', 'ticks', 'eatB')
    monthly = []

    def month_of():
        if getattr(env, '_has_daylight', False):
            return int(interactions.light_index(env) * 12 // env.light_period)
        return int((env.tick_count % TPY) * 12 // TPY)

    state = {}

    orig_pred = predation.apply_predation

    def wrapped_pred(env_, actions):
        inf_before = env_.fgs[INF].biomass.astype(np.float64)
        orig_pred(env_, actions)
        inf_after = env_.fgs[INF].biomass.astype(np.float64)
        B = state['B']
        eat = actions.eat[i_ben].sum(axis=0).astype(np.float64)
        rest = actions.rest[i_ben].astype(np.float64)
        move = actions.move[i_ben].sum(axis=0).astype(np.float64)
        tm = interactions.metabolism_temperature(env_)
        tmult = float(tm[i_ben]) if tm is not None else 1.0
        cf = impacts.impact_energy_cost_factor(env_)[i_ben].astype(np.float64)
        cost = B * (rest * c_rest + eat * c_eat + move * c_move) * rm * tmult * cf
        gains = env_.fgs[BEN].temp_energy_gains.astype(np.float64)
        ie = inf_before - inf_after
        acc['gains'] += gains
        acc['cost'] += cost
        acc['eatB'] += B * eat
        acc['restB'] += B * rest
        acc['moveB'] += B * move
        acc['inf_eat'] += ie
        m = state['m']
        mon = state['mon']
        mon['gains'][m] += gains.sum()
        mon['cost'][m] += cost.sum()
        mon['inf_eat'][m] += ie.sum()
        mon['tmult'][m] += tmult
        mon['eatB'][m] += (B * eat).sum()

    orig_dm = population_change._apply_decision_maker_population_change

    def wrapped_dm(env_, fg_id, fg):
        if fg_id != BEN:
            return orig_dm(env_, fg_id, fg)
        s = fg.energy_level.astype(np.float64)
        B = fg.biomass.astype(np.float64)
        before = float(env_.loss_starvation.get(BEN, 0.0))
        orig_dm(env_, fg_id, fg)
        lost = float(env_.loss_starvation.get(BEN, 0.0)) - before
        w = B * np.maximum(0.0, u - s)
        tot = w.sum()
        if lost > 0 and tot > 0:
            cell = lost * w / tot
            acc['starve'] += cell
            state['mon']['starve'][state['m']] += lost

    orig_sweep = population_change.apply_extinction_threshold

    def wrapped_sweep(env_):
        b0 = env_.fgs[BEN].biomass.astype(np.float64)
        orig_sweep(env_)
        d = b0 - env_.fgs[BEN].biomass.astype(np.float64)
        acc['sweep'] += d
        state['mon']['sweep'][state['m']] += d.sum()

    predation.apply_predation = wrapped_pred
    population_change._apply_decision_maker_population_change = wrapped_dm
    population_change.apply_extinction_threshold = wrapped_sweep

    for y in range(years):
        for k in keys:
            acc[k] = np.zeros((60, 60))
        mon = {k: [0.0] * 12 for k in mkeys}
        state['mon'] = mon
        for t in range(TPY):
            B = ben.biomass.astype(np.float64)
            s = ben.energy_level.astype(np.float64)
            state['B'] = B
            m = month_of()
            state['m'] = m
            acc['B'] += B
            acc['Bs'] += B * s
            low = B * (s < u)
            acc['Blow'] += low
            acc['occ'] += (B > 0)
            acc['inf_B'] += inf_fg.biomass
            acc['epi_B'] += env.fgs[EPI].biomass
            mon['B'][m] += B.sum()
            mon['Bs'][m] += (B * s).sum()
            mon['Blow'][m] += low.sum()
            mon['ticks'][m] += 1
            env.step()
        np.savez_compressed(os.path.join(out_dir, f'year_{y + 1:02d}.npz'),
                            **{k: v.astype(np.float32) for k, v in acc.items()},
                            ben_end=ben.biomass.astype(np.float32),
                            fill_end=ben.energy_level.astype(np.float32),
                            inf_end=inf_fg.biomass.astype(np.float32))
        monthly.append(mon)
        with open(os.path.join(out_dir, 'monthly.json'), 'w') as fh:
            json.dump({'seed': seed, 'ckpt': ckpt, 'library': library,
                       'rm': rm, 'u': u, 'c': [c_rest, c_eat, c_move],
                       'years': monthly}, fh)
        print(f'year {y + 1} done: ben {ben.biomass.sum():.0f} t, '
              f'starve {acc["starve"].sum():.0f}, sweep {acc["sweep"].sum():.0f}',
              file=sys.stderr, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 4:
        print(__doc__)
        sys.exit(2)
    main(a[0], int(a[1]), int(a[2]), a[3], a[4] if len(a) > 4 else None)
