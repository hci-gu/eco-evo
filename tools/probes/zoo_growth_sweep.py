"""Sweep zooplankton growth_rate under the viability rig's frozen behaviour.

Evidence for mareld_resume.txt sections 109, 110, 111 and 112. Reports,
per (growth_rate, seed): survival, final-window level vs spawn, the
DELIVERED production/biomass ratio (delivered is growth_rate *
<s_X - u_X>_B * 1460 and is always BELOW the ceiling growth_rate *
(1 - u_X) * 1460), and the phytoplankton dead-cell count,
which is the failure mode total biomass hides because phyto has
seed_rate 0 and exact zero is absorbing.

Everything is overridden in memory; fg_library.yaml is never written.
Quote a number from here WITH its --mortality setting and its grid -
both move the outcome (sections 109.4, 109.8).

    python3 tools/probes/zoo_growth_sweep.py --grid 60 60 --ticks 3000 \
        --seeds 2 --mortality on --values 0.0208 0.046 0.0913

--phyto-growth / --phyto-season / --phyto-seed override the supply
group instead, which is what sections 109.5 and 112.4 measure.
"""
import argparse, sys, time
import numpy as np
sys.path.insert(0, '.')

from inference import build_env
from lib.diagnostics import viability
from lib.environments.ecosystem_env.population_change import (
    DEFAULT_MASS_BALANCE)

TICKS_PER_YEAR = 1460


def run(g, seed, ticks, grid, behaviour, spawn, mortality, phyto_g=None,
        phyto_seed=None, mass_balance=False):
    env = build_env('mareld2.yaml', grid, seed=seed, verbose=False,
                    apply_natural_mortality=mortality, migration=False,
                    tick_hours=6, mass_balance=mass_balance)
    zoo = env.fgs['zooplankton']
    zoo.growth_rate = float(g)
    if phyto_g is not None:
        env.fgs['phytoplankton'].growth_rate = float(phyto_g)
    if phyto_seed is not None:
        env.fgs['phytoplankton'].seed_rate = float(phyto_seed)
    u = float(zoo.params.get('maintenance_level', 0.0) or 0.0)
    sat = 1.0  # hunger gate h = 1 - s closes at a full reserve (section 130)
    provider = viability.install_behaviour(env, behaviour, seed=seed)
    if spawn == 'colocated':
        viability.colocate_spawn(env)

    phyto = env.fgs['phytoplankton']
    habitable = int(np.count_nonzero(phyto.biomass > 0))
    dead = []
    surplus = []

    def on_tick(k, total):
        dead.append(int(np.count_nonzero(phyto.biomass <= 0.0)))
        b = np.asarray(zoo.biomass, dtype=np.float64)
        tot = b.sum()
        if tot > 0:
            s = np.asarray(zoo.energy_level, dtype=np.float64)
            surplus.append(float((b * np.maximum(0.0, s - u)).sum() / tot))
        else:
            surplus.append(0.0)

    res = viability.run_rollout(env, ticks, provider, on_tick)
    w = max(1, ticks // 10)
    out = {'g': g, 'seed': seed, 'occupied_phyto_at_spawn': habitable,
           'dead_start': dead[w - 1], 'dead_mid': dead[ticks // 2 - 1],
           'dead_end': dead[-1]}
    for fid in ('phytoplankton', 'zooplankton', 'pelagic_fish', 'gadoids',
                'porpoises'):
        if fid not in res.series:
            continue
        v = res.series[fid]
        ref = res.reference[fid] or float('nan')
        out[f'{fid}_end'] = float(np.mean(v[-w:])) / ref
        out[f'{fid}_prev'] = float(np.mean(v[-2 * w:-w])) / ref
        out[f'{fid}_min'] = float(v.min()) / ref
        out[f'{fid}_ext'] = res.extinct_tick[fid]
    mean_surplus = float(np.mean(surplus))
    out['mean_surplus'] = mean_surplus
    out['ceiling_PB'] = g * (sat - u) * TICKS_PER_YEAR
    out['realised_PB'] = g * mean_surplus * TICKS_PER_YEAR
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', type=int, nargs=2, default=[30, 30])
    p.add_argument('--ticks', type=int, default=1500)
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--behaviour', default='greedy')
    p.add_argument('--spawn', default='colocated')
    p.add_argument('--mortality', default='on')
    p.add_argument('--phyto-growth', dest='phyto_growth', type=float,
                   default=None)
    # Mirrors the engine default, which flipped ON in section 120.
    # Hard-coding False here silently measured the legacy term.
    p.add_argument('--mass-balance', dest='mass_balance',
                   action='store_true',
                   default=DEFAULT_MASS_BALANCE,
                   help='No-op; the engine default (section 120).')
    p.add_argument('--no-mass-balance', dest='mass_balance',
                   action='store_false',
                   help='Legacy growth term, pre-section-116.')
    p.add_argument('--phyto-seed', dest='phyto_seed', type=float, default=None)
    p.add_argument('--seed-base', dest='seed_base', type=int, default=0)
    p.add_argument('--values', type=float, nargs='+',
                   default=[0.01, 0.02, 0.035, 0.05, 0.07, 0.09])
    a = p.parse_args()
    mort = a.mortality == 'on'
    hdr = (f"{'g':>7} {'seed':>4} {'ceilPB':>7} {'realPB':>7} {'<surp>':>7} "
           f"{'phy_end':>8} {'phy_min':>8} {'zoo_end':>8} {'zoo_min':>8} "
           f"{'pel_end':>8} {'gad_end':>8} {'por_end':>8} "
           f"{'dead0':>6} {'deadM':>6} {'deadE':>6} {'ext':>22}")
    if a.mass_balance:
        # Under --mass-balance the growth term is capped by the reserve,
        # so growth_rate * <s-u> * 1460 is the WISH and not the spend.
        # growth_mass_balance.py reads production off the losses, which
        # is the estimator that stays valid. Section 117.
        print("NOTE: realPB is the growth-term WISH, not production, with "
              "--mass-balance.")
    print(hdr); print('-' * len(hdr))
    for g in a.values:
        for seed in range(a.seed_base, a.seed_base + a.seeds):
            t0 = time.time()
            r = run(g, seed, a.ticks, tuple(a.grid), a.behaviour, a.spawn, mort,
                    a.phyto_growth, a.phyto_seed,
                    a.mass_balance)
            ext = ','.join(f"{k[:3]}@{v}" for k, v in
                           [(k[:-4], r[k]) for k in r if k.endswith('_ext')]
                           if v) or '-'
            print(f"{g:>7.4g} {seed:>4} {r['ceiling_PB']:>7.1f} "
                  f"{r['realised_PB']:>7.1f} {r['mean_surplus']:>7.4f} "
                  f"{r['phytoplankton_end']:>8.3f} {r['phytoplankton_min']:>8.3f} "
                  f"{r['zooplankton_end']:>8.3f} {r['zooplankton_min']:>8.3f} "
                  f"{r.get('pelagic_fish_end', float('nan')):>8.3f} "
                  f"{r.get('gadoids_end', float('nan')):>8.3f} "
                  f"{r.get('porpoises_end', float('nan')):>8.3f} "
                  f"{r['dead_start']:>6} {r['dead_mid']:>6} {r['dead_end']:>6} "
                  f"{ext:>22}  [{time.time()-t0:.0f}s]", flush=True)


main()
