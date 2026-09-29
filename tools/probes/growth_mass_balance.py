"""Is the growth term mass-balanced against what was actually eaten?

Measures, over a window at equilibrium: phytoplankton grazed (ton/tick),
zooplankton biomass PRODUCED by the growth term (ton/tick, positive part
only), and zooplankton losses. The implied gross growth efficiency is
produced / grazed; copepod GGE in the literature is ~20-40 %.
"""
import sys, argparse
import numpy as np
sys.path.insert(0, '.')
from inference import build_env
from lib.diagnostics import viability
from lib.environments.ecosystem_env.population_change import (
    DEFAULT_MASS_BALANCE)
from lib.environments.ecosystem_env.currents import (
    add_current_arguments, current_options)

p = argparse.ArgumentParser()
p.add_argument('--phyto-growth', type=float, default=None)
p.add_argument('--zoo-growth', type=float, default=0.0913)
p.add_argument('--ticks', type=int, default=3000)
p.add_argument('--window', type=int, default=300)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--zoo-rm', dest='zoo_rm', type=float, default=None,
               help='Override zooplankton resting_metabolism '
                    '(MJ/t/tick). Section 118.')
# Mirrors the engine default, which flipped ON in section 120.
# Hard-coding False here silently measured the legacy term.
p.add_argument('--mass-balance', dest='mass_balance',
               action='store_true',
               default=DEFAULT_MASS_BALANCE,
               help='No-op; the engine default (section 120).')
p.add_argument('--no-mass-balance', dest='mass_balance',
               action='store_false',
               help='Legacy growth term, pre-section-116.')
p.add_argument('--zoo-fc', dest='zoo_fc', type=float, default=None,
               help='Override zooplankton feeding_cost (activity '
                    'multiplier on resting_metabolism). Section 124.')
p.add_argument('--zoo-me', dest='zoo_me', type=float, default=None,
               help='Override zooplankton max_energy_reserve (MJ/t). The '
                    'spawned reserve is rescaled so s_X(0) is unchanged. '
                    'Section 124.')
p.add_argument('--behaviour', choices=('greedy', 'policy'),
               default='greedy',
               help="'greedy' (default): frozen greedy, colocated spawn - "
                    "the mechanics bench. 'policy': the trained policies "
                    "in --checkpoints on the configured spawn (section 123 "
                    "- the frozen corner is not a proxy for them).")
p.add_argument('--checkpoints', default=None,
               help="Checkpoint dir for --behaviour policy.")
p.add_argument('--migration', choices=('on', 'off'), default='off')
add_current_arguments(p)
a = p.parse_args()

env = build_env('mareld2.yaml', (60, 60), seed=a.seed, verbose=False,
                apply_natural_mortality=True,
                migration=(a.migration == 'on'),
                currents=current_options(a), tick_hours=6,
                mass_balance=a.mass_balance)
zoo, phyto = env.fgs['zooplankton'], env.fgs['phytoplankton']
zoo.growth_rate = a.zoo_growth
if a.phyto_growth is not None:
    phyto.growth_rate = a.phyto_growth
u = float(zoo.params.get('maintenance_level', 0) or 0)
g = float(zoo.growth_rate)
nm = float(zoo.natural_mortality)
if a.zoo_rm is not None:
    zoo.resting_metabolism = float(a.zoo_rm)
    zoo.params['resting_metabolism'] = float(a.zoo_rm)
if a.zoo_fc is not None:
    zoo.params['feeding_cost'] = float(a.zoo_fc)
if a.zoo_me is not None:
    # Keep the spawned fill level s_X = R / (B * ME) where it was.
    zoo.energy_reserve = (zoo.energy_reserve * np.float32(
        a.zoo_me / float(zoo.max_energy_reserve))).astype(
            zoo.energy_reserve.dtype)
    zoo.max_energy_reserve = float(a.zoo_me)
    zoo.params['max_energy_reserve'] = float(a.zoo_me)
if a.behaviour == 'policy':
    if not a.checkpoints:
        p.error('--behaviour policy needs --checkpoints')
    provider = viability.install_behaviour(
        env, 'policy', seed=a.seed, checkpoint_dir=a.checkpoints)
else:
    provider = viability.install_behaviour(env, 'greedy', seed=a.seed)
    viability.colocate_spawn(env)
# install_behaviour rebuilds the static caches, which is where
# dm_resting_metabolism is read from the FG - assert the override landed
# rather than trusting it.
_zi = list(env.dm_ids).index('zooplankton')
if a.zoo_rm is not None:
    assert abs(float(env.dm_resting_metabolism[_zi]) - a.zoo_rm) < 1e-6, \
        'resting_metabolism override did not reach the cache'
if a.zoo_fc is not None:
    assert abs(float(env.dm_cost_eat[_zi]) - a.zoo_fc) < 1e-6, \
        'feeding_cost override did not reach the cache'
if a.zoo_me is not None:
    assert abs(float(env.dm_max_energy_reserve[_zi]) - a.zoo_me) < 1e-3, \
        'max_energy_reserve override did not reach the cache'
RM = float(env.dm_resting_metabolism[_zi])
COST_EAT = float(env.dm_cost_eat[_zi])
ME = float(env.dm_max_energy_reserve[_zi])

acc = dict(grazed=0.0, prod=0.0, starv=0.0, nat=0.0, pred=0.0, n=0,
           zoo_b=0.0, phy_b=0.0)
prev = {'phyto_loss': 0.0, 'zoo_starv': 0.0, 'zoo_pred': 0.0}
start = a.ticks - a.window


def on_tick(k, total):
    pl = float(env.loss_predation.get('phytoplankton', 0.0))
    zs = float(env.loss_starvation.get('zooplankton', 0.0))
    zp = float(env.loss_predation.get('zooplankton', 0.0))
    if k > start:
        b = np.asarray(zoo.biomass, dtype=np.float64)
        s = np.asarray(zoo.energy_level, dtype=np.float64)
        # The growth term as the engine applies it: mortality first, then
        # B*(1-nm)*g*(s-u) on the positive cells only.
        bd = b * (1.0 - nm) * g * (s - u)
        acc['grazed'] += pl - prev['phyto_loss']
        acc['prod'] += float(bd[bd > 0].sum())
        acc['starv'] += zs - prev['zoo_starv']
        acc['pred'] += zp - prev['zoo_pred']
        acc['nat'] += float(b.sum()) * nm
        acc['zoo_b'] += float(b.sum())
        acc['phy_b'] += float(phyto.biomass.sum())
        acc['n'] += 1
    prev['phyto_loss'], prev['zoo_starv'], prev['zoo_pred'] = pl, zs, zp


viability.run_rollout(env, a.ticks, provider, on_tick)
n = acc['n']
grazed, prod = acc['grazed']/n, acc['prod']/n
zb, pb = acc['zoo_b']/n, acc['phy_b']/n
print(f"behaviour={a.behaviour}, feeding_cost {COST_EAT:g}, "
      f"max_energy_reserve {ME:g} MJ/t (ME/ec {ME/4500:.3f})")
print(f"mass_balance={a.mass_balance}, "
      f"phyto growth_rate {phyto.growth_rate:g}, zoo growth_rate {g:g}, "
      f"window = last {n} ticks of {a.ticks}")
print(f"  standing stock        zoo {zb:>10.0f} t   phyto {pb:>9.0f} t"
      f"   zoo:phyto {zb/pb:>5.2f} : 1")
print(f"  phyto grazed          {grazed:>10.1f} t/tick")
losses = (acc['starv'] + acc['nat'] + acc['pred']) / n
print(f"  zoo lost: starvation  {acc['starv']/n:>10.1f}  natural "
      f"{acc['nat']/n:>8.1f}  predation {acc['pred']/n:>8.1f}"
      f"   sum {losses:>9.1f} t/tick")
# Production is read off the LOSSES, which is exact at equilibrium and is
# the only estimator valid under --mass-balance: there the growth term is
# capped by the reserve, so B*(1-nm)*g*(s-u) is the WISH, not the spend.
# With the flag off the two agree to a few per cent, which is the check.
print(f"  zoo production: from losses {losses:>8.1f} t/tick"
      f"   (growth-term wish {prod:>8.1f}"
      f"{' - NOT the spend, the reserve caps it' if a.mass_balance else ''})")
# Respiration as a share of assimilated energy: the literature
# cross-check is ~50 % of ingestion in carbon terms (section 118).
ingested = grazed * 2000.0                 # MJ/tick in the prey eaten
assim_in = ingested * 0.65                 # MJ/tick reaching the reserve
# Charges every tonne at the eat cost. Greedy also moves mass (at
# movement_cost < feeding_cost) and a policy also rests, so this is an
# upper bound in both modes; the overshoot grows with feeding_cost.
respired = zb * RM * COST_EAT              # MJ/tick burnt, eat-cost mix
prod_mj = losses * 4500.0                  # MJ/tick built into new biomass
print(f"  resting_metabolism {RM:g} MJ/t/tick = {RM*4:.1f} MJ/t/day = "
      f"{100*RM*4/4500:.3f} %/day of body energy")
# Literature bases are per INGESTION, so report that and keep the
# assimilated share alongside it. assimilation_factor is 0.65, so
# respiration + production must come to 65 % of ingestion if the tick's
# energy budget closes; whatever is missing went somewhere else
# (the clip at ME*B, or the reserve scaled away with starved biomass).
r_ing = 100 * respired / max(ingested, 1e-9)
p_ing = 100 * prod_mj / max(ingested, 1e-9)
print(f"  of INGESTED energy: respired {r_ing:>5.1f} %  built {p_ing:>5.1f} %"
      f"  -> accounted {r_ing + p_ing:>5.1f} % of the 65 % assimilated"
      f"  (gap {65.0 - r_ing - p_ing:>5.1f} %)")
print(f"  literature: respiration ~50 % of ingestion, GGE 20-40 %"
      f"   (respired: upper bound, all mass at the eat cost)")
ceiling = 100.0 * 2000.0 * 0.65 / 4500.0
print(f"  IMPLIED gross growth efficiency = production / grazed = "
      f"{100*losses/max(grazed, 1e-9):>5.1f} %   (copepod GGE 20-40 %)")
print(f"  ceiling from the library: energy_gain / energy_content = "
      f"2000 * 0.65 / 4500 = {ceiling:.1f} %")
