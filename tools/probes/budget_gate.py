"""Energy break-even per decision maker, over the DIET rather than per pair.

What the engine actually does
-----------------------------
``predation.apply_predation`` sums the intake over every prey on the
menu and hands the predator ONE energy gain; the hunger gate and the
feeding cost are applied once, to that total. The budget a DM has to
close is therefore a property of its *ration*, not of any single pair.

A DM that eats at 100 % of the time, at the reserve level where it just
breaks even (s_X = u_X), takes in

    ration = a * h(u_X)                         [t prey / t pred / tick]
    gain   = ration * quality                   [MJ / t pred / tick]

where ``quality`` is the ration-weighted mean assimilated energy content
of the prey it eats, and pays

    cost = resting_metabolism * feeding_cost    [MJ / t pred / tick]

so survival by eating alone requires

    quality >= need_quality = cost / ration,
    h(u_X)  = max(0, 1 - u_X / satiation_scale).

Why the per-pair form was wrong (Section 74.2b, superseded)
-----------------------------------------------------------
This probe used to evaluate the identity pair by pair and label a pair
INFEASIBLE when its own ``sat_min`` exceeded 1. For
``porpoises -> gadoids`` it reported 1.227 and Section 74.2b concluded
that no cod density whatsoever can pay for a porpoise, i.e. that the
pair was a calibration bug.

It is not a bug and the pair is not "pure loss". It is the junk-food
hypothesis (MacLeod et al. 2007; Spitz et al. 2012): a porpoise cannot
live on lean gadoid ALONE, and the literature says the same. What the
model must be checked against is the minimum share of the high-quality
prey in the ration - reported below as ``min share`` - and Kattegat /
Skagerrak stomach data (50-70 % clupeids by mass) clears it.

Printed next to each requirement is the density the live probe world
actually delivers, so feasible-but-unreachable stays distinguishable
from infeasible.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import train as train_mod
from lib.config.config_loader import project_tick_hours
from lib.world.energy_balance import resolve_satiation_scale
from lib.world.tick_time import tick_label, ticks_per_day

PROJECT = 'mareld2.yaml'
# %bm/day is the only real-time figure in this probe, so it is the
# only thing the project's tick length changes here. Section 97.
TICK_HOURS = project_tick_hours(PROJECT)
TICKS_PER_DAY = ticks_per_day(TICK_HOURS)

H = W = 16
builder = train_mod._ProbeEnvBuilder(
    project_path=PROJECT, grid_size=(H, W),
    apply_natural_mortality=True, migration=True)
env = builder()
env._build_static_caches()


def _pred_costs(i, fg_id):
    """Resting metabolism and the eat-cost multiplier, from the caches.

    ``feeding_cost`` left ``FunctionalGroup`` when the action costs
    moved into the env caches (``dm_cost_eat``), which is what made this
    probe unrunnable - and Section 74's numbers unreproducible - for
    several arcs.
    """
    rm = float(env.dm_resting_metabolism[i])
    c_eat = float(env.dm_cost_eat[i])
    params = env.fgs[fg_id].params
    u_x = float(params.get('maintenance_level', 0.0) or 0.0)
    scale = float(resolve_satiation_scale(params.get('satiation_scale')))
    return rm, c_eat, u_x, scale


print("=== per (predator, prey): what ONE prey alone would have to do ===")
print("A row that does not clear the bar on its own is low-quality food,")
print("not pure loss - the verdict is the per-predator block below.")
print()
print(f"{'predator':>14} {'prey':>18} {'u_x':>5} {'sat_s':>6} {'h(u)':>6} "
      f"{'cost':>7} {'a*gain':>8} {'sat_min':>8} {'need_vis':>9} "
      f"{'B/cell':>8} {'max':>8} {'alone':>12}")

for i, pid in enumerate(env.dm_ids):
    rm, c_eat, u_x, scale = _pred_costs(i, pid)
    h_u = max(0.0, 1.0 - u_x / scale)
    cost = rm * c_eat
    for j, prey_id in enumerate(env.global_fg_order):
        if not env.eat_static_mask[i, j]:
            continue
        a = float(env.max_intake_mat[i, j])
        ht = float(env.handling_time_mat[i, j])
        gain_t = float(env.energy_gain_mat[i, j])
        max_gain = a * gain_t
        if h_u <= 0.0 or max_gain <= 0.0:
            sat_min = float('inf')
        else:
            sat_min = cost / (max_gain * h_u)
        Bp = env.fgs[prey_id].biomass
        occ = Bp > 0
        mean_occ = float(Bp[occ].mean()) if occ.any() else 0.0
        mx = float(Bp.max())
        if sat_min >= 1.0:
            need = float('inf')
            verdict = "not alone"
        else:
            need = sat_min / ((1.0 - sat_min) * a * ht) if a * ht > 0 else 0.0
            verdict = "ok" if mx >= need else "unreachable"
        print(f"{pid:>14} {prey_id:>18} {u_x:>5.2f} {scale:>6.2f} "
              f"{h_u:>6.3f} {cost:>7.1f} {max_gain:>8.1f} {sat_min:>8.3f} "
              f"{need:>9.3f} {mean_occ:>8.2f} {mx:>8.2f} {verdict:>12}")

print()
print("=== per predator: the budget the engine actually closes ===")
print(f"{'predator':>14} {'ration':>9} {'%bm/day':>8} {'cost':>7} "
      f"{'need_q':>9} {'best_q':>9} {'worst_q':>9} {'min share':>10} "
      f"{'verdict':>12}")

for i, pid in enumerate(env.dm_ids):
    rm, c_eat, u_x, scale = _pred_costs(i, pid)
    h_u = max(0.0, 1.0 - u_x / scale)
    cost = rm * c_eat
    menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
    if not menu or h_u <= 0.0:
        print(f"{pid:>14} {'-':>9} {'-':>8} {cost:>7.1f} "
              f"{'-':>9} {'-':>9} {'-':>9} {'-':>10} {'NO INTAKE':>12}")
        continue

    # The physiological ration is a per-predator property; the library
    # convention is one max_intake_rate per predator, so take the row
    # maximum rather than assuming the pairs agree.
    a = float(np.max(env.max_intake_mat[i, menu]))
    ration = a * h_u
    quality = {env.global_fg_order[j]: float(env.energy_gain_mat[i, j])
               for j in menu}
    need_q = cost / ration if ration > 0 else float('inf')
    best_id = max(quality, key=quality.get)
    worst_id = min(quality, key=quality.get)
    best_q, worst_q = quality[best_id], quality[worst_id]

    if best_q < need_q:
        share, verdict = float('nan'), "INFEASIBLE"
    elif worst_q >= need_q:
        share, verdict = 0.0, "any diet"
    else:
        share = (need_q - worst_q) / (best_q - worst_q)
        verdict = "mix needed"
    print(f"{pid:>14} {ration:>9.5f} {ration*TICKS_PER_DAY*100:>7.1f}% "
          f"{cost:>7.1f} {need_q:>9.0f} {best_q:>9.0f} {worst_q:>9.0f} "
          f"{share:>10.3f} {verdict:>12}")
    if verdict == "mix needed":
        print(f"{'':>14} at least {share*100:.1f} % {best_id} in the ration, "
              f"the rest {worst_id}")

print()
print(f"Units: ration in t prey / t predator / {tick_label(TICK_HOURS)}; "
      f"cost and quality in")
print("MJ / t. quality is assimilation_factor * energy_content, i.e. the")
print("energy the predator keeps per tonne eaten.")

print()
print("Sensitivity for porpoises -> pelagic_fish (the pair in question):")
i = env.dm_ids.index('porpoises')
j = env.global_fg_order.index('pelagic_fish')
a = float(env.max_intake_mat[i, j])
ht = float(env.handling_time_mat[i, j])
gain_t = float(env.energy_gain_mat[i, j])
cost = float(env.dm_resting_metabolism[i]) * float(env.dm_cost_eat[i])
print(f"{'u_x':>6} {'sat_scale':>10} {'h(u)':>7} {'sat_min':>8} "
      f"{'need_visible_t/cell':>20}")
for u_x in (0.5, 0.4, 0.3):
    for scale in (0.9, 1.0, 1.37):
        h_u = max(0.0, 1.0 - u_x / scale)
        sat_min = cost / (a * gain_t * h_u)
        need = (sat_min / ((1.0 - sat_min) * a * ht)
                if sat_min < 1.0 else float('inf'))
        print(f"{u_x:>6.2f} {scale:>10.2f} {h_u:>7.3f} {sat_min:>8.3f} "
              f"{need:>20.3f}")
