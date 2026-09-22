"""Structural guard for the DM energy-balance gate (Section 23 / 67).

The gate asks a single question per decision maker: can it cover its own
feeding metabolism at ITS OWN maintenance level u_X, where the hunger gate
only lets through h(u_X) = 1 - u_X/satiation_scale of the physiological
intake ceiling?

    realized(u_X) = h(u_X) * max_intake_rate * energy_content * assim
    net_eat       = realized(u_X) - feeding_cost * resting_metabolism > 0

Equivalently: ceiling/Rest_X > feeding_cost/h(u_X).

The pre-Section-67 gate evaluated the intake side at h = 1, a state that
requires s_X = 0 - i.e. an already-dead animal - and therefore admitted
configurations that shrink at ~2 %/tick with unlimited prey available.
This module is the same kind of shape guard as
tests/test_holling_response.py: it locks the CONTRACT, not a calibration.

``satiation_scale`` is per-FG (Section 69, default
``HUNGER_SATIATION_SCALE`` = 0.8), so the live-project checks must read
it from the FG params - otherwise the gate silently evaluates a
different hunger window than the runtime.

HISTORY: test_live_project_decision_makers_can_break_even was
deliberately RED for `porpoises` when it was added in Section 67
(net_eat = -16.31 MJ/ton/tick, max allowed feeding_cost 1.22 vs. the
configured 1.40). Section 69 fixed the parameters, not the assertion:
herring energy_content 6500 -> 7500 and porpoise satiation_scale
0.8 -> 0.9. Do not weaken the gate; fix the parameters.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.config.config_loader import setup_full_mareld_mvp  # noqa: E402
from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.world.energy_balance import (  # noqa: E402
    HUNGER_SATIATION_SCALE,
    REFERENCE_RESTING_COST,
    evaluate_energy_balance,
    hunger_at,
    resolve_satiation_scale,
)
from lib.world.functional_group import FunctionalGroup  # noqa: E402


# A generic, comfortably viable decision maker used as the baseline for
# the single-parameter perturbations below.
HEALTHY = dict(
    fg_id="viable",
    intake_ceiling=300.0,
    resting_metabolism=10.0,
    maintenance_level=0.5,
    feeding_cost=1.5,
    resting_cost=1.0,
    movement_cost=2.0,
)


def _build_live_env():
    fgs = setup_full_mareld_mvp(grid_size=(8, 8), seed=0, spawn_seed=0)
    grid_config = {'width': 8, 'height': 8, 'cell_size': 1000.0,
                   'tick_duration': 6.0}
    env = EcosystemEnvironment(grid_config, fgs, {})
    env._build_static_caches()
    return env


def _balance_for(env, i, fg_id):
    """Gate result for DM row ``i`` using the live runtime caches."""
    a = np.asarray(env.max_intake_mat, dtype=np.float64)[i]
    gain = np.asarray(env.energy_gain_mat, dtype=np.float64)[i]
    row = a * gain
    j = int(np.argmax(row))
    params = env.fgs[fg_id].params
    return evaluate_energy_balance(
        fg_id,
        intake_ceiling=float(row[j]),
        resting_metabolism=params.get('resting_metabolism', 0.0),
        maintenance_level=params.get('maintenance_level', 0.0),
        feeding_cost=params.get('feeding_cost', 1.0),
        resting_cost=params.get('resting_cost', REFERENCE_RESTING_COST),
        movement_cost=params.get('movement_cost'),
        best_prey=env.global_fg_order[j],
        # Per-FG hunger window (Section 69). Must be threaded through or
        # the gate evaluates a different h(u_X) than get_hunger() does.
        hunger_scale=params.get('satiation_scale'),
    )


# ---------- The hunger gate itself ----------

def test_hunger_at_matches_functional_group_runtime():
    """``hunger_at`` must mirror ``FunctionalGroup.get_hunger`` exactly."""
    fg = FunctionalGroup("probe", {'max_energy_reserve': 1000.0})
    fg.biomass = np.ones((1, 5))
    ratios = np.array([[0.0, 0.25, 0.5, 0.8, 1.0]])
    fg.energy_reserve = fg.biomass * ratios * fg.max_energy_reserve
    runtime = np.asarray(fg.get_hunger()).ravel()
    scalar = [hunger_at(s) for s in ratios.ravel()]
    np.testing.assert_allclose(runtime, scalar)


def test_full_hunger_requires_an_empty_reserve():
    """h = 1 only at s_X = 0 - the state the old gate assumed.

    This is why the gate may not be evaluated at h = 1: no surviving
    population sits there.
    """
    assert hunger_at(0.0) == 1.0
    assert hunger_at(HUNGER_SATIATION_SCALE) == 0.0
    assert hunger_at(1.0) == 0.0
    assert 0.0 < hunger_at(0.5) < 1.0


def test_hunger_at_maintenance_is_the_evaluation_point():
    b = evaluate_energy_balance(**HEALTHY)
    expected = 1.0 - HEALTHY['maintenance_level'] / HUNGER_SATIATION_SCALE
    assert b.hunger_at_maintenance == pytest.approx(expected)
    assert b.realized_intake == pytest.approx(
        expected * HEALTHY['intake_ceiling'])
    assert b.realized_intake < b.intake_ceiling


# ---------- The gate contracts ----------

def test_healthy_decision_maker_passes_cleanly():
    b = evaluate_energy_balance(**HEALTHY)
    assert b.ok, b.failures
    assert not b.warnings, b.warnings
    assert b.net_eat > 0.0


def test_gate_rejects_feeding_cheaper_than_resting():
    """``feeding_cost`` is an activity multiplier RELATIVE to rest.

    Anything below ``resting_cost`` claims that hunting burns less than
    lying still, which the rest/eat/move cost model in
    ``_apply_movement`` cannot express.
    """
    cfg = dict(HEALTHY, feeding_cost=0.9)
    b = evaluate_energy_balance(**cfg)
    assert not b.ok
    assert any("below resting_cost" in m for m in b.failures), b.failures


def test_gate_rejects_a_closed_hunger_window():
    """u_X >= the satiation scale means intake is identically 0 at u_X."""
    b = evaluate_energy_balance(**dict(HEALTHY, maintenance_level=0.85))
    assert not b.ok
    assert b.hunger_at_maintenance == 0.0
    assert any("hunger gate is fully closed" in m for m in b.failures), \
        b.failures


def test_gate_catches_the_porpoise_failure_mode():
    """The Section 67 regression case: passes at h=1, starves at u_X.

    Numbers are the porpoise row of fg_library.yaml
    (max_intake_rate 0.05 * energy_content 6500 * assim 0.9 = 292.5;
    resting_metabolism 90, feeding_cost 1.4, maintenance_level 0.5).
    """
    b = evaluate_energy_balance(
        "porpoise_like", intake_ceiling=292.5, resting_metabolism=90.0,
        maintenance_level=0.5, feeding_cost=1.4, resting_cost=1.0,
        movement_cost=1.6)
    # The OLD gate (evaluated at h = 1) is comfortably satisfied ...
    assert b.net_eat_at_h1 == pytest.approx(166.5, abs=1e-6)
    assert b.net_eat_at_h1 > b.rest_cost
    # ... yet the animal cannot break even at its maintenance level.
    assert b.hunger_at_maintenance == pytest.approx(0.375)
    assert b.realized_intake == pytest.approx(109.6875)
    assert b.net_eat == pytest.approx(-16.3125)
    assert not b.ok
    assert b.max_feeding_cost == pytest.approx(1.21875)


def test_break_even_condition_is_the_headroom_ratio():
    """net_eat > 0  <=>  ceiling/Rest_X > feeding_cost/h(u_X)."""
    for fc in [1.0, 1.2, 1.21875, 1.4, 2.0]:
        b = evaluate_energy_balance(
            "sweep", intake_ceiling=292.5, resting_metabolism=90.0,
            maintenance_level=0.5, feeding_cost=fc)
        assert (b.net_eat > 0.0) == (b.headroom_ratio > b.required_ratio), (
            f"feeding_cost={fc}: net_eat={b.net_eat:+.4f} but "
            f"headroom={b.headroom_ratio:.4f} vs "
            f"required={b.required_ratio:.4f}")


def test_max_feeding_cost_is_the_exact_break_even_boundary():
    base = dict(HEALTHY)
    fc_max = evaluate_energy_balance(**base).max_feeding_cost
    just_below = evaluate_energy_balance(
        **dict(base, feeding_cost=fc_max * (1.0 - 1e-9)))
    just_above = evaluate_energy_balance(
        **dict(base, feeding_cost=fc_max * (1.0 + 1e-9)))
    assert just_below.net_eat > 0.0
    assert just_above.net_eat < 0.0
    assert not just_above.ok


def test_net_eat_is_monotone_in_the_intake_ceiling():
    """More energetic prey can never make the balance worse."""
    prev = -np.inf
    for ceiling in np.geomspace(1.0, 1e4, 50):
        b = evaluate_energy_balance(**dict(HEALTHY, intake_ceiling=ceiling))
        assert b.net_eat > prev
        prev = b.net_eat


def test_soft_gates_stay_quiet_for_a_healthy_configuration():
    b = evaluate_energy_balance(**HEALTHY)
    assert b.eat_minus_rest > 0.0
    assert b.eat_minus_move > 0.0
    assert not b.warnings, b.warnings


def test_soft_gates_report_eating_losing_to_resting_and_moving():
    """The behavioural soft gates from Section 24.A, at h(u_X).

    A ceiling far below the feeding metabolism makes eating strictly
    worse than both resting and searching, so both warning channels must
    fire (and the hard maintenance gate must fail as well). Note that
    ``eat - move <= 0`` requires ``movement_cost < feeding_cost``:
    searching is only the better option when it is the cheaper action.
    """
    b = evaluate_energy_balance(
        "starving", intake_ceiling=50.0, resting_metabolism=90.0,
        maintenance_level=0.5, feeding_cost=2.0, resting_cost=1.0,
        movement_cost=1.2)
    assert not b.ok
    assert b.eat_minus_rest < 0.0
    assert b.eat_minus_move < 0.0
    assert any("worse than resting" in m for m in b.warnings), b.warnings
    assert any("unaffordable" in m for m in b.warnings), b.warnings


# ---------- The live project ----------

def test_live_project_decision_makers_can_break_even():
    """Every DM in the project must survive at its own maintenance level.

    EXPECTED RED for `porpoises` under the current fg_library.yaml - see
    the module docstring and Section 67. Fix the parameters, do not
    relax the assertion.
    """
    env = _build_live_env()
    if env.N_dm == 0:
        pytest.skip("No decision makers in the project configuration")
    broken = []
    for i, fg_id in enumerate(env.dm_ids):
        b = _balance_for(env, i, fg_id)
        if not b.ok:
            broken.append(f"{fg_id}:\n  " + "\n  ".join(b.failures)
                          + "\n" + b.report())
    assert not broken, (
        f"{len(broken)} of {env.N_dm} decision makers cannot cover their "
        f"own feeding metabolism at the maintenance level:\n\n"
        + "\n\n".join(broken))


def test_live_project_feeding_cost_never_undercuts_resting_cost():
    """Separated from the break-even test so the two fail independently."""
    env = _build_live_env()
    offenders = []
    for fg_id in env.dm_ids:
        params = env.fgs[fg_id].params
        fc = float(params.get('feeding_cost', 1.0) or 1.0)
        cr = float(params.get('resting_cost', REFERENCE_RESTING_COST) or 0.0)
        if fc < cr:
            offenders.append(f"{fg_id}: feeding_cost={fc:g} < "
                             f"resting_cost={cr:g}")
    assert not offenders, (
        "feeding_cost is an activity multiplier relative to rest and must "
        "not be below resting_cost:\n  " + "\n  ".join(offenders))


def test_live_project_intake_ceilings_are_configured():
    """Guards against a vacuous gate: every DM needs a positive ceiling."""
    env = _build_live_env()
    a = np.asarray(env.max_intake_mat, dtype=np.float64)
    gain = np.asarray(env.energy_gain_mat, dtype=np.float64)
    ceilings = (a * gain).max(axis=1)
    zero = [fg_id for i, fg_id in enumerate(env.dm_ids)
            if ceilings[i] <= 0.0]
    assert not zero, (
        "decision makers with no positive intake ceiling (missing prey, "
        "max_intake_rate or energy_content): " + ", ".join(zero))


# ---------- Per-FG satiation scale (Section 69) ----------

def test_missing_satiation_scale_falls_back_to_the_default():
    """Legacy FGs that never declare the field must be unchanged."""
    for absent in (None, "", 0.0, 0, "not-a-number"):
        assert resolve_satiation_scale(absent) == HUNGER_SATIATION_SCALE


def test_satiation_scale_override_is_honoured():
    assert resolve_satiation_scale(0.9) == pytest.approx(0.9)
    assert hunger_at(0.5, 0.9) == pytest.approx(1.0 - 0.5 / 0.9)
    assert hunger_at(0.5, 0.8) == pytest.approx(0.375)


def test_per_fg_scale_reaches_the_runtime_hunger_gate():
    """``get_hunger`` must use the FG's own scale, not the module default.

    This is the coupling that makes the gate meaningful: if the runtime
    ignored ``satiation_scale`` the GUI would accept a configuration the
    simulation never realizes.
    """
    ratios = np.array([[0.0, 0.25, 0.5, 0.75]])
    for scale in (None, 0.9, 1.0):
        params = {'max_energy_reserve': 1000.0}
        if scale is not None:
            params['satiation_scale'] = scale
        fg = FunctionalGroup("probe", params)
        fg.biomass = np.ones_like(ratios)
        fg.energy_reserve = fg.biomass * ratios * fg.max_energy_reserve
        runtime = np.asarray(fg.get_hunger()).ravel()
        expected = [hunger_at(s, scale) for s in ratios.ravel()]
        np.testing.assert_allclose(runtime, expected, err_msg=f"{scale=}")


def test_raising_the_scale_can_only_widen_the_hunger_window():
    """Monotonicity in the scale, so the knob has an unambiguous sign."""
    prev = -np.inf
    for scale in np.linspace(0.55, 1.0, 20):
        h = hunger_at(0.5, scale)
        assert h > prev, f"h(u_X) not increasing at scale={scale:g}"
        prev = h


def test_porpoises_break_even_with_a_usable_margin():
    """Section 69 regression case, pinned to the two applied changes.

    ceiling = max_intake_rate 0.05 * herring energy_content 7500 * assim
    0.9 = 337.5; with satiation_scale 0.9 the gate lets through
    h(u_X) = 1 - 0.5/0.9 = 0.4444 of it, against feed_cost 1.4 * 90 = 126.

    The margin matters, not just the sign: the gate is derived with
    a_eff = a (unlimited visible prey), and with a*h = 1 the realized
    a_eff/a is B_vis/(1 + B_vis). net_eat = +0.56 (the herring change
    alone) would demand ~231 ton of visible herring per cell, which never
    occurs; +24 demands ~5.3 ton, i.e. one school cell.
    """
    b = evaluate_energy_balance(
        "porpoise_like", intake_ceiling=0.05 * 7500.0 * 0.9,
        resting_metabolism=90.0, maintenance_level=0.5, feeding_cost=1.4,
        resting_cost=1.0, movement_cost=1.6, hunger_scale=0.9)
    assert b.hunger_at_maintenance == pytest.approx(1.0 - 0.5 / 0.9)
    assert b.net_eat == pytest.approx(24.0, abs=1e-6)
    assert b.max_feeding_cost == pytest.approx(1.6666667, abs=1e-6)
    assert b.ok, b.failures
    assert not b.warnings, b.warnings


def test_live_project_porpoises_carry_the_scale_override():
    """Guard against the fix being reverted in fg_library.yaml only.

    Porpoises are the single DM whose ceiling/Rest_X headroom (3.25 with
    the old herring value) cannot absorb the default 0.8 gate, so the
    override must actually be present on the live FG.
    """
    env = _build_live_env()
    if 'porpoises' not in env.fgs:
        pytest.skip("porpoises not in the project configuration")
    scale = resolve_satiation_scale(
        env.fgs['porpoises'].params.get('satiation_scale'))
    assert scale > HUNGER_SATIATION_SCALE, (
        "porpoises need a satiation_scale above the default to break even "
        f"at u_X; got {scale:g}")


# ---------- The budget is closed over the DIET (Section 92) ----------
#
# ``apply_predation`` sums the intake over the whole menu and applies
# the hunger gate and the feeding cost ONCE, to the total. A per-pair
# break-even is therefore the wrong question: a prey that cannot pay for
# the predator on its own is low-quality food, not pure loss. Section
# 74.2b concluded the opposite from ``porpoises -> gadoids``
# (sat_min = 1.227) and is superseded by these two checks.

TICKS_PER_DAY = 4.0


def _diet_budget(env, i, fg_id):
    """(ration, need_quality, {prey: quality}) for DM row ``i``."""
    params = env.fgs[fg_id].params
    menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
    scale = resolve_satiation_scale(params.get('satiation_scale'))
    h_u = hunger_at(params.get('maintenance_level', 0.0), scale)
    cost = (float(params.get('resting_metabolism', 0.0))
            * float(params.get('feeding_cost', 1.0)))
    a = float(np.max(np.asarray(env.max_intake_mat, dtype=np.float64)[i, menu]))
    ration = a * h_u
    quality = {env.global_fg_order[j]: float(env.energy_gain_mat[i, j])
               for j in menu}
    return ration, (cost / ration if ration > 0 else np.inf), quality


def test_live_project_no_decision_maker_is_infeasible_on_its_whole_menu():
    """The real infeasibility test: not one pair, but the best diet.

    A DM is beyond rescue only when even a pure diet of its single best
    prey cannot cover its feeding metabolism at the maintenance level.
    """
    env = _build_live_env()
    if env.N_dm == 0:
        pytest.skip("No decision makers in the project configuration")
    starving = []
    for i, fg_id in enumerate(env.dm_ids):
        ration, need_q, quality = _diet_budget(env, i, fg_id)
        best = max(quality.values()) if quality else 0.0
        if best < need_q:
            starving.append(
                f"{fg_id}: best prey gives {best:.0f} MJ/t against a "
                f"requirement of {need_q:.0f} MJ/t")
    assert not starving, (
        "decision makers no diet at all can pay for:\n  "
        + "\n  ".join(starving))


def test_live_project_porpoises_need_a_clupeid_majority_not_a_pure_diet():
    """The junk-food hypothesis, as a number the library must satisfy.

    A porpoise cannot live on lean gadoid alone (MacLeod et al. 2007;
    Spitz et al. 2012) and the model must say so - but it must also
    stay inside what the stomach data supply: 50-70 % clupeids by mass
    in Kattegat / Skagerrak. Anything above that window would mean the
    modelled porpoise is hungrier than the real one.
    """
    env = _build_live_env()
    if 'porpoises' not in env.dm_ids:
        pytest.skip("porpoises not in the project configuration")
    i = env.dm_ids.index('porpoises')
    ration, need_q, quality = _diet_budget(env, i, 'porpoises')

    best_id = max(quality, key=quality.get)
    worst_id = min(quality, key=quality.get)
    assert best_id == 'pelagic_fish', best_id
    assert quality[worst_id] < need_q <= quality[best_id], (
        f"{quality} against a requirement of {need_q:.0f} MJ/t")

    share = ((need_q - quality[worst_id])
             / (quality[best_id] - quality[worst_id]))
    assert 0.40 <= share <= 0.70, (
        f"minimum {best_id} share in the ration is {share:.3f}, outside the "
        f"50-70 % clupeid window the stomach data supply")


def test_live_project_porpoise_ration_is_the_literature_one():
    """Section 92, point 3: the ceiling caps, the gate no longer rations.

    ``max_intake_rate`` is a *physiological* ceiling and must sit above
    the highest ration ever measured (Kastelein: 4-9.5 % of body mass
    per day); the ration realised at the maintenance level is the
    literature-anchored number and must land inside that window. Before
    Section 92 the ceiling was 20 %/day - twice the observed maximum -
    and the un-anchored ``satiation_scale`` was setting the ration.
    """
    env = _build_live_env()
    if 'porpoises' not in env.dm_ids:
        pytest.skip("porpoises not in the project configuration")
    i = env.dm_ids.index('porpoises')
    menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
    a = float(np.max(np.asarray(env.max_intake_mat, dtype=np.float64)[i, menu]))
    ration, _, _ = _diet_budget(env, i, 'porpoises')

    ceiling_pct = a * TICKS_PER_DAY * 100.0
    ration_pct = ration * TICKS_PER_DAY * 100.0
    assert 10.0 <= ceiling_pct <= 16.0, (
        f"intake ceiling {ceiling_pct:.1f} %bm/day is not a physiological "
        f"ceiling for a harbour porpoise")
    assert 4.0 <= ration_pct <= 9.5, (
        f"realised ration {ration_pct:.1f} %bm/day is outside the Kastelein "
        f"window")
    assert ration_pct < ceiling_pct


def test_live_project_break_even_margins_are_not_marginal():
    """A DM that only just clears the gate cannot realize it in the world.

    The gate assumes a_eff = a. Requiring net_eat > 5 % of feed_cost
    keeps the required prey density inside what the Holling response can
    actually deliver.
    """
    env = _build_live_env()
    if env.N_dm == 0:
        pytest.skip("No decision makers in the project configuration")
    thin = []
    for i, fg_id in enumerate(env.dm_ids):
        b = _balance_for(env, i, fg_id)
        if b.feed_cost > 0.0 and b.net_eat <= 0.05 * b.feed_cost:
            thin.append(f"{fg_id}: net_eat={b.net_eat:+.3f} vs "
                        f"5 % of feed_cost={0.05 * b.feed_cost:.3f}")
    assert not thin, (
        "decision makers whose break-even margin is within 5 % of their "
        "feeding metabolism - the gate passes but only at prey densities "
        "the functional response cannot supply:\n  " + "\n  ".join(thin))
