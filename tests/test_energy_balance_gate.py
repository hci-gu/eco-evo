"""Structural guard for the DM energy-balance gate (Section 23 / 67).

The gate asks a single question per decision maker: can it cover its own
feeding metabolism at ITS OWN maintenance level u_X, where the hunger gate
lets through h(u_X) of the physiological intake ceiling? Since section
133 appetite is full up to u_X (h(u_X) = 1), so the question is whether
the FG covers its feeding metabolism at its full intake ceiling.

    realized(u_X) = h(u_X) * max_intake_rate * energy_content * assim
    net_eat       = realized(u_X) - feeding_cost * resting_metabolism > 0

Equivalently: ceiling/Rest_X > feeding_cost/h(u_X).

The pre-Section-67 gate evaluated the intake side at h = 1, a state that
requires s_X = 0 - i.e. an already-dead animal - and therefore admitted
configurations that shrink at ~2 %/tick with unlimited prey available.
This module is the same kind of shape guard as
tests/test_holling_response.py: it locks the CONTRACT, not a calibration.

The hunger gate is h = 1 - s for every FG. The per-FG ``satiation_scale``
of Sections 69 and 92 (default 0.8, porpoises 0.9 then 1.37) was removed
in section 130, together with the tests that pinned it.

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
    REFERENCE_RESTING_COST,
    evaluate_energy_balance,
    hunger_at,
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


def _ceiling_mat(env):
    """Holling ceiling per pair: 1/h, or the attack rate when h = 0.

    Since section 134 a pair can carry its own attack rate (half-
    saturation 1/(a h)), so ``max_intake_mat`` is no longer the ceiling.
    """
    a = np.asarray(env.max_intake_mat, dtype=np.float64)
    h = np.asarray(env.handling_time_mat, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(h > 0.0, 1.0 / np.where(h > 0.0, h, 1.0), a) * (a > 0)


def _balance_for(env, i, fg_id):
    """Gate result for DM row ``i`` using the live runtime caches."""
    a = _ceiling_mat(env)[i]
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


def test_appetite_is_full_up_to_maintenance():
    """h = min(1, (1 - s)/(1 - u)) (section 133)."""
    assert hunger_at(0.0, 0.5) == 1.0
    assert hunger_at(0.5, 0.5) == 1.0
    assert hunger_at(0.75, 0.5) == pytest.approx(0.5)
    assert hunger_at(1.0, 0.5) == 0.0
    assert hunger_at(0.5, 1.0) == 0.0          # no feeding window
    assert hunger_at(0.3) == pytest.approx(0.7)  # u = 0: h = 1 - s


def test_hunger_at_maintenance_is_the_evaluation_point():
    b = evaluate_energy_balance(**HEALTHY)
    expected = 1.0  # full appetite at u_X (section 133)
    assert b.hunger_at_maintenance == pytest.approx(expected)
    assert b.realized_intake == pytest.approx(
        expected * HEALTHY['intake_ceiling'])
    assert b.realized_intake == pytest.approx(b.intake_ceiling)


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
    """u_X >= 1 means intake is identically 0 at u_X."""
    b = evaluate_energy_balance(**dict(HEALTHY, maintenance_level=1.0))
    assert not b.ok
    assert b.hunger_at_maintenance == 0.0
    assert any("hunger gate is fully closed" in m for m in b.failures), \
        b.failures


def test_maintenance_gate_equals_the_full_hunger_gate():
    """With full appetite at u_X the two views coincide (section 133).

    The Section 67 failure mode (passes at h = 1, starves at u_X) needed
    h(u_X) < 1 and cannot occur any more. Porpoise numbers: ceiling
    0.035 * 7500 * 0.9 = 236.25, resting_metabolism 90, feeding_cost 1.4.
    """
    b = evaluate_energy_balance(
        "porpoise_like", intake_ceiling=236.25, resting_metabolism=90.0,
        maintenance_level=0.5, feeding_cost=1.4, resting_cost=1.0,
        movement_cost=1.6)
    assert b.hunger_at_maintenance == pytest.approx(1.0)
    assert b.realized_intake == pytest.approx(b.intake_ceiling)
    assert b.net_eat == pytest.approx(b.net_eat_at_h1)
    assert b.net_eat == pytest.approx(110.25)
    assert b.ok, b.failures


def test_break_even_condition_is_the_headroom_ratio():
    """net_eat > 0  <=>  ceiling/Rest_X > feeding_cost/h(u_X)."""
    for fc in [1.0, 2.0, 3.25, 3.3, 4.0]:
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
    a = _ceiling_mat(env)
    gain = np.asarray(env.energy_gain_mat, dtype=np.float64)
    ceilings = (a * gain).max(axis=1)
    zero = [fg_id for i, fg_id in enumerate(env.dm_ids)
            if ceilings[i] <= 0.0]
    assert not zero, (
        "decision makers with no positive intake ceiling (missing prey, "
        "max_intake_rate or energy_content): " + ", ".join(zero))


# ---------- The hunger gate closes at a full reserve (section 130) ----------

def test_hunger_gate_closes_exactly_at_a_full_reserve():
    """No satiation_scale any more: h = 1 - s, so appetite ends at s = 1."""
    ratios = np.array([[0.0, 0.5, 0.99, 1.0]])
    fg = FunctionalGroup("probe", {'max_energy_reserve': 1000.0,
                                   'satiation_scale': 1.37})  # ignored
    fg.biomass = np.ones_like(ratios)
    fg.energy_reserve = fg.biomass * ratios * fg.max_energy_reserve
    np.testing.assert_allclose(np.asarray(fg.get_hunger()).ravel(),
                               1.0 - ratios.ravel())
    assert not hasattr(fg, "satiation_scale")
    # With a maintenance level the appetite is full up to it (section 133).
    fg2 = FunctionalGroup("probe", {'max_energy_reserve': 1000.0,
                                    'maintenance_level': 0.5})
    r2 = np.array([[0.0, 0.5, 0.75, 1.0]])
    fg2.biomass = np.ones_like(r2)
    fg2.energy_reserve = fg2.biomass * r2 * fg2.max_energy_reserve
    np.testing.assert_allclose(np.asarray(fg2.get_hunger()).ravel(),
                               [1.0, 1.0, 0.5, 0.0])
    assert [hunger_at(x, 0.5) for x in r2.ravel()] == [1.0, 1.0, 0.5, 0.0]


def test_porpoises_break_even_with_a_usable_margin():
    """The live porpoise budget (section 133: full appetite at u_X).

    ceiling = max_intake_rate 0.035 * herring 7500 * assim 0.9 = 236.25
    against feed_cost 1.4 * 90 = 126.
    """
    b = evaluate_energy_balance(
        "porpoise_like", intake_ceiling=0.035 * 7500.0 * 0.9,
        resting_metabolism=90.0, maintenance_level=0.5, feeding_cost=1.4,
        resting_cost=1.0, movement_cost=1.6)
    assert b.hunger_at_maintenance == pytest.approx(1.0)
    assert b.net_eat == pytest.approx(110.25, abs=1e-6)
    assert b.max_feeding_cost == pytest.approx(2.625, abs=1e-6)
    assert b.ok, b.failures


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
    u_x = params.get('maintenance_level', 0.0)
    h_u = hunger_at(u_x, u_x)
    cost = (float(params.get('resting_metabolism', 0.0))
            * float(params.get('feeding_cost', 1.0)))
    a = float(np.max(_ceiling_mat(env)[i, menu]))
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


KASTELEIN_MAX = 0.095   # highest measured ration, fraction of body mass per day
KASTELEIN_MIN = 0.04


def _porpoise_cost_and_quality(env):
    i = env.dm_ids.index('porpoises')
    params = env.fgs['porpoises'].params
    cost = (float(params.get('resting_metabolism', 0.0))
            * float(params.get('feeding_cost', 1.0)))
    menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
    a = float(np.max(_ceiling_mat(env)[i, menu]))
    quality = {env.global_fg_order[j]: float(env.energy_gain_mat[i, j])
               for j in menu}
    return cost, a, quality


def test_live_project_porpoises_need_clupeids_at_a_real_ration():
    """The junk-food hypothesis (MacLeod et al. 2007; Spitz et al. 2012).

    With full appetite at u_X (section 133) the physiological ceiling
    alone no longer rations, so the test is posed at the highest ration
    actually measured (Kastelein, 9.5 % of body mass per day): a pure
    lean-gadoid diet must NOT pay for the feeding metabolism there, and
    the minimum clupeid share must stay within the 50-70 % that Kattegat /
    Skagerrak stomachs contain (it may be lower - porpoises eat more
    clupeids than the minimum).
    """
    env = _build_live_env()
    if 'porpoises' not in env.dm_ids:
        pytest.skip("porpoises not in the project configuration")
    cost, _a, quality = _porpoise_cost_and_quality(env)
    best_id = max(quality, key=quality.get)
    worst_id = min(quality, key=quality.get)
    assert best_id == 'pelagic_fish', best_id
    ration = KASTELEIN_MAX / TICKS_PER_DAY
    need_q = cost / ration
    assert quality[worst_id] < need_q <= quality[best_id], (
        f"{quality} against a requirement of {need_q:.0f} MJ/t at the "
        f"Kastelein maximum ration")
    share = ((need_q - quality[worst_id])
             / (quality[best_id] - quality[worst_id]))
    assert 0.0 < share <= 0.70, share


def test_live_project_porpoise_ration_is_the_literature_one():
    """The ceiling caps; the break-even ration is the literature number.

    ``max_intake_rate`` is a *physiological* ceiling and must sit above
    the highest ration ever measured (Kastelein: 4-9.5 % of body mass per
    day). The ration that just pays the feeding metabolism on clupeids
    must land inside that window (section 133; before it, the ration was
    h(u_X) * ceiling, Section 92).
    """
    env = _build_live_env()
    if 'porpoises' not in env.dm_ids:
        pytest.skip("porpoises not in the project configuration")
    cost, a, quality = _porpoise_cost_and_quality(env)
    ceiling_pct = a * TICKS_PER_DAY * 100.0
    ration_pct = cost / max(quality.values()) * TICKS_PER_DAY * 100.0
    assert 10.0 <= ceiling_pct <= 16.0, (
        f"intake ceiling {ceiling_pct:.1f} %bm/day is not a physiological "
        f"ceiling for a harbour porpoise")
    assert KASTELEIN_MIN * 100 <= ration_pct <= KASTELEIN_MAX * 100, (
        f"break-even ration {ration_pct:.1f} %bm/day is outside the "
        f"Kastelein window")
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
