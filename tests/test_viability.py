"""The long-term viability rig: criterion, frozen behaviour arms, CLI.

``VIABILITY.md`` defines "viable" as four conditions per functional group
per seed - survival, floor, ceiling and stationarity - measured on long
rollouts with the behaviour frozen and no reward function involved. These
tests pin that definition down so it cannot drift silently, because every
later calibration decision is quoted against it.

Contracts asserted:

1.  The horizon and the final window follow from ``years``/``window_frac``.
2.  Nonsensical criteria are rejected at construction.
3.  A flat trajectory at the spawn level passes all four conditions.
4.  A settled equilibrium below ``floor`` fails, and says so.
5.  Runaway growth fails on the ceiling.
6.  A trajectory that is still doubling fails on stationarity, even when
    it is nowhere near the floor.
7.  Extinction inside the horizon fails, and reports the tick.
8.  Groups absent at spawn are outside the criterion entirely.
9.  The worst seed decides an arm, and a failing seed is the one reported.
10. The summary is strict JSON (no Infinity) and carries the verdict.
11. ``run_rollout`` books the spawn level at t=0 and one sample per tick.
12. The ``neutral`` arm installs no network and acts uniformly at random
    over the legal actions.
13. The ``eat`` arm never moves and spends all its mass on legal prey.
14. Unknown arms, and ``policy`` without a checkpoint, are refused.
15. The CLI accepts the project's ``n*m`` grid spelling, rejects unknown
    arms, and lets ``--ticks`` override ``--years``.
16. End to end on a tiny grid, the exit code carries the verdict.

The rig is two-factorial (see the module docstring of
``lib/diagnostics/viability.py``), and the second factor carries its own
contracts:

17. Only a normative arm may decide; ``neutral``/``random`` alone report
    no verdict rather than a spurious NOT VIABLE, and ``greedy`` wins
    over ``eat`` when both were run.
18. ``greedy`` climbs the intake payoff, eats at a local optimum, never
    rests voluntarily, and never moves off-grid or against the mask.
19. Travelling must pay for itself: an improvement smaller than the
    travel discount does not buy a lost feeding tick.
20. The reachable-payoff potential is a max, not a sum - it peaks at the
    best cell, not at the centre of mass.
21. ``greedy_hide`` rests where the predation pressure or the energy
    reserve says so, and is otherwise identical to ``greedy``.
22. Co-location preserves total biomass and cell count, places every
    predator on its prey, and runs bottom up through the food web.
23. The tick-0 overlap is measured and reported, because a FAIL cannot
    be read without it.
"""
import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.diagnostics import viability  # noqa: E402
from lib.environments.ecosystem_env.constants import EAT_START  # noqa: E402

PROJECT = os.path.join(_ROOT, "mareld2.yaml")


def _result(series, reference=None, extinct=None, occupancy=None, seed=0):
    """A synthetic SeedResult, so the criterion is tested without a tick."""
    series = {fid: np.asarray(values, dtype=np.float64)
              for fid, values in series.items()}
    reference = reference or {fid: float(values[0])
                             for fid, values in series.items()}
    return viability.SeedResult(
        seed=seed,
        reference=reference,
        series=series,
        extinct_tick=extinct or {fid: None for fid in series},
        occupancy=occupancy or {fid: 1 for fid in series},
        habitable_cells=1,
    )


def _verdict(values, reference=None, extinct=None, **criterion_kwargs):
    criterion = viability.ViabilityCriterion(**criterion_kwargs)
    result = _result({"fg": values},
                     reference=None if reference is None else {"fg": reference},
                     extinct=None if extinct is None else {"fg": extinct})
    verdicts = viability.evaluate_seed(result, criterion)
    assert len(verdicts) == 1
    return verdicts[0]


# ---------------------------------------------------------------- 1. and 2.


def test_horizon_and_window_follow_from_years():
    criterion = viability.ViabilityCriterion(years=5.0, window_frac=0.10)
    assert criterion.ticks == 5 * viability.TICKS_PER_YEAR == 7300
    assert criterion.window == 730


def test_window_is_at_least_one_tick():
    criterion = viability.ViabilityCriterion(years=1.0 / 1460, window_frac=0.1)
    assert criterion.ticks == 1
    assert criterion.window == 1


@pytest.mark.parametrize("kwargs", [
    {"years": 0.0},
    {"seeds": 0},
    {"floor": 0.0},
    {"floor": 1.5},
    {"ceiling": 0.5},
    {"max_drift": 1.0},
    {"window_frac": 0.0},
    {"window_frac": 0.75},
])
def test_nonsensical_criteria_are_rejected(kwargs):
    with pytest.raises(ValueError):
        viability.ViabilityCriterion(**kwargs)


# ------------------------------------------------------------------ 3. - 8.


def test_flat_trajectory_at_spawn_passes():
    verdict = _verdict(np.full(100, 50.0), years=100.0 / 1460)
    assert verdict.ok
    assert verdict.reason == "ok"
    assert verdict.ratio == pytest.approx(1.0)
    assert verdict.drift == pytest.approx(1.0)


def test_settled_equilibrium_below_floor_fails():
    values = np.full(100, 5.0)
    verdict = _verdict(values, reference=100.0, years=100.0 / 1460)
    assert verdict.survived and verdict.stationary_ok
    assert not verdict.floor_ok
    assert verdict.reason == "below floor"
    # The measured equilibrium is still reported: it is the number the
    # spawn should be recalibrated against.
    assert verdict.equilibrium == pytest.approx(5.0)
    assert verdict.ratio == pytest.approx(0.05)


def test_runaway_growth_fails_on_the_ceiling():
    verdict = _verdict(np.full(100, 500.0), reference=10.0,
                       years=100.0 / 1460)
    assert verdict.survived and verdict.floor_ok
    assert not verdict.ceiling_ok
    assert verdict.reason == "above ceiling"


def test_still_doubling_fails_on_stationarity():
    # Geometric growth: the final window is still well above the one
    # before it, while floor and ceiling are both satisfied.
    values = 100.0 * np.exp(np.linspace(0.0, 3.0, 100))
    verdict = _verdict(values, reference=100.0, years=100.0 / 1460,
                       ceiling=100.0, max_drift=1.2)
    assert verdict.floor_ok and verdict.ceiling_ok and verdict.survived
    assert not verdict.stationary_ok
    assert verdict.reason == "still drifting"
    assert verdict.drift > 1.2


def test_extinction_inside_the_horizon_fails_and_reports_the_tick():
    values = np.concatenate([np.full(10, 50.0), np.zeros(90)])
    verdict = _verdict(values, reference=50.0, extinct=11,
                       years=100.0 / 1460)
    assert not verdict.survived
    assert verdict.extinct_tick == 11
    assert verdict.reason == "extinct@11"
    assert verdict.min_ratio == pytest.approx(0.0)


def test_groups_absent_at_spawn_are_outside_the_criterion():
    criterion = viability.ViabilityCriterion(years=100.0 / 1460)
    result = _result({"present": np.full(100, 5.0),
                      "absent": np.zeros(100)},
                     reference={"present": 5.0, "absent": 0.0})
    verdicts = viability.evaluate_seed(result, criterion)
    assert [v.fg_id for v in verdicts] == ["present"]


# --------------------------------------------------------------- 9. and 10.


def _arm(per_seed_values, **criterion_kwargs):
    criterion = viability.ViabilityCriterion(**criterion_kwargs)
    arm = viability.ArmVerdict(behaviour=viability.NEUTRAL,
                               criterion=criterion)
    for seed, (values, reference) in per_seed_values.items():
        result = _result({"fg": np.asarray(values, dtype=np.float64)},
                         reference={"fg": reference}, seed=seed)
        arm.per_seed[seed] = viability.evaluate_seed(result, criterion)
    return arm


def test_the_worst_seed_decides_the_arm():
    arm = _arm({1: (np.full(100, 100.0), 100.0),
                2: (np.full(100, 2.0), 100.0)},
               years=100.0 / 1460)
    assert not arm.ok
    assert arm.worst("fg").ratio == pytest.approx(0.02)
    assert arm.worst("fg").reason == "below floor"


def test_an_arm_is_viable_only_when_every_seed_is():
    arm = _arm({1: (np.full(100, 100.0), 100.0),
                2: (np.full(100, 90.0), 100.0)},
               years=100.0 / 1460)
    assert arm.ok
    assert arm.fg_ids == ["fg"]


def test_summary_is_strict_json_and_carries_the_verdict():
    # A group that dies leaves an infinite drift; strict JSON has no
    # Infinity, so the summary must degrade it to null.
    values = np.concatenate([np.full(50, 10.0), np.zeros(50)])
    criterion = viability.ViabilityCriterion(years=100.0 / 1460)
    arm = viability.ArmVerdict(behaviour=viability.EAT, criterion=criterion)
    result = _result({"fg": values}, reference={"fg": 10.0}, extinct={"fg": 51})
    arm.per_seed[1] = viability.evaluate_seed(result, criterion)

    summary = viability.summary_dict([arm])
    assert summary["viable"] is False
    assert summary["arms"][0]["behaviour"] == viability.EAT
    assert summary["arms"][0]["groups"]["fg"]["extinct_tick"] == 51
    text = json.dumps(summary, allow_nan=False)
    assert "Infinity" not in text


# -------------------------------------------------------------- 11. - 14.


def _tiny_env(grid=(4, 4), seed=7):
    from inference import build_env

    return build_env(PROJECT, grid, seed=seed, verbose=False,
                     apply_natural_mortality=False, migration=False)


def test_rollout_books_spawn_at_t0_and_one_sample_per_tick():
    env = _tiny_env()
    spawn = {fid: float(fg.biomass.sum()) for fid, fg in env.fgs.items()}
    provider = viability.install_behaviour(env, viability.NEUTRAL)
    result = viability.run_rollout(env, 3, provider)

    assert env.tick_count == 3
    for fid, total in spawn.items():
        assert result.reference[fid] == pytest.approx(total)
        assert len(result.series[fid]) == 3
    assert result.habitable_cells == env.H * env.W


def test_neutral_arm_installs_no_network_and_acts_uniformly():
    env = _tiny_env()
    provider = viability.install_behaviour(env, viability.NEUTRAL)
    assert provider is None          # goes through the policy controller
    assert env.policies == {}
    assert not env._batched_ready

    observation = env.get_observation()
    actions = env.policy_controller.forward(observation)
    probs = np.concatenate([
        actions.move,
        actions.rest[:, None],
        actions.eat,
    ], axis=1)
    # Every cell is either uniform over its legal actions, or - below the
    # split threshold - collapsed onto a single one.
    for cell in probs[0].reshape(probs.shape[1], -1).T:
        legal = cell[cell > 1e-6]
        assert legal.size >= 1
        if legal.size > 1:
            assert np.allclose(legal, legal[0], rtol=1e-4)
        assert cell.sum() == pytest.approx(1.0, rel=1e-4)


def test_eat_arm_never_moves_and_spends_its_mass_on_legal_prey():
    env = _tiny_env()
    provider = viability.install_behaviour(env, viability.EAT)
    actions = provider(env)

    assert np.all(actions.move == 0.0)
    eat_total = actions.eat.sum(axis=1)
    assert np.all((np.abs(eat_total - 1.0) < 1e-5) | (eat_total == 0.0))
    # Resting happens exactly where there is nothing legal to eat.
    assert np.allclose(actions.rest, (eat_total == 0.0).astype(np.float32))

    from lib.environments.ecosystem_env import decisions

    mask = decisions.build_action_mask(env, EAT_START + env.N_all)
    assert np.all(actions.eat[mask[:, EAT_START:EAT_START + env.N_all] == 0] == 0)


def test_random_arm_is_reproducible_for_a_given_seed():
    first = _tiny_env()
    viability.install_behaviour(first, viability.RANDOM, seed=11)
    second = _tiny_env()
    viability.install_behaviour(second, viability.RANDOM, seed=11)
    for fid in first.dm_ids:
        assert np.allclose(
            first.policies[fid].net[0].weight.detach().numpy(),
            second.policies[fid].net[0].weight.detach().numpy())


def test_unknown_and_unconfigured_arms_are_refused():
    env = _tiny_env()
    with pytest.raises(ValueError):
        viability.install_behaviour(env, "hope")
    with pytest.raises(ValueError):
        viability.install_behaviour(env, viability.POLICY)


# -------------------------------------------------------------- 15. and 16.


def _cli():
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
    import viability as cli  # noqa: E402  (tools/viability.py)
    return cli


@pytest.mark.parametrize("text,expected", [
    ("20*20", (20, 20)),
    ("16x24", (16, 24)),
    ("32", (32, 32)),
])
def test_cli_accepts_the_project_grid_spelling(text, expected):
    assert _cli().parse_grid(text) == expected


def test_cli_rejects_a_too_small_grid():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _cli().parse_grid("2*2")


def test_cli_rejects_unknown_behaviours():
    import argparse
    cli = _cli()
    assert cli.parse_behaviours("eat, neutral") == ["eat", "neutral"]
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_behaviours("eat,hope")


def test_cli_ticks_override_years():
    cli = _cli()
    args = cli.build_parser().parse_args(["--ticks", "2920", "--years", "5"])
    assert args.ticks == 2920
    criterion = viability.ViabilityCriterion(
        years=args.ticks / viability.TICKS_PER_YEAR)
    assert criterion.ticks == 2920


def test_cli_exposes_both_factors_of_the_rig():
    cli = _cli()
    default = cli.build_parser().parse_args([])
    assert default.spawn == viability.SPAWN_COLOCATED
    assert default.behaviour == [viability.GREEDY, viability.GREEDY_HIDE,
                                 viability.EAT]
    configured = cli.build_parser().parse_args(["--spawn", "configured"])
    assert configured.spawn == viability.SPAWN_CONFIGURED
    alias = cli.build_parser().parse_args(["--spawn", "configured",
                                           "--colocated-spawn"])
    assert alias.spawn == viability.SPAWN_COLOCATED
    assert cli.parse_behaviours("greedy,greedy_hide") == ["greedy",
                                                          "greedy_hide"]


# ------------------------------------------------------------------- 17.


def _flat_arm(behaviour, level, reference=100.0):
    criterion = viability.ViabilityCriterion(years=100.0 / 1460)
    arm = viability.ArmVerdict(behaviour=behaviour, criterion=criterion)
    result = _result({"fg": np.full(100, level)}, reference={"fg": reference})
    arm.per_seed[1] = viability.evaluate_seed(result, criterion)
    return arm


def test_only_a_normative_arm_may_decide():
    # A uniform or untrained policy dying says nothing about the world,
    # so a run of those arms alone must not produce NOT VIABLE.
    diagnostic = [_flat_arm(viability.NEUTRAL, 1.0),
                  _flat_arm(viability.RANDOM, 1.0)]
    assert viability.overall_verdict(diagnostic) is None
    assert viability.summary_dict(diagnostic)["viable"] is None
    assert viability.summary_dict(diagnostic)["normative_corner"] is False


def test_greedy_decides_over_the_other_arms():
    arms = [_flat_arm(viability.EAT, 100.0),
            _flat_arm(viability.GREEDY, 1.0),
            _flat_arm(viability.NEUTRAL, 100.0)]
    assert viability.deciding_arm(arms).behaviour == viability.GREEDY
    assert viability.overall_verdict(arms) is False

    summary = viability.summary_dict(arms, spawn=viability.SPAWN_COLOCATED)
    assert summary["verdict_arm"] == viability.GREEDY
    assert summary["normative_corner"] is True
    assert summary["spawn"] == viability.SPAWN_COLOCATED
    # The configured-spawn corner is diagnostic, not normative.
    assert viability.summary_dict(
        arms, spawn=viability.SPAWN_CONFIGURED)["normative_corner"] is False


# -------------------------------------------------------------- 18. - 20.


def _probabilities(actions):
    return np.concatenate([actions.move, actions.rest[:, None],
                           actions.eat], axis=1)


def test_greedy_arm_is_a_distribution_over_legal_actions_only():
    from lib.environments.ecosystem_env import decisions
    from lib.environments.ecosystem_env.constants import MOVE_SLICE

    env = _tiny_env(grid=(6, 6))
    provider = viability.install_behaviour(env, viability.GREEDY)
    actions = provider(env)
    probs = _probabilities(actions)

    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    mask = decisions.build_action_mask(env, EAT_START + env.N_all)
    assert np.all(actions.move[mask[:, MOVE_SLICE] == 0] == 0.0)
    assert np.all(actions.eat[mask[:, EAT_START:EAT_START + env.N_all] == 0]
                  == 0.0)
    # Rest is never chosen voluntarily: only where the cell neither
    # climbs nor has anything legal to eat.
    moving = actions.move.sum(axis=1) > 0
    feeding = actions.eat.sum(axis=1) > 0
    assert np.all(actions.rest[moving | feeding] == 0.0)


def test_greedy_eats_at_a_local_optimum_and_walks_towards_a_better_cell():
    env = _tiny_env(grid=(6, 6))
    viability.install_behaviour(env, viability.GREEDY)
    payoff = viability._intake_payoff(env)
    potential = viability._reachable_payoff(env, payoff)
    actions = viability._greedy_actions(env)

    moving = actions.move.sum(axis=1) > 0
    # Exactly the cells whose best neighbour beats staying, after the
    # travel discount, are the ones that move.
    neighbours = viability._neighbour_potential(env, potential)
    neighbours = np.where(env.move_mask > 0, neighbours, -np.inf)
    expected = (np.float32(viability.TRAVEL_DISCOUNT)
                * neighbours.max(axis=1) > payoff)
    assert np.array_equal(moving, expected)


def _stub_env(width):
    class _Env:
        H, W = 1, width
        dtype = np.float32
    return _Env()


def test_travel_must_pay_for_the_feeding_tick_it_costs():
    # Three cells in a row, the far one slightly better. Walking there
    # costs a tick of eating, so a 15 % improvement must not be enough
    # while a 100 % improvement must be.
    env = _stub_env(3)
    discount = viability.TRAVEL_DISCOUNT

    def moves(far):
        payoff = np.array([[[1.0, 0.0, far]]], dtype=np.float32)
        potential = viability._reachable_payoff(env, payoff)
        return bool(discount * potential[0, 0, 1] > payoff[0, 0, 0])

    assert not moves(1.15)
    assert moves(2.0)


def test_the_potential_peaks_at_the_best_cell_not_the_centre_of_mass():
    # A summed (diffusive) potential would peak in the middle of the
    # mediocre block; the max-plus form must peak on the single best
    # cell, or every group walks to the centre of the grid and starves.
    env = _stub_env(9)
    payoff = np.array([[[10.0, 0, 0, 0, 3.0, 3.0, 3.0, 3.0, 3.0]]],
                      dtype=np.float32)
    potential = viability._reachable_payoff(env, payoff)
    assert int(np.argmax(potential[0, 0])) == 0
    # ... and the value decays by exactly the discount per cell walked.
    assert potential[0, 0, 1] == pytest.approx(
        10.0 * viability.TRAVEL_DISCOUNT, rel=1e-5)


# ------------------------------------------------------------------- 21.


def test_greedy_hide_rests_when_the_reserve_is_full_and_greedy_does_not():
    env = _tiny_env(grid=(6, 6))
    viability.install_behaviour(env, viability.GREEDY_HIDE)
    # A full reserve makes intake worthless, so the hiding arm rests.
    for i, fid in enumerate(env.dm_ids):
        env.fgs[fid].energy_reserve = (
            env.fgs[fid].biomass * env.dm_max_energy_reserve[i]).astype(
                env.dtype, copy=False)

    plain = viability._greedy_actions(env, hide=False)
    hiding = viability._greedy_actions(env, hide=True)

    biomass = np.stack([env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    alive = biomass > 0
    assert np.all(hiding.rest[alive] == 1.0)
    assert np.all(hiding.eat.sum(axis=1)[alive] == 0.0)
    assert np.all(_probabilities(hiding).sum(axis=1) == pytest.approx(1.0,
                                                                     abs=1e-5))
    # The plain arm keeps feeding where it can - the difference between
    # the arms is what hiding is worth, so it must not be zero here.
    assert plain.eat.sum() > 0.0


def test_greedy_hide_rests_under_predation_pressure():
    env = _tiny_env(grid=(6, 6))
    viability.install_behaviour(env, viability.GREEDY_HIDE)
    pressure = viability._predator_biomass_field(env)
    biomass = np.stack([env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    threatened = (biomass > 0) & (
        pressure >= viability.HIDE_PREDATOR_RATIO * biomass)
    if not np.any(threatened):
        pytest.skip("no DM is outweighed by its predators in this layout")
    actions = viability._greedy_actions(env, hide=True)
    assert np.all(actions.rest[threatened] == 1.0)


# -------------------------------------------------------------- 22. - 23.


def test_trophic_order_puts_prey_before_its_predators():
    menus = {"porpoises": ["pelagic_fish"],
             "pelagic_fish": ["zooplankton"],
             "zooplankton": ["phytoplankton"]}
    order = viability.trophic_order(
        menus, ["porpoises", "pelagic_fish", "zooplankton"])
    assert order.index("zooplankton") < order.index("pelagic_fish")
    assert order.index("pelagic_fish") < order.index("porpoises")


def test_trophic_order_survives_a_cycle():
    menus = {"a": ["b"], "b": ["a"]}
    assert sorted(viability.trophic_order(menus, ["a", "b"])) == ["a", "b"]


def test_colocation_moves_predators_onto_prey_and_conserves_biomass():
    env = _tiny_env(grid=(12, 12), seed=3)
    before_total = {fid: float(fg.biomass.sum()) for fid, fg in env.fgs.items()}
    before_occupancy = {fid: int(np.count_nonzero(fg.biomass > 0))
                        for fid, fg in env.fgs.items()}
    before = {row["predator"] + ">" + row["prey"]: row["biomass_with_prey"]
              for row in viability.spawn_overlap(env)}

    moved = viability.colocate_spawn(env)
    after = {row["predator"] + ">" + row["prey"]: row["biomass_with_prey"]
             for row in viability.spawn_overlap(env)}

    assert moved, "at least the decision makers with a menu must move"
    for fid, total in before_total.items():
        assert float(env.fgs[fid].biomass.sum()) == pytest.approx(
            total, rel=1e-4)
        # Cell count is held fixed so per-cell density and patchiness
        # are unchanged - only the geometry moves.
        assert int(np.count_nonzero(env.fgs[fid].biomass > 0)) <= \
            before_occupancy[fid]
    for pair, overlap in after.items():
        assert overlap >= before[pair] - 1e-9
    # The primary pair the critique was about must end fully overlapped.
    assert after["pelagic_fish>zooplankton"] == pytest.approx(1.0)


def test_overlap_reports_both_numbers_a_verdict_must_be_read_with():
    env = _tiny_env(grid=(12, 12), seed=3)
    rows = viability.spawn_overlap(env)
    assert rows
    for row in rows:
        assert set(row) == {"predator", "prey", "biomass_with_prey",
                            "prey_seen"}
        assert 0.0 <= row["biomass_with_prey"] <= 1.0
        assert row["prey_seen"] >= 0.0
    text = viability.format_overlap(rows)
    assert "predator" in text and rows[0]["predator"] in text


# ------------------------------------------------------------------- 24.
#
# The diet. Both hand-coded arms used to spread the eat mass EVENLY over
# the legal prey present, which is not an upper bound on intake and is
# therefore not something a normative arm may do: for porpoises an even
# herring/gadoid split lands within three parts per thousand of
# break-even while a herring diet clears it by 19 %. Section 92.


def _legal_eat_mask(env):
    from lib.environments.ecosystem_env import decisions
    mask = decisions.build_action_mask(env, EAT_START + env.N_all)
    return mask, mask[:, EAT_START:EAT_START + env.N_all] > 0


def test_waterfill_takes_the_best_prey_first_and_spills_into_the_next():
    # Two prey, the first worth twice as much but saturating at 30 % of
    # the action mass. An energy maximiser fills it and spends the rest
    # on the runner-up.
    score = np.array([[[[2.0]], [[1.0]]]], dtype=np.float32)
    caps = np.array([[[[0.3]], [[1.0]]]], dtype=np.float32)
    shares = viability._waterfill(score, caps)
    assert shares[0, 0, 0, 0] == pytest.approx(0.3)
    assert shares[0, 1, 0, 0] == pytest.approx(0.7)

    # With no binding cap the whole mass goes to the better prey - an
    # even split would be a strictly worse diet.
    shares = viability._waterfill(score, np.ones_like(caps))
    assert shares[0, 0, 0, 0] == pytest.approx(1.0)
    assert shares[0, 1, 0, 0] == pytest.approx(0.0)


def test_waterfill_always_spends_the_whole_action_mass():
    # Every prey saturated well below the unit mass: the leftover has to
    # go somewhere, and it goes to the best of them, so the arms keep
    # paying the same feeding cost as under the even split.
    score = np.array([[[[2.0]], [[1.0]]]], dtype=np.float32)
    caps = np.array([[[[0.1]], [[0.2]]]], dtype=np.float32)
    shares = viability._waterfill(score, caps)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[0, 0, 0, 0] > shares[0, 1, 0, 0]


def test_eat_shares_are_a_distribution_over_legal_prey_only():
    env = _tiny_env(grid=(6, 6))
    viability.install_behaviour(env, viability.EAT)
    mask, legal = _legal_eat_mask(env)
    shares, feeding = viability._eat_shares(env, mask)

    assert np.array_equal(feeding, legal.any(axis=1, keepdims=True))
    assert np.all(shares[~legal] == 0.0)
    totals = shares.sum(axis=1)
    assert np.allclose(totals[feeding[:, 0]], 1.0, atol=1e-5)
    assert np.all(totals[~feeding[:, 0]] == 0.0)


def test_the_diet_is_never_worse_than_the_even_split_it_replaced():
    """The arms may only get friendlier, and strictly so somewhere.

    Compared on the energy the engine would actually deliver, i.e. with
    the share above each prey's harvest cap discarded the way
    ``apply_predation`` discards it.
    """
    env = _tiny_env(grid=(6, 6))
    viability.install_behaviour(env, viability.EAT)
    mask, legal = _legal_eat_mask(env)
    shares, feeding = viability._eat_shares(env, mask)

    per_prey = viability._per_prey_intake(env)
    caps = viability._saturating_share(env, per_prey)
    gain = env.energy_gain_mat[:, :, None, None] * per_prey

    even = legal.astype(np.float32)
    even /= np.maximum(even.sum(axis=1, keepdims=True), 1e-12)

    chosen = (np.minimum(shares, caps) * gain).sum(axis=1)
    uniform = (np.minimum(even, caps) * gain).sum(axis=1)
    assert np.all(chosen >= uniform - 1e-5)
    assert chosen.sum() > uniform.sum()


def test_the_top_predator_eats_by_marginal_yield_never_fifty_fifty():
    """The substantive case: porpoises must not take a 50/50 diet.

    An even herring/gadoid split is 5685 MJ/t assimilated against a
    5670 MJ/t requirement; a pure herring diet is 6750. An arm that
    hands the top predator junk food cannot pronounce on the top
    predator's budget.

    The criterion is marginal yield ``energy_gain * a_eff``, not energy
    content alone: a cell dense in gadoid can out-yield the same cell's
    thin herring, and eating the gadoid there is the right answer. What
    must never happen is the mass being split without regard to either.
    """
    env = _tiny_env(grid=(6, 6))
    if "porpoises" not in env.dm_ids:
        pytest.skip("no porpoises in the project configuration")
    i = env.dm_ids.index("porpoises")
    viability.install_behaviour(env, viability.EAT)
    mask, legal = _legal_eat_mask(env)
    shares, _ = viability._eat_shares(env, mask)

    per_prey = viability._per_prey_intake(env)
    yields = (env.energy_gain_mat[:, :, None, None] * per_prey)[i]
    caps = viability._saturating_share(env, per_prey)[i]

    menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
    if len(menu) < 2:
        pytest.skip("porpoises have a single-prey menu here")

    # Water-filling optimality: a prey may only be served once every
    # strictly better one is saturated.
    for j in menu:
        for k in menu:
            if j == k:
                continue
            better = legal[i, j] & legal[i, k] & (yields[j] > yields[k])
            served = better & (shares[i, k] > 1e-6)
            assert np.all(shares[i, j][served] >= caps[j][served] - 1e-5)

    both = legal[i, menu[0]] & legal[i, menu[1]]
    if not np.any(both):
        pytest.skip("no cell holds two prey at once")
    split = np.isclose(shares[i, menu[0]][both], 0.5, atol=1e-3)
    assert not np.all(split), "the even split is back"


def test_cli_exit_code_carries_the_verdict(tmp_path, capsys):
    cli = _cli()
    summary_path = tmp_path / "viability.json"
    code = cli.main([
        "--project", PROJECT, "--grid", "4*4", "--ticks", "4",
        "--seeds", "1", "--behaviour", "eat",
        "--json", str(summary_path), "--quiet",
    ])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert code == (0 if summary["viable"] else 1)
    assert summary["criterion"]["ticks"] == 4
    assert summary["arms"][0]["behaviour"] == "eat"
    assert "overall:" in capsys.readouterr().out
