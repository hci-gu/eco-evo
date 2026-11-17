"""Section 276: the time a move takes (research_notes/Observabilitetsparen
design.md 2.2).

With ``relocation_speed`` U a one-cell (1 km) move takes the share
tau = min(1, 1 km / (U x tick)) of the tick; the moving share forages for
1 - tau of it in the cell it leaves, with the policy's eat preference read
before a sub-unit cell collapses to one action, pays tau x movement_cost +
(1 - tau) x feeding_cost, and takes its foraging share of the intake (net of
new tissue) and of the new tonnes with it. Absent (tau 1) is the old rule,
bit for bit.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.config import config_loader  # noqa: E402
from lib.environments.ecosystem_env import decisions, movement, predation  # noqa: E402
from lib.environments.ecosystem_env.state import ActionProbabilities  # noqa: E402
from lib.world import relocation  # noqa: E402
from test_edge_closure import DAYLIGHT, STAGES  # noqa: E402
from test_gpu_ecosystem import _assert_parity, device, make_env  # noqa: E402,F401

PROJECT = ROOT / "mareld2.yaml"
LIBRARY = ROOT / "fgconfig" / "fg_library.yaml"

# design.md 2.2, the member-weighted tau at 6 h.
DESIGN_TAU = {"planktivorous_fish": 0.20, "mackerel": 0.035, "predatory_fish": 0.073,
              "whiting": 0.23, "benthivorous_fish": 0.20, "porpoises": 0.071,
              "seals": 0.033}


# ---- tau from the speed --------------------------------------------------------

def test_tau_from_a_speed_and_from_members():
    assert relocation.tau(None, 6) == 1.0
    assert relocation.tau(0.65, 6) == pytest.approx(1000 / (0.65 * 21600))
    assert relocation.tau(0.65, 1) == pytest.approx(1000 / (0.65 * 3600))
    assert relocation.tau(0.1, 1) == 1.0          # a move longer than the tick
    members = [[0.55, 0.6], [0.45, 0.13]]
    expected = 0.55 * 1000 / (0.6 * 21600) + 0.45 * 1000 / (0.13 * 21600)
    assert relocation.tau(members, 6) == pytest.approx(expected)
    # The slow members' time dominates: not the tau of the mean speed.
    assert relocation.tau(members, 6) > relocation.tau(0.55 * 0.6 + 0.45 * 0.13, 6) * 1.5
    for bad, match in ((0.0, "> 0"), (-1, "> 0"), ([[0.5, 1.0]], "sum"),
                       ([[1.2, 1.0]], r"\(0, 1\]"), ([[1.0]], "share, speed"),
                       ("fast", "speed")):
        with pytest.raises(ValueError, match=match):
            relocation.tau(bad, 6)


def test_the_library_speeds_give_the_design_taus():
    species = yaml.safe_load(open(LIBRARY, encoding="utf-8-sig"))["species_definitions"]
    for fid, expected in DESIGN_TAU.items():
        value = species[fid].get(relocation.SPEED_KEY)
        assert value is not None, fid
        assert relocation.tau(value, 6) == pytest.approx(expected, abs=0.005), fid
    assert relocation.SPEED_KEY not in species["zooplankton"]   # drifts, never moves


def test_the_loader_derives_tau_at_the_runs_tick_length():
    taus = {}
    for hours in (6, 3):
        fgs = config_loader.load_project_config(
            str(PROJECT), library_path=str(LIBRARY), grid_size=(8, 8), seed=1,
            tick_hours=hours)[0]
        taus[hours] = {fid: fg.params.get(relocation.TAU_KEY) for fid, fg in fgs.items()}
    assert taus[6]["porpoises"] == pytest.approx(1000 / (0.65 * 21600))
    assert taus[3]["porpoises"] == pytest.approx(2 * taus[6]["porpoises"])
    assert taus[6]["zooplankton"] is None


# ---- the rule on the reference engine -----------------------------------------

def _actions(env, move=0.0, eat=0.0, rest=None, prey="b"):
    """Every DM cell: ``move`` north, ``eat`` on ``prey``, the rest resting."""
    H, W, D, N = env.H, env.W, env.N_dm, env.N_all
    m = np.zeros((D, 4, H, W), np.float32)
    m[:, 0] = move
    e = np.zeros((D, N, H, W), np.float32)
    e[:, env.global_fg_order.index(prey)] = eat
    r = np.full((D, H, W), 1.0 - move - eat if rest is None else rest, np.float32)
    return ActionProbabilities(m, r, e)


def _tau_env(tau=0.25, **kw):
    kw.setdefault("min_split", 0.0)
    kw.setdefault("extinction_factor", 0.0)
    env = make_env(regen={"a": {relocation.TAU_KEY: tau}}, **kw)
    env.build_static_caches()
    return env


def test_a_moving_cell_forages_one_minus_tau_with_its_preference():
    tau = 0.25
    gains = {}
    for name, (move, eat) in {"eat": (0.0, 1.0), "move": (1.0, 0.0)}.items():
        env = _tau_env(tau)
        # A sparse predator: no harvest cap binds, intake is linear in T.
        env.fgs["a"].biomass *= 1e-3
        env.fgs["a"].energy_reserve *= 1e-3
        actions = _actions(env, move=move, eat=eat, rest=0.0)
        # The policy wanted b before the cell collapsed to the move.
        preference = np.zeros_like(actions.eat)
        preference[:, env.global_fg_order.index("b")] = 1.0
        actions.forage = decisions.forage_with(env, actions.eat, actions.move, preference)
        predation.apply_predation(env, actions)
        gains[name] = env.fgs["a"].temp_energy_gains.copy()
    fed = gains["eat"] > 1e-6
    assert fed.sum() > 5
    np.testing.assert_allclose(gains["move"][fed] / gains["eat"][fed], 1.0 - tau, rtol=1e-5)
    # Without the policy's preference a cell with no eat share forages nothing.
    env = _tau_env(tau)
    predation.apply_predation(env, _actions(env, move=1.0, eat=0.0, rest=0.0))
    assert np.all(env.fgs["a"].temp_energy_gains == 0.0)


def test_the_moving_share_pays_tau_and_takes_its_share_of_the_intake():
    tau = 0.25
    env = _tau_env(tau)
    a = env.dm_ids.index("a")
    fg = env.fgs["a"]
    fg.temp_energy_gains = np.full_like(fg.biomass, 0.3) * (fg.biomass > 0)
    B, R = fg.biomass.copy(), fg.energy_reserve.copy()
    move, eat, rest = 0.5, 0.3, 0.2
    out = movement.apply_energy_costs(env, _actions(env, move=move, eat=eat, rest=rest))
    rm = float(env.dm_resting_metabolism[a])
    cm, ce, cr = (float(env.dm_cost_move[a]), float(env.dm_cost_eat[a]),
                  float(env.dm_cost_rest[a]))
    share = (1 - tau) * move / (eat + (1 - tau) * move)
    gains = fg.temp_energy_gains
    moving = np.maximum(R * move - B * move * rm * (tau * cm + (1 - tau) * ce), 0) + share * gains
    staying = (np.maximum(R * rest - B * rest * rm * cr, 0)
               + np.maximum(R * eat - B * eat * rm * ce, 0) + (1 - share) * gains)
    np.testing.assert_allclose(out.moving_reserve[a].sum(axis=0), moving, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(out.moving_reserve[a][0], moving, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(out.stationary_reserve[a], staying, rtol=1e-5, atol=1e-7)


def test_tau_one_is_bit_identical():
    """Forcing the section 276 path with tau 1 reproduces the old engine."""
    plain = _tau_env(1.0, migration=True, mortality=True)
    assert not plain._has_tau
    forced = _tau_env(1.0, migration=True, mortality=True)
    forced._has_tau = True
    rng = np.random.default_rng(3)
    for _ in range(4):
        p = rng.dirichlet(np.ones(5 + plain.N_all), size=(plain.N_dm, plain.H, plain.W))
        p = np.moveaxis(p, -1, 1).astype(np.float32)
        acts = ActionProbabilities(p[:, :4], p[:, 4], p[:, 5:])
        np.random.seed(7)
        plain.step(copy.deepcopy(acts))
        np.random.seed(7)
        forced.step(copy.deepcopy(acts))
        for fid in plain.fgs:
            np.testing.assert_array_equal(plain.fgs[fid].biomass, forced.fgs[fid].biomass)
            np.testing.assert_array_equal(plain.fgs[fid].energy_reserve,
                                          forced.fgs[fid].energy_reserve)


# ---- the tensor engine ------------------------------------------------------------

@pytest.mark.parametrize("min_split", [0.0, 1.0])
def test_tensor_engine_matches_with_tau(device, min_split):
    """min_split 1 puts some cells below one unit: they collapse to one action
    and forage with the preference read before the collapse. Handling 0.6:
    at the fixture's 0.2 the extra foraging takes prey c to the 0.999
    harvest cap in two cells, where B - intake cancels to float32 noise of
    1e-4 relative in both engines alike (not a formula difference)."""
    env = make_env(migration=True, mortality=True, min_split=min_split, extinction_factor=0.0,
                   handling=0.6,
                   regen={"a": {relocation.TAU_KEY: 0.2}, "b": {relocation.TAU_KEY: 0.6}})
    model = _assert_parity(env, device, ticks=8)
    assert model.tau_on


def test_tensor_engine_matches_with_tau_and_material_growth(device):
    env = make_env(min_split=0.0, extinction_factor=0.0, handling=0.6, material=(0.4, 6.0, 0.05),
                   regen={"a": {relocation.TAU_KEY: 0.2}})
    model = _assert_parity(env, device, ticks=6)
    assert model.tau_on and model.income_on


def test_tensor_engine_matches_with_tau_and_stages(device):
    env = make_env(migration=True, min_split=0.0, extinction_factor=0.0, handling=0.6,
                   stages={"a": dict(STAGES)}, daylight=dict(DAYLIGHT),
                   regen={"a": {relocation.TAU_KEY: 0.2}})
    model = _assert_parity(env, device, ticks=6)
    assert model.tau_on and model.stages_on
