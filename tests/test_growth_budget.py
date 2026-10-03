"""growth_rate derived from r_max, M1 and the predation matrix.

Contract (lib/world/growth_budget.py): a prey's growth_rate pays for the
predation of every checked, active, decision-making predator, so
checking a predator raises the prey's growth by exactly that predator's
share and unchecking lowers it by the same amount, leaving the net
low-density growth r_max invariant.
"""
import math
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.config.config_loader import (  # noqa: E402
    load_config,
    load_project_config,
    setup_full_mareld_mvp,
)
from lib.world.growth_budget import (  # noqa: E402
    TICKS_PER_YEAR,
    active_predation,
    annual_rate_from_tick_loss,
    derived_growth_rate,
    growth_budget,
    max_gate,
)

LIBRARY = os.path.join(_ROOT, "fgconfig", "fg_library.yaml")
PROJECT = os.path.join(_ROOT, "mareld2.yaml")


def _toy():
    species = {
        "prey": {"is_decision_maker": True, "r_max": 0.6,
                 "natural_mortality": 1.0 - math.exp(-0.4 / TICKS_PER_YEAR),
                 "maintenance_level": 0.5},
        "fox": {"is_decision_maker": True},
        "owl": {"is_decision_maker": True},
        "kelp": {"is_decision_maker": False, "r_max": 2.0},
        "crab": {"is_decision_maker": False},
    }
    inter = {
        "fox_preys_on_prey": {"preys_on": True, "predation_mortality": 0.3},
        "owl_preys_on_prey": {"preys_on": False, "predation_mortality": 0.1},
        "crab_preys_on_prey": {"preys_on": True, "predation_mortality": 5.0},
        "prey_preys_on_kelp": {"preys_on": True, "predation_mortality": 1.0},
    }
    return species, inter


def test_decision_maker_budget_formula():
    species, inter = _toy()
    b = growth_budget("prey", species, inter)
    assert b.gate == pytest.approx(0.5)
    assert b.m1_annual == pytest.approx(0.4)
    assert b.predation == {"fox": 0.3}
    assert b.growth_rate == pytest.approx((0.6 + 0.4 + 0.3) / (0.5 * TICKS_PER_YEAR))


def test_net_low_density_growth_is_r_max_whatever_the_matrix():
    species, inter = _toy()
    for owl_on in (False, True):
        inter["owl_preys_on_prey"]["preys_on"] = owl_on
        b = growth_budget("prey", species, inter)
        net = b.growth_rate * b.gate * TICKS_PER_YEAR - b.m1_annual - b.predation_total
        assert net == pytest.approx(0.6)


def test_check_and_uncheck_move_growth_by_the_predator_share():
    species, inter = _toy()
    g_off = derived_growth_rate("prey", species, inter)
    inter["owl_preys_on_prey"]["preys_on"] = True
    g_on = derived_growth_rate("prey", species, inter)
    assert g_on - g_off == pytest.approx(0.1 / (0.5 * TICKS_PER_YEAR))
    inter["owl_preys_on_prey"]["preys_on"] = False
    assert derived_growth_rate("prey", species, inter) == pytest.approx(g_off)


def test_only_active_decision_making_predators_count():
    species, inter = _toy()
    # crab is a non decision maker: it cannot hunt in the tick.
    assert "crab" not in active_predation("prey", species, inter)
    # A muted / absent predator does not count either.
    assert active_predation("prey", species, inter, active_predators=["prey"]) == {}


def test_non_decision_maker_has_no_gate_and_no_m1():
    species, inter = _toy()
    b = growth_budget("kelp", species, inter)
    assert b.gate == 1.0 and b.m1_annual == 0.0
    assert b.growth_rate == pytest.approx((2.0 + 1.0) / TICKS_PER_YEAR)


def test_legacy_entry_without_r_max_keeps_its_growth_rate():
    species, inter = _toy()
    assert derived_growth_rate("fox", species, inter) is None


def test_closed_growth_window_is_reported_not_divided_by():
    species, inter = _toy()
    species["prey"]["maintenance_level"] = 1.0
    b = growth_budget("prey", species, inter)
    assert b.error and b.growth_rate is None
    assert derived_growth_rate("prey", species, inter) is None


def test_annual_rate_inverts_the_loss_rule():
    m = 1.0 - math.exp(-0.58 / TICKS_PER_YEAR)
    assert annual_rate_from_tick_loss(m) == pytest.approx(0.58)
    assert annual_rate_from_tick_loss(0.0) == 0.0


def test_max_gate_is_a_full_reserve_minus_maintenance():
    assert max_gate({"maintenance_level": 0.5}) == pytest.approx(0.5)
    assert max_gate({"maintenance_level": 0.3}) == pytest.approx(0.7)


def test_mortality_off_drops_m1_from_the_budget():
    species, inter = _toy()
    on = growth_budget("prey", species, inter)
    off = growth_budget("prey", species, inter, include_m1=False)
    assert off.m1_annual == 0.0
    assert on.growth_rate - off.growth_rate == pytest.approx(
        0.4 / (0.5 * TICKS_PER_YEAR))
    # NDMs carry no M1, so the flag does not touch them.
    assert growth_budget("kelp", species, inter, include_m1=False).growth_rate \
        == pytest.approx(growth_budget("kelp", species, inter).growth_rate)


# ---------- the live library and project ----------

def _active_in_project():
    project = load_config(PROJECT)
    ids = []
    for key in ("decision_makers", "non_decision_makers"):
        for fg in project.get(key, []) or []:
            if isinstance(fg, dict) and not fg.get("muted"):
                ids.append(fg["group_id"])
    return ids


# benthic_community keeps its hand-set growth_rate (reverted, section 136).
HAND_SET = {"benthic_community"}


def _expected(fid, spec, S, I, active, include_m1=True):
    g = derived_growth_rate(fid, S, I, active, include_m1=include_m1)
    return float(spec["growth_rate"]) if g is None else g


def test_every_library_fg_has_a_literature_budget():
    lib = load_config(LIBRARY)
    for fid, spec in lib["species_definitions"].items():
        if fid in HAND_SET:
            assert "r_max" not in spec, fid
        else:
            assert float(spec.get("r_max", 0.0)) > 0.0, fid
        assert "satiation_scale" not in spec, fid  # removed, section 130


def test_stored_growth_rate_is_the_derived_one_for_the_project():
    """The number in fg_library.yaml must not drift from its budget."""
    lib = load_config(LIBRARY)
    active = _active_in_project()
    S, I = lib["species_definitions"], lib["interaction_definitions"]
    for fid, spec in S.items():
        g = _expected(fid, spec, S, I, active)
        assert float(spec["growth_rate"]) == pytest.approx(g, rel=1e-5), fid


def test_loader_derives_growth_from_the_active_predators():
    lib = load_config(LIBRARY)
    S, I = lib["species_definitions"], lib["interaction_definitions"]
    fgs, *_ = load_project_config(PROJECT, library_path=LIBRARY,
                                  grid_size=(6, 6), seed=0)
    active = _active_in_project()
    for fid, fg in fgs.items():
        assert fg.growth_rate == pytest.approx(
            _expected(fid, S[fid], S, I, active)), fid
    # The library-only path counts every library FG as present, so a prey
    # of the muted seals / seabirds pays for them there and not in the
    # project.
    full = setup_full_mareld_mvp(grid_size=(6, 6), seed=0, spawn_seed=0)
    if "seals" not in active and "pelagic_fish" in fgs:
        assert full["pelagic_fish"].growth_rate > fgs["pelagic_fish"].growth_rate


def test_loader_leaves_m1_out_when_mortality_is_off():
    lib = load_config(LIBRARY)
    S, I = lib["species_definitions"], lib["interaction_definitions"]
    active = _active_in_project()
    off, *_ = load_project_config(PROJECT, library_path=LIBRARY,
                                  grid_size=(6, 6), seed=0,
                                  apply_natural_mortality=False)
    for fid, fg in off.items():
        assert fg.growth_rate == pytest.approx(
            _expected(fid, S[fid], S, I, active, include_m1=False)), fid
    full_off = setup_full_mareld_mvp(grid_size=(6, 6), seed=0, spawn_seed=0,
                                     apply_natural_mortality=False)
    full_on = setup_full_mareld_mvp(grid_size=(6, 6), seed=0, spawn_seed=0)
    for fid in ("zooplankton", "pelagic_fish", "gadoids"):
        assert full_off[fid].growth_rate < full_on[fid].growth_rate
    assert full_off["phytoplankton"].growth_rate == pytest.approx(
        full_on["phytoplankton"].growth_rate)
