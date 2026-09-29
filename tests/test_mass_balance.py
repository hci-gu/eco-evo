"""``--mass-balance``: the growth term pays for the biomass it adds.

Without the flag, ``_apply_decision_maker_population_change`` adds
``B * growth_rate * (s_X - u_X)`` and leaves ``energy_reserve``
untouched, so new biomass arrives carrying an implied ``energy_content``
that was never withdrawn from anywhere. Measured at 198-344 % of the mass
actually eaten against a 28.9 % ceiling from the library's own numbers
(mareld_resume.txt sections 114, 115).

With the flag, the growth term is still GATED by satiation exactly as
``Strategi.pdf`` specifies, but the wish is capped by the reserve
standing above the maintenance line and that reserve is debited for what
is built (section 116).

Contracts asserted here:

1. ON by default since section 120, because the library is calibrated
   for it; ``--no-mass-balance`` is the way back to the pre-116 tick and
   that tick is still bit-identical to what it always was.
2. On, the reserve is debited by exactly ``dB * energy_content``.
3. On, a reserve that cannot fund the wish caps the growth, and the
   reserve lands on the maintenance line rather than below it.
4. On, ``growth_rate`` above ``max_energy_reserve / energy_content`` is
   unreachable, because cap and wish are both linear in the surplus.
   This is the bound section 115.3 tabulates.
5. The starvation branch is untouched by the flag: a loss is not a
   purchase.
6. The GPU mirror agrees with the reference.
7. The shared CLI helper accepts both spellings and defaults to off, so
   the CPU and GPU parsers cannot drift apart.
"""
import argparse
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env import population_change  # noqa: E402
from lib.environments.ecosystem_env.population_change import (  # noqa: E402
    DEFAULT_MASS_BALANCE, LEGACY_MASS_BALANCE, add_mass_balance_argument,
)
from lib.world.functional_group import FunctionalGroup  # noqa: E402

H = W = 3
B0 = 4.0
ME = 1000.0          # max_energy_reserve, MJ per ton
EC = 4000.0          # energy_content, MJ per ton -> ME/EC = 0.25
U = 0.25             # maintenance_level
GRID_CONFIG = {'width': W, 'height': H, 'cell_size': 1000.0,
               'tick_duration': 6.0}


def _build_env(mass_balance=False, growth_rate=0.1, s_x=0.75,
               starve_rate=0.0, energy_content=EC):
    """One decision maker whose only active term is growth.

    No prey, no movement, no mortality, so the biomass can only change
    through the growth / starvation branch. ``s_x`` sets the energy fill
    ratio directly, which fixes the surplus at ``s_x - U``.
    """
    params = {
        'is_decision_maker': True,
        'max_energy_reserve': ME,
        'energy_content': energy_content,
        'resting_metabolism': 0.0,
        'maintenance_level': U,
        'movement_speed': 0.0,
        'growth_rate': growth_rate,
        'starve_rate': starve_rate,
        'natural_mortality': 0.0,
        'min_split_biomass': 0.0,
        'extinction_threshold_factor': 0.0,
        'menu': [],
    }
    fg = FunctionalGroup('grazer', params)
    fg.initialize_state((H, W), initial_biomass=np.full((H, W), B0))
    fg.energy_reserve = np.full((H, W), B0 * s_x * ME, dtype=np.float32)
    env = EcosystemEnvironment(GRID_CONFIG, {'grazer': fg},
                               apply_natural_mortality=False,
                               mass_balance=mass_balance)
    env.ordered_fg_ids = ['grazer']
    return env, fg


def _step(env, fg):
    r0 = float(fg.energy_reserve[0, 0])
    population_change._apply_decision_maker_population_change(
        env, 'grazer', fg)
    return (float(fg.biomass[0, 0]) - B0,
            r0 - float(fg.energy_reserve[0, 0]))


# ---------------------------------------------------------------------------
# 1. Default off, and off is the legacy term.
# ---------------------------------------------------------------------------

def test_flag_defaults_to_on_and_the_env_agrees():
    """Section 120. The env constructor and the CLI must not disagree:
    a library calibrated for one tick and a default that gives the other
    is the trap 119.2 describes."""
    assert DEFAULT_MASS_BALANCE is True
    env = EcosystemEnvironment(GRID_CONFIG, {}, apply_natural_mortality=False)
    assert env.mass_balance is True


def test_legacy_constant_is_separate_from_the_default():
    """``LEGACY_MASS_BALANCE`` records what runs PREDATING the flag were
    doing, so flipping the default cannot relabel an old history."""
    assert LEGACY_MASS_BALANCE is False
    from lib.runners.training_progress import comparable_config
    assert comparable_config({})["mass_balance"] is False


def test_off_adds_biomass_and_charges_nothing():
    env, fg = _build_env(mass_balance=False)
    gain, paid = _step(env, fg)
    assert gain == pytest.approx(B0 * 0.1 * (0.75 - U), rel=1e-6)
    assert paid == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2-3. On: the reserve pays, and the price is the cap.
# ---------------------------------------------------------------------------

def test_on_debits_the_reserve_by_gain_times_energy_content():
    env, fg = _build_env(mass_balance=True)
    gain, paid = _step(env, fg)
    # Reserve above maintenance is B0*(0.75-0.25)*ME = 2000 MJ, which
    # funds 0.5 t at EC = 4000; the wish is only 0.2 t, so the wish wins.
    assert gain == pytest.approx(B0 * 0.1 * (0.75 - U), rel=1e-6)
    assert paid == pytest.approx(gain * EC, rel=1e-6)


def test_on_caps_the_wish_at_what_the_reserve_can_fund():
    """A huge ``growth_rate`` cannot outrun the reserve above maintenance."""
    env, fg = _build_env(mass_balance=True, growth_rate=10.0)
    gain, paid = _step(env, fg)
    available = B0 * (0.75 - U) * ME
    assert gain == pytest.approx(available / EC, rel=1e-6)
    assert paid == pytest.approx(available, rel=1e-6)
    # Everything above the maintenance line was spent, nothing below it.
    assert float(fg.energy_reserve[0, 0]) == pytest.approx(
        B0 * U * ME, rel=1e-6)


def test_on_never_drives_the_reserve_negative():
    env, fg = _build_env(mass_balance=True, growth_rate=1e6)
    _step(env, fg)
    assert float(fg.energy_reserve.min()) >= 0.0


# ---------------------------------------------------------------------------
# 4. The ME/EC bound of section 115.3.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("growth_rate", [0.25, 0.5, 1.0])
def test_growth_rate_above_me_over_ec_is_unreachable(growth_rate):
    """``cap / wish = ME / (EC * growth_rate)``, so ME/EC = 0.25 binds."""
    env, fg = _build_env(mass_balance=True, growth_rate=growth_rate)
    gain, _ = _step(env, fg)
    bound = B0 * (ME / EC) * (0.75 - U)
    assert gain == pytest.approx(bound, rel=1e-6)


def test_zero_energy_content_leaves_the_wish_alone():
    """No price is definable, so the term degrades to the legacy one."""
    env, fg = _build_env(mass_balance=True, energy_content=0.0)
    gain, paid = _step(env, fg)
    assert gain == pytest.approx(B0 * 0.1 * (0.75 - U), rel=1e-6)
    assert paid == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Starvation is a loss, not a purchase.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mass_balance", [False, True])
def test_starvation_branch_is_identical_either_way(mass_balance):
    env, fg = _build_env(mass_balance=mass_balance, s_x=0.05,
                         starve_rate=0.5)
    gain, _ = _step(env, fg)
    # surplus = 0.05 - 0.25 = -0.2, loss = B0 * 0.5 * 0.2
    assert gain == pytest.approx(-B0 * 0.5 * 0.2, rel=1e-6)


def test_starvation_is_bit_identical_across_the_flag():
    off, fg_off = _build_env(mass_balance=False, s_x=0.05, starve_rate=0.5)
    on, fg_on = _build_env(mass_balance=True, s_x=0.05, starve_rate=0.5)
    _step(off, fg_off)
    _step(on, fg_on)
    assert np.array_equal(fg_off.biomass, fg_on.biomass)
    assert np.array_equal(fg_off.energy_reserve, fg_on.energy_reserve)


# ---------------------------------------------------------------------------
# 6. The GPU mirror.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mass_balance", [False, True])
@pytest.mark.parametrize("growth_rate", [0.1, 10.0])
# EC = 0 is the one branch where the two implementations are structurally
# different - the reference returns early, the mirror guards with
# ``torch.where(ec > 0, ...)`` - so parity has to be asserted there too.
@pytest.mark.parametrize("energy_content", [EC, 0.0])
def test_gpu_population_step_matches_the_reference(mass_balance, growth_rate,
                                                   energy_content):
    torch = pytest.importorskip("torch")
    from lib.gpu.ecosystem import TensorEcosystem

    env, fg = _build_env(mass_balance=mass_balance, growth_rate=growth_rate,
                         energy_content=energy_content)
    env.build_static_caches()
    env._season_phase = {'grazer': 0.0}
    model = TensorEcosystem(env, device="cpu")
    b = torch.as_tensor(np.asarray(fg.biomass).reshape(1, 1, H * W),
                        dtype=torch.float32)
    r = torch.as_tensor(np.asarray(fg.energy_reserve).reshape(1, 1, H * W),
                        dtype=torch.float32)
    gb, gr, _ = model.population(b, r, 0, 0.0, 1.0)

    population_change._apply_decision_maker_population_change(
        env, 'grazer', fg)
    np.testing.assert_allclose(
        gb.reshape(H, W).numpy(), np.asarray(fg.biomass), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        gr.reshape(H, W).numpy(), np.asarray(fg.energy_reserve),
        rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. The shared CLI helper.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spelling", ["--mass-balance", "--mass_balance"])
def test_cli_accepts_both_spellings(spelling):
    parser = argparse.ArgumentParser()
    add_mass_balance_argument(parser)
    assert parser.parse_args([spelling]).mass_balance is True


def test_cli_defaults_to_the_module_default():
    parser = argparse.ArgumentParser()
    add_mass_balance_argument(parser)
    assert parser.parse_args([]).mass_balance is DEFAULT_MASS_BALANCE


@pytest.mark.parametrize("spelling", ["--no-mass-balance", "--no_mass_balance"])
def test_cli_opt_out_turns_it_off(spelling):
    """The way back to the pre-116 tick, both spellings."""
    parser = argparse.ArgumentParser()
    add_mass_balance_argument(parser)
    assert parser.parse_args([spelling]).mass_balance is False
