"""``--mortality_multiplier``: one global scale on every FG's rate.

The natural-mortality term is density independent and lives in the tick,
not in the reward path: ``population_change`` multiplies biomass and
energy reserve by ``1 - natural_mortality`` on the reference engine, and
``lib/gpu/ecosystem.py`` bakes the same factor into ``mortality_keep``.
The flag scales the rate itself, so both engines must pick it up and
they must agree.

Contracts asserted here:

1. The default 1.0 leaves the tick bit-identical - this is what keeps
   every existing run and checkpoint comparable.
2. The factor multiplies the rate, not the survival fraction: at
   ``m = 0.5`` a rate of 0.1 becomes 0.05, i.e. ``keep = 0.95``.
3. ``m = 0`` is exactly ``--mortality off``, and the flag is inert when
   mortality is off to begin with.
4. The GPU mirror agrees with the reference, including the clamp that
   keeps ``keep`` non-negative for absurdly large factors.
5. The shared CLI helper rejects negative and non-finite factors, so the
   CPU and GPU parsers cannot drift apart.
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
    add_mortality_multiplier_argument,
)
from lib.world.functional_group import FunctionalGroup  # noqa: E402

H = W = 3
RATE = 0.1
B0 = 4.0
GRID_CONFIG = {'width': W, 'height': H, 'cell_size': 1000.0,
               'tick_duration': 6.0}


def _build_env(multiplier=1.0, mortality=True, rate=RATE):
    """One decision maker with no prey, no movement and no growth.

    ``maintenance_level = 0`` and ``energy_level = 0`` make the surplus
    term vanish, so the only thing that can change the biomass in
    ``_apply_decision_maker_population_change`` is the mortality branch.
    """
    params = {
        'is_decision_maker': True,
        'max_energy_reserve': 1000.0,
        'energy_content': 1000.0,
        'resting_metabolism': 0.0,
        'maintenance_level': 0.0,
        'movement_speed': 0.0,
        'growth_rate': 0.0,
        'starve_rate': 0.0,
        'natural_mortality': rate,
        'min_split_biomass': 0.0,
        'extinction_threshold_factor': 0.0,
        'menu': [],
    }
    fg = FunctionalGroup('predator', params)
    fg.initialize_state((H, W), initial_biomass=np.full((H, W), B0))
    env = EcosystemEnvironment(GRID_CONFIG, {'predator': fg},
                               apply_natural_mortality=mortality,
                               mortality_multiplier=multiplier)
    env.ordered_fg_ids = ['predator']
    return env, fg


def _one_growth_step(env, fg):
    population_change._apply_decision_maker_population_change(
        env, 'predator', fg)
    return float(fg.biomass[0, 0])


# ---------------------------------------------------------------------------
# 1-3. The reference engine.
# ---------------------------------------------------------------------------

def test_default_multiplier_is_the_unscaled_library_rate():
    env, fg = _build_env()
    assert env.mortality_multiplier == 1.0
    assert _one_growth_step(env, fg) == pytest.approx(B0 * (1.0 - RATE))


@pytest.mark.parametrize("multiplier", [0.5, 2.0, 3.0])
def test_multiplier_scales_the_rate_not_the_survival_fraction(multiplier):
    """``keep = 1 - m*rate``, which is NOT ``(1 - rate)*m``."""
    env, fg = _build_env(multiplier)
    expected = B0 * (1.0 - multiplier * RATE)
    assert _one_growth_step(env, fg) == pytest.approx(expected, rel=1e-6)


def test_zero_multiplier_equals_mortality_off():
    scaled, fg_scaled = _build_env(0.0, mortality=True)
    off, fg_off = _build_env(1.0, mortality=False)
    assert _one_growth_step(scaled, fg_scaled) == B0
    assert _one_growth_step(off, fg_off) == B0


def test_multiplier_is_inert_when_mortality_is_off():
    env, fg = _build_env(5.0, mortality=False)
    assert _one_growth_step(env, fg) == B0


def test_negative_multiplier_is_clamped_by_the_environment():
    """Defensive: a negative rate would *create* biomass out of nothing."""
    env, _ = _build_env(-2.0)
    assert env.mortality_multiplier == 0.0


def test_huge_multiplier_cannot_drive_biomass_negative():
    env, fg = _build_env(100.0)
    assert _one_growth_step(env, fg) == 0.0


def test_energy_reserve_follows_the_scaled_rate():
    env, fg = _build_env(0.5)
    fg.energy_reserve = np.full((H, W), 20.0, dtype=env.dtype)
    population_change._apply_decision_maker_population_change(
        env, 'predator', fg)
    assert float(fg.energy_reserve[0, 0]) == pytest.approx(
        20.0 * (1.0 - 0.5 * RATE), rel=1e-6)


# ---------------------------------------------------------------------------
# 4. The tensor mirror.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("multiplier", [0.0, 0.5, 1.0, 4.0, 100.0])
def test_gpu_mortality_keep_mirrors_the_reference(multiplier):
    torch = pytest.importorskip("torch")
    from lib.gpu.ecosystem import TensorEcosystem

    env, _ = _build_env(multiplier)
    model = TensorEcosystem(env, "cpu")
    keep = model.mortality_keep.reshape(-1).tolist()
    assert keep == [pytest.approx(max(0.0, 1.0 - multiplier * RATE))]
    assert torch.isfinite(model.mortality_keep).all()


def test_gpu_ignores_the_multiplier_when_mortality_is_off():
    pytest.importorskip("torch")
    from lib.gpu.ecosystem import TensorEcosystem

    env, _ = _build_env(0.5, mortality=False)
    model = TensorEcosystem(env, "cpu")
    assert model.mortality_keep.reshape(-1).tolist() == [1.0]


# ---------------------------------------------------------------------------
# 5. The shared CLI flag.
# ---------------------------------------------------------------------------

def _parse(argv):
    parser = argparse.ArgumentParser()
    add_mortality_multiplier_argument(parser)
    return parser.parse_args(argv)


def test_flag_defaults_to_one_and_accepts_both_spellings():
    assert _parse([]).mortality_multiplier == 1.0
    assert _parse(["--mortality_multiplier", "0.5"]).mortality_multiplier == 0.5
    assert _parse(["--mortality-multiplier", "2"]).mortality_multiplier == 2.0


@pytest.mark.parametrize("value", ["-0.5", "nan", "inf", "abc"])
def test_flag_rejects_negative_and_non_finite(value):
    with pytest.raises(SystemExit):
        _parse(["--mortality_multiplier", value])


def test_gpu_builder_carries_the_multiplier_through_with_world():
    """``with_world`` rebuilds the builder positionally - easy to drop."""
    from lib.gpu.config import EnvironmentBuilder

    builder = EnvironmentBuilder(mortality=True, mortality_multiplier=0.25)
    assert builder.with_world(7).mortality_multiplier == 0.25


# ---------------------------------------------------------------------------
# 6. The monitoring envs must use the same ecology as the training rollouts.
#
# ``--mortality on --mortality_multiplier 0`` has to behave exactly like
# ``--mortality off`` everywhere, not only inside the tick: the survival
# progress evaluator and the live probe build their own environments, and
# each dropped factor silently reintroduces the full library mortality.
# ---------------------------------------------------------------------------

def _cpu_builder(multiplier, mortality=True):
    from types import SimpleNamespace

    return SimpleNamespace(project_path=None, grid_height=3, grid_width=3,
                           apply_natural_mortality=mortality, migration=False,
                           mortality_multiplier=multiplier, currents=None)


def test_progress_config_carries_the_multiplier():
    from types import SimpleNamespace

    from lib.runners.training_progress import inference_config

    trainer = SimpleNamespace(env_builder=_cpu_builder(0.0))
    assert inference_config(trainer, "cpu")["mortality_multiplier"] == 0.0


def test_progress_env_is_built_with_the_multiplier():
    from types import SimpleNamespace

    from lib.runners.training_progress import (build_inference_env,
                                               inference_config)

    config = inference_config(SimpleNamespace(env_builder=_cpu_builder(0.0)),
                              "cpu")
    env = build_inference_env(config, seed=7)
    assert env.mortality_multiplier == 0.0


def test_progress_config_without_the_key_still_resumes():
    """Histories written before the factor existed used the default."""
    from lib.runners.training_progress import comparable_config

    stored = {"grid": [3, 3], "mortality": True}
    fresh = dict(stored, mortality_multiplier=1.0)
    assert comparable_config(stored) == comparable_config(fresh)
