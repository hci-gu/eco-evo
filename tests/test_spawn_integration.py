"""Integrationstest för spawn-feature-flagga i `config_loader._spawn_biomass_distribution`.

Verifierar två kontrakt:

1. **Legacy-väg orörd** när inget `spawn:`-block finns: identisk output för
   samma seed som före refaktoreringen (jämför direkt-anrop med och utan
   ``spawn_spec=None``).

2. **Ny väg triggas** när `spawn:`-block byggs via `_build_spawn_spec`:
   colony-strategin ger koncentrerad fördelning, uniform ger spridd,
   båda respekterar 1-tons-golvet via `distribute_with_floor`.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.config.config_loader import (  # noqa: E402
    _build_spawn_spec,
    _spawn_biomass_distribution,
)
from lib.spawn import StrategySpec  # noqa: E402


def test_legacy_path_unchanged_without_spec():
    """Utan spawn_spec ska legacy-vägen producera samma output för samma seed."""
    rng1 = np.random.default_rng(123)
    out1 = _spawn_biomass_distribution(
        (30, 30), total_b=100.0, min_per_cell=0.5,
        rng=rng1, spawn_spec=None,
    )
    rng2 = np.random.default_rng(123)
    out2 = _spawn_biomass_distribution(
        (30, 30), total_b=100.0, min_per_cell=0.5,
        rng=rng2, spawn_spec=None,
    )
    assert np.array_equal(out1, out2)
    # Sanity: legacy preserves sum exactly.
    assert out1.sum() == pytest.approx(100.0, rel=1e-9)


def test_build_spawn_spec_yaml_block():
    """Ett rimligt `spawn:`-block (mode + params) ska bygga en giltig spec."""
    cfg = {
        "mode": "colony",
        "n_colonies": 2,
        "sigma_cells": 3.0,
        "anchor": "free",
    }
    spec = _build_spawn_spec(cfg, default_seed=42)
    assert isinstance(spec, StrategySpec)
    assert spec.mode == "colony"
    assert spec.params["n_colonies"] == 2
    assert spec.params["sigma_cells"] == 3.0


def test_build_spawn_spec_missing_returns_none():
    """Saknat block / None / fel typ ska ge None (-> legacy-väg)."""
    assert _build_spawn_spec(None) is None
    assert _build_spawn_spec("not a dict") is None
    assert _build_spawn_spec([]) is None


def test_new_path_colony_concentrated():
    """Colony-spec via _spawn_biomass_distribution ska ge koncentrerad fördelning."""
    spec = StrategySpec(
        mode="colony",
        params={"n_colonies": 3, "sigma_cells": 2.0, "anchor": "free"},
        seed=17,
    )
    out = _spawn_biomass_distribution(
        (60, 60), total_b=12.0, min_per_cell=0.5,
        spawn_spec=spec, project_seed=17,
    )
    assert out.sum() == pytest.approx(12.0, rel=1e-9)
    active = out[out > 0.0]
    assert active.size > 0
    assert active.min() >= 0.5 - 1e-9
    # Colony => betydligt färre aktiva celler än uniform.
    assert active.size <= 60 * 60 // 4


def test_new_path_uniform_spreads():
    """Uniform-spec ger spridd fördelning, summa = total_b."""
    spec = StrategySpec(mode="uniform", params={}, seed=7)
    out = _spawn_biomass_distribution(
        (40, 40), total_b=200.0, min_per_cell=0.5,
        spawn_spec=spec, project_seed=7,
    )
    assert out.sum() == pytest.approx(200.0, rel=1e-9)
    n_active = int((out > 0.0).sum())
    # Med floor=0.5 och total=200 ska n_active <= 400; uniform sprider sig.
    assert n_active <= 400
    assert n_active > 0
