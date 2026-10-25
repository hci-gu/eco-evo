"""Shared pytest setup for the suite.

``torch.compile`` budgets recompilations per code object, and the GPU
tests build many ``RolloutRunner`` instances whose ``_tick`` differs in
shapes, flags and dtypes. Run as a suite, those variants accumulate on
one frame and exhaust ``torch._dynamo.config.recompile_limit`` (8 by
default), so the CUDA execution tests fail with
``FailOnRecompileLimitHit`` in the full run and pass on their own. A
training run compiles a single runner, so the budget is a property of
the test process, not of production: give every test a clean cache and
keep the limit meaningful as a per-test statement.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _fresh_dynamo_cache():
    yield
    torch._dynamo.reset()


@pytest.fixture(autouse=True)
def _private_viewer_settings(tmp_path, monkeypatch):
    """Every test gets its own viewer settings file (section 213).

    The live viewer remembers the display language in
    lib/viz/viewer_settings.json; without this a language the user picked
    would change what the viewer tests see, and the tests would overwrite
    the user's choice.
    """
    monkeypatch.setenv("MARELD_VIEWER_SETTINGS",
                       str(tmp_path / "viewer_settings.json"))
