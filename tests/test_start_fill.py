"""Field start reserve and the fish starvation line (section 247).

Contracts asserted:

1. ``start_fill`` is 12 fills in [0, 1] on a decision maker; an NDM or a
   bad table is refused. A migrating DM without it falls back to its
   ``outside_fill``.
2. ``FunctionalGroup.initialize_state``: with a table, inference starts at
   the table's value on the start day (the annual mean without a day) and
   training draws in [c - 0.3, c + 0.3] clipped; without a table the old
   u-centred / 0.7 path is unchanged.
3. The loader passes the calendar's start day; the GPU spawner's centre per
   world equals the CPU value on the same start tick.
4. The mareld2 library values of the five fish.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.world import daylight
from lib.world.functional_group import FunctionalGroup

FILL = [0.5, 0.36, 0.26, 0.2, 0.32, 0.81, 0.94, 0.96, 0.88, 0.88, 0.76, 0.68]
DM = {"is_decision_maker": True, "max_energy_reserve": 1000.0, "maintenance_level": 0.08,
      "start_fill": FILL}


def test_table_is_checked():
    assert FunctionalGroup("a", DM).start_fill == FILL
    for bad in (FILL[:11], FILL[:11] + [1.2], "x"):
        with pytest.raises(ValueError):
            FunctionalGroup("a", dict(DM, start_fill=bad))
    with pytest.raises(ValueError):
        FunctionalGroup("n", {"start_fill": FILL})
    assert FunctionalGroup("a", {k: v for k, v in DM.items() if k != "start_fill"}).start_fill is None


@pytest.mark.parametrize("day", [None, 15.0, 100.0, 300.0])
def test_inference_starts_at_the_field_fill(day):
    fg = FunctionalGroup("a", DM)
    b = np.full((3, 4), 2.0)
    fg.initialize_state((3, 4), initial_biomass=b, start_day=day)
    centre = float(np.mean(FILL)) if day is None else float(daylight.monthly_value(FILL, day))
    np.testing.assert_allclose(fg.energy_reserve, b * centre * 1000.0, rtol=1e-12)
    for month in (0, 3, 8):
        assert fg.start_fill_centre(daylight._MONTH_MID_DAY[month]) == pytest.approx(FILL[month])


def test_training_draws_around_the_field_fill():
    fg = FunctionalGroup("a", DM)
    b = np.ones((40, 40))
    fg.initialize_state((40, 40), initial_biomass=b, randomize_energy=True,
                        rng=np.random.default_rng(1), start_day=daylight._MONTH_MID_DAY[0])
    s = fg.energy_reserve / 1000.0
    assert s.min() >= 0.2 - 1e-12 and s.max() <= 0.8 + 1e-12
    assert s.mean() == pytest.approx(0.5, abs=0.02)
    fg.initialize_state((40, 40), initial_biomass=b, randomize_energy=True,
                        rng=np.random.default_rng(1), start_day=daylight._MONTH_MID_DAY[3])  # April, 0.2
    s = fg.energy_reserve / 1000.0
    assert s.min() >= 0.0 and s.max() <= 0.5 + 1e-12


def test_groups_without_a_table_keep_the_old_start():
    plain = {k: v for k, v in DM.items() if k != "start_fill"}
    fg = FunctionalGroup("a", dict(plain, maintenance_level=0.5))
    b = np.ones((30, 30))
    fg.initialize_state((30, 30), initial_biomass=b, start_day=100.0)
    np.testing.assert_allclose(fg.energy_reserve, 0.7 * 1000.0)
    fg.initialize_state((30, 30), initial_biomass=b, randomize_energy=True,
                        rng=np.random.default_rng(2), start_day=100.0)
    s = fg.energy_reserve / 1000.0
    assert s.min() >= 0.2 - 1e-12 and s.max() <= 0.8 + 1e-12


def test_migrating_dm_falls_back_to_its_arrival_fill():
    arrival = [0.7, 0.58, 0.45, 0.37, 0.13, 0.51, 0.57, 0.91, 1.0, 1.0, 0.94, 0.85]
    params = {"is_decision_maker": True, "max_energy_reserve": 1.0, "outside_density": 1.0,
              "outside_exchange_rate": 0.01, "outside_fill": arrival,
              "outside_entry": [{"edge": "west"}]}
    fg = FunctionalGroup("m", params)
    assert fg.start_fill == arrival
    assert FunctionalGroup("m", dict(params, start_fill=FILL)).start_fill == FILL


def _project_fgs(mode):
    from lib.config.config_loader import load_project_config
    return load_project_config(str(ROOT / "mareld2.yaml"), mode=mode, spawn_seed=5,
                               grid_size=(6, 7))[0]


def test_loader_passes_the_start_day():
    fgs = _project_fgs("inference")
    fg = fgs["planktivorous_fish"]
    cfg = fg.params["daylight"]
    day = cfg["start_tick"] * cfg["tick_hours"] / 24.0
    s = fg.energy_reserve / (fg.biomass * fg.max_energy_reserve)
    np.testing.assert_allclose(s[fg.biomass > 0], fg.start_fill_centre(day), rtol=1e-6)
    # Zooplankton has no table: the old 0.7.
    z = fgs["zooplankton"]
    np.testing.assert_allclose((z.energy_reserve / (z.biomass * z.max_energy_reserve))[z.biomass > 0],
                               0.7, rtol=1e-6)


def test_gpu_spawner_centre_matches_the_cpu_value():
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.ecosystem import TensorEcosystem
    from lib.gpu.spawn import WorldSpawner
    spec = ProjectSpec(EnvironmentBuilder(project_path=str(ROOT / "mareld2.yaml"), grid=(6, 7)))
    model = TensorEcosystem(spec.env, "cpu")
    spawner = WorldSpawner(spec, model)
    keys = torch.arange(3)
    biomass = spawner.biomass(keys)
    light = torch.tensor([0, 400, 1100])
    hours = float(spec.env.daylight["tick_hours"])
    fixed = copy.copy(spec)
    fixed.randomize_energy = False
    spawner_fixed = WorldSpawner(fixed, model)
    reserve = spawner_fixed.reserves(biomass, keys, light)
    drawn = spawner.reserves(biomass, keys, light)
    for j, fid in enumerate(model.ids):
        fg = spec.env.fgs[fid]
        cap = biomass[:, j] * model.max_reserve[0, j, 0]
        alive = cap > 0
        for w in range(3):
            fill = (reserve[w, j][alive[w]] / cap[w][alive[w]])
            s = (drawn[w, j][alive[w]] / cap[w][alive[w]])
            if getattr(fg, "start_fill", None) is None:
                if fid in spec.env.dm_ids:
                    torch.testing.assert_close(fill, torch.full_like(fill, 0.7))
                continue
            c = fg.start_fill_centre(float(light[w]) * hours / 24.0)
            torch.testing.assert_close(fill, torch.full_like(fill, c), rtol=1e-5, atol=1e-6)
            assert float(s.min()) >= max(0.0, c - 0.3) - 1e-5
            assert float(s.max()) <= min(1.0, c + 0.3) + 1e-5


def test_mareld2_fish_values():
    import yaml
    sd = yaml.safe_load(open(ROOT / "fgconfig" / "fg_library.yaml"))["species_definitions"]
    expected = {"planktivorous_fish": (0.079, 0.0675), "benthivorous_fish": (0.078, 0.0791),
                "whiting": (0.070, 0.0866), "predatory_fish": (0.070, 0.0681),
                "mackerel": (0.057, 0.0942)}
    for fid, (u, rate) in expected.items():
        assert sd[fid]["maintenance_level"] == pytest.approx(u), fid
        assert sd[fid]["starve_rate"] == pytest.approx(rate), fid
        table = sd[fid].get("start_fill") or sd[fid].get("outside_fill")
        assert len(table) == 12 and min(table) > u, fid      # living fish above the line
    for fid in ("zooplankton", "porpoises", "seals"):
        assert sd[fid]["maintenance_level"] == 0.5 and "start_fill" not in sd[fid]
