"""Impact pipeline: project YAML -> loader metadata -> FG impact tables.

``mareld2.yaml`` ships with every impact ``muted: true``, so the active
metadata is empty for the project as-is. These tests therefore cover both
cases: the muted project (empty metadata, but the 4-value contract intact)
and a temporary un-muted copy (metadata + a usable per-FG impact table that
``lib.environments.ecosystem_env.impacts`` can consume).
"""

import numpy as np
import pytest
import yaml

from lib.config.config_loader import load_project_config, load_impact_spawn_specs
from lib.environments.ecosystem import EcosystemEnvironment
from lib.environments.ecosystem_env import impacts as impacts_mod

PROJECT = 'mareld2.yaml'
GRID = (20, 20)


@pytest.fixture(scope='module')
def unmuted_project(tmp_path_factory):
    """A copy of the project with ``windfarm_noise`` un-muted."""
    with open(PROJECT) as f:
        data = yaml.safe_load(f)
    for iv in data.get('impact_variables', []) or []:
        if iv.get('impact_id') == 'windfarm_noise':
            iv['muted'] = False
    path = tmp_path_factory.mktemp('impacts') / 'project_unmuted.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path)


def test_load_project_config_returns_four_values():
    out = load_project_config(PROJECT, grid_size=GRID, seed=1)
    assert len(out) == 4
    fgs, impact_vars, impact_ranges, observable_impact_vars = out
    assert fgs
    # All impacts are muted in the shipped project.
    assert impact_vars == []
    assert impact_ranges == {}
    assert observable_impact_vars == []


def test_unmuted_project_yields_impact_metadata(unmuted_project):
    fgs, impact_vars, impact_ranges, observable = load_project_config(
        unmuted_project, grid_size=GRID, seed=1)
    assert impact_vars == ['windfarm_noise']
    assert observable == ['windfarm_noise']
    vmin, vmax = impact_ranges['windfarm_noise']
    assert vmin == pytest.approx(0.0)
    assert vmax > vmin
    # Spawn strategy for the impact map is read from the project file.
    specs = load_impact_spawn_specs(unmuted_project)
    assert 'windfarm_noise' in specs


def test_impacted_fg_gets_usable_impact_table(unmuted_project):
    fgs, impact_vars, _ranges, observable = load_project_config(
        unmuted_project, grid_size=GRID, seed=1)
    assert 'porpoises' in fgs
    impact_block = fgs['porpoises'].params.get('impact')
    assert impact_block and 'windfarm_noise' in impact_block

    table = impacts_mod.extract_impact_table(impact_block['windfarm_noise'])
    assert table is not None, "porpoises must have a usable impact table"
    xs, bf, ef = table
    assert xs.ndim == bf.ndim == ef.ndim == 1
    assert len(xs) == len(bf) == len(ef) >= 2
    assert np.all(np.diff(xs) > 0), "table must be sorted by value"

    # Interpolation clips to the endpoints (no extrapolation).
    b_lo, e_lo = impacts_mod.interp_impact(table, float(xs[0]) - 1000.0)
    b_hi, e_hi = impacts_mod.interp_impact(table, float(xs[-1]) + 1000.0)
    assert float(b_lo) == pytest.approx(float(bf[0]))
    assert float(e_lo) == pytest.approx(float(ef[0]))
    assert float(b_hi) == pytest.approx(float(bf[-1]))
    assert float(e_hi) == pytest.approx(float(ef[-1]))


def test_impact_table_reaches_env_and_costs_energy(unmuted_project):
    fgs, impact_vars, ranges, observable = load_project_config(
        unmuted_project, grid_size=GRID, seed=3)
    H, W = GRID
    grid_config = {'width': W, 'height': H, 'cell_size': 1000.0,
                   'tick_duration': 6.0}
    env = EcosystemEnvironment(grid_config, fgs,
                               observable_impact_vars=observable)
    vmax = ranges['windfarm_noise'][1]
    for iid in impact_vars:
        env.grid.add_map(iid, np.full((H, W), vmax, dtype=np.float32))
    env.build_static_caches()

    # The porpoise table is the only one with impact_affects=true.
    idx = env.dm_ids.index('porpoises')
    entries = env.dm_impact_tables[idx]
    assert [iid for iid, _t in entries] == ['windfarm_noise']

    # Observable impacts occupy one centre slot plus one per neighbour.
    factor = impacts_mod.impact_energy_cost_factor(env)
    assert factor.shape == (env.N_dm, H, W)
    assert float(factor[idx].max()) > 1.0, (
        "saturated noise map must raise the porpoise metabolic cost")


def test_observable_impacts_widen_the_observation_layout(unmuted_project):
    H, W = GRID
    grid_config = {'width': W, 'height': H, 'cell_size': 1000.0,
                   'tick_duration': 6.0}
    dims = {}
    for observable in ([], ['windfarm_noise']):
        fgs, impact_vars, ranges, _obs = load_project_config(
            unmuted_project, grid_size=GRID, seed=5)
        env = EcosystemEnvironment(grid_config, fgs,
                                   observable_impact_vars=observable)
        for iid in impact_vars:
            env.grid.add_map(iid, np.zeros((H, W), dtype=np.float32))
        env.build_static_caches()
        dims[len(observable)] = np.asarray(env.per_dm_in_dim).copy()

    # One centre channel + four neighbour channels per observable impact.
    assert np.all(dims[1] - dims[0] == 5)
