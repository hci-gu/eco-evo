"""Paper scenario runs (lib/scenarios): the scenario file, the manifest
changes, warming relative to today and the impact schedule, masks and
per-impact catch of ``run_variant``."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import inference  # noqa: E402
from lib.scenarios import runner as scenario_runner  # noqa: E402
from lib.scenarios.spec import (MaterializedManifest, deep_merge, expand_scale,  # noqa: E402
                                load_scenario, resolve_same_as, scenario_manifest)
from lib.world import temperature  # noqa: E402
from test_inference_gpu_engine import _policies  # noqa: E402

GRID = (6, 7)


def _write(tmp_path, spec, name="s.yaml"):
    spec = dict(spec)
    spec.setdefault("base", str(ROOT / "mareld2.yaml"))
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec))
    return str(path)


# ------------------------------------------------------------------ spec

def test_same_as_needs_the_same_world(tmp_path):
    _write(tmp_path, {"seeds": [0], "schedule": [{"from_year": 0, "scale": {"fishing": 1.0}}]},
           "today.yaml")
    sweep = {"seeds": [0], "warming_c": 0.0,
             "schedule": [{"from_year": 0, "scale": {"fishing": 1.0}}],
             "sweep": {"name": "dT", "values": [0.0, 2.0], "set": ["warming_c"]}}
    v0, v2 = load_scenario(_write(tmp_path, {**sweep, "same_as": {"dT=0": "today/base"}},
                                  "warm.yaml"))
    assert resolve_same_as(v0).scenario == "today" and resolve_same_as(v2) is None
    # dT = 2 is not today's world.
    bad = load_scenario(_write(tmp_path, {**sweep, "same_as": {"dT=2": "today/base"}},
                               "warm_bad.yaml"))[1]
    with pytest.raises(ValueError, match="warming_c"):
        resolve_same_as(bad)
    with pytest.raises(ValueError, match="no variant"):
        load_scenario(_write(tmp_path, {**sweep, "same_as": {"dT=9": "today/base"}}, "w.yaml"))


def test_the_paper_aliases_are_the_same_worlds():
    from glob import glob
    runs = 0
    for path in glob(str(ROOT / "scenarios" / "paper" / "*.yaml")):
        for v in load_scenario(path):
            runs += resolve_same_as(v) is None    # raises on a different world
    assert runs == 11


def test_find_runs_follows_a_link(tmp_path):
    from lib.scenarios import results
    src = tmp_path / "today" / "base" / "seed_0"
    link = tmp_path / "warm" / "dT=0" / "seed_0"
    meta = {"scenario": "today", "variant": "base", "seed": 0, "years": 1.0,
            "grid": [6, 7], "run_name": "x"}
    src.mkdir(parents=True)
    link.mkdir(parents=True)
    (src / "meta.json").write_text(json.dumps(meta))
    (link / "meta.json").write_text(json.dumps(
        {**meta, "scenario": "warm", "variant": "dT=0", "sweep_name": "dT",
         "sweep_value": 0.0, "data": "../../../today/base/seed_0"}))
    runs = results.find_runs(str(tmp_path)).set_index("scenario")
    assert Path(runs.loc["warm", "path"]) == src
    assert runs.loc["warm", "sweep_value"] == 0.0


def test_sweep_makes_one_variant_per_value(tmp_path):
    path = _write(tmp_path, {
        "schedule": [{"from_year": 0, "scale": {"fishing": 0.2}},
                     {"from_year": 5, "scale": {"fishing": 1.0}}],
        "sweep": {"name": "k", "values": [0.5, 2.0], "set": ["schedule.1.scale.fishing"]},
    }, "opt.yaml")
    variants = load_scenario(path)
    assert [v.name for v in variants] == ["k=0.5", "k=2"]
    assert all(v.scenario == "opt" for v in variants)
    sched = variants[1].schedule()
    assert sched[0] == (0.0, {"bottom_trawling": 0.2, "pelagic_trawling": 0.2})
    assert sched[1] == (5.0, {"bottom_trawling": 2.0, "pelagic_trawling": 2.0})
    assert variants[1].warmup_scale() == sched[0][1]


def test_schedule_carries_unchanged_intensities_over(tmp_path):
    v = load_scenario(_write(tmp_path, {"schedule": [
        {"from_year": 1, "scale": {"bottom_trawling": 0.0}},
        {"from_year": 2, "scale": {"pelagic_trawling": 3.0}}]}))[0]
    assert v.schedule() == [(0.0, {}), (1.0, {"bottom_trawling": 0.0}),
                            (2.0, {"bottom_trawling": 0.0, "pelagic_trawling": 3.0})]


def test_explicit_impact_wins_over_alias():
    assert expand_scale({"fishing": 0.5, "pelagic_trawling": 2}) == {
        "bottom_trawling": 0.5, "pelagic_trawling": 2.0}


def test_unknown_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown scenario keys"):
        load_scenario(_write(tmp_path, {"warming": 2}))


def test_deep_merge_merges_lists_by_id():
    base = {"impact_variables": [{"impact_id": "a", "value_max": 1},
                                 {"impact_id": "b", "muted": True}],
            "other": [1, 2]}
    out = deep_merge(base, {"impact_variables": [{"impact_id": "b", "muted": False}],
                            "other": [3]})
    assert out["impact_variables"] == [{"impact_id": "a", "value_max": 1},
                                       {"impact_id": "b", "muted": False}]
    assert out["other"] == [3]
    assert base["impact_variables"][1]["muted"] is True


def test_manifest_changes(tmp_path):
    v = load_scenario(_write(tmp_path, {
        "nutrient_reservoir_scale": 2.0,
        "impacts": {"windfarm_noise": {"map": "x.npz"}}}))[0]
    project = scenario_manifest(v)
    with open(ROOT / "mareld2.yaml") as f:
        base = yaml.safe_load(f)
    old = base["simulation_settings"]["nutrients"]["subsurface_layer"]["reservoir_umol_l"]
    new = project["simulation_settings"]["nutrients"]["subsurface_layer"]["reservoir_umol_l"]
    np.testing.assert_allclose(new, 2.0 * np.asarray(old))
    noise = [iv for iv in project["impact_variables"] if iv["impact_id"] == "windfarm_noise"][0]
    # Present, but never observed: the checkpoints' input layout is kept.
    assert noise["muted"] is False and noise["observable"] is False


def test_materialized_manifest_is_removed(tmp_path):
    v = load_scenario(_write(tmp_path, {}))[0]
    with MaterializedManifest(v) as path:
        assert Path(path).parent == ROOT
        assert Path(path).exists()
    assert not Path(path).exists()


# ------------------------------------------------------------------ warming

def test_warming_is_q10_relative_to_today():
    env = inference.build_env(str(ROOT / "mareld2.yaml"), GRID, seed=1, verbose=False)
    met = env.metabolism_temp_table.copy()
    ing = env.ingestion_temp_table.copy()
    gro = env.growth_temp_table.copy()
    grazers = [e["temperature"].copy() if e["temperature"] is not None else None
               for e in env.ndm_grazers]
    applied = scenario_runner.apply_warming(env, 2.0)
    for i, fid in enumerate(env.dm_ids):
        params = env.fgs[fid].params
        s = temperature.species_settings(params) if params.get("temperature") else None
        f = float(s[0]) ** 0.2 if s else 1.0
        np.testing.assert_allclose(env.metabolism_temp_table[:, i], met[:, i] * f, rtol=1e-6)
        q = temperature.ingestion_q10(params) if params.get("temperature") else None
        f = float(q) ** 0.2 if q else 1.0
        np.testing.assert_allclose(env.ingestion_temp_table[:, i], ing[:, i] * f, rtol=1e-6)
        q = temperature.growth_q10(params) if params.get("temperature") else None
        f = float(q) ** 0.2 if q else 1.0
        np.testing.assert_allclose(env.growth_temp_table[:, i], gro[:, i] * f, rtol=1e-6)
    for entry, old in zip(env.ndm_grazers, grazers):
        if old is None:
            continue
        q = float(env.fgs[entry["grazer"]].params["grazing_q10"])
        np.testing.assert_allclose(entry["temperature"], old * q ** 0.2, rtol=1e-6)
    # Every fish group with a Q10 is warmed, not only the fixed-t_ref one.
    assert "ingestion" in applied["benthivorous_fish"]
    assert "metabolism" in applied["planktivorous_fish"]


# ------------------------------------------------------------------ runs

@pytest.fixture
def random_policies(monkeypatch):
    def load(env, checkpoint_dir, verbose=True):
        return _policies(env)
    monkeypatch.setattr(inference, "load_policies_and_stats", load)


def _run(tmp_path, spec, days=4):
    spec = dict(spec)
    spec.setdefault("run_name", "dummy")
    spec.setdefault("run", {"grid": list(GRID), "warmup_days": 0, "currents": "off"})
    v = load_scenario(_write(tmp_path, spec))[0]
    out = tmp_path / "out"
    scenario_runner.run_variant(v, 3, str(out), device="cpu", years=days / 365.0,
                                verbose=False)
    return dict(np.load(out / "timeseries.npz")), json.loads((out / "meta.json").read_text())


def test_catch_per_impact_adds_up_to_the_impact_loss(tmp_path, random_policies):
    ts, meta = _run(tmp_path, {"schedule": [{"from_year": 0, "scale": {"fishing": 1.0}}]})
    impact = list(ts["loss_causes"]).index("impact")
    assert list(ts["impact_ids"]) == ["bottom_trawling", "pelagic_trawling"]
    np.testing.assert_allclose(ts["catch"].sum(1), ts["losses"][:, impact], rtol=1e-6, atol=1e-9)
    g = list(ts["group_ids"])
    # Only pelagic trawling has a table on... the planktivores take both;
    # the benthos only bottom trawling.
    assert ts["catch"][-1, 1, g.index("benthic_infauna")] == 0.0
    assert ts["catch"][-1, 0, g.index("benthic_infauna")] > 0.0
    assert ts["biomass"].shape == (5, len(g))
    assert meta["ticks"] == 16


def test_schedule_switches_the_fishing_on(tmp_path, random_policies):
    ts, meta = _run(tmp_path, {"schedule": [
        {"from_year": 0, "scale": {"fishing": 0.0}},
        {"from_year": 2 / 365.0, "scale": {"fishing": 1.0}}]})
    impact = list(ts["loss_causes"]).index("impact")
    total = ts["losses"][:, impact].sum(-1)
    assert total[2] == 0.0                     # days 0-2 unfished
    assert total[4] > total[3] > 0.0           # fished from day 2
    # The intensity in force over the interval ending at each row.
    np.testing.assert_allclose(ts["scale"][:, 0], [0, 0, 0, 1, 1])


def test_mask_closes_the_masked_cells(tmp_path, random_policies):
    mask = np.ones(GRID, dtype=np.float32)
    np.savez(tmp_path / "m.npz", mask=mask)
    ts, _ = _run(tmp_path, {"masks": [{"map": "m.npz", "impacts": ["fishing"]}]})
    impact = list(ts["loss_causes"]).index("impact")
    assert ts["losses"][-1, impact].sum() == 0.0
