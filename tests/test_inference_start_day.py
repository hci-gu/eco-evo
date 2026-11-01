"""``daylight.inference_start_day_of_year`` fixes the start day of inference
worlds (inference.py, the long-run probe) and leaves training worlds on
``start_day_of_year`` (section 232.6)."""
import pytest

from lib.world import daylight


def _project(**block):
    base = {"enabled": True, "latitude_deg": 58.15, "start_day_of_year": "random"}
    base.update(block)
    return {"simulation_settings": {"daylight": base}}


def test_inference_day_only_in_inference_mode():
    project = _project(inference_start_day_of_year=1)
    assert daylight.parse_settings(project)["start_day_of_year"] is None
    assert daylight.parse_settings(project, mode="train")["start_day_of_year"] is None
    assert daylight.parse_settings(project, mode="inference")["start_day_of_year"] == 1


def test_without_the_key_inference_keeps_the_training_start():
    assert daylight.parse_settings(_project(), mode="inference")["start_day_of_year"] is None
    fixed = _project(start_day_of_year=148)
    assert daylight.parse_settings(fixed, mode="inference")["start_day_of_year"] == 148


def test_bad_inference_day_is_refused():
    with pytest.raises(ValueError):
        daylight.parse_settings(_project(inference_start_day_of_year=0), mode="inference")
    with pytest.raises(ValueError):
        daylight.parse_settings(_project(inference_start_day_of_year="May"))


def test_loader_starts_inference_worlds_on_the_day(tmp_path):
    from lib.config.config_loader import load_config, load_project_config
    project = load_config("mareld2.yaml")
    project["simulation_settings"]["daylight"]["inference_start_day_of_year"] = 1
    starts = set()
    for seed in (1, 2, 3):
        fgs, *_ = load_project_config("mareld2.yaml", grid_size=(5, 6), seed=seed,
                                      mode="inference", project_config=project)
        starts.add(next(iter(fgs.values())).params["daylight"]["start_tick"])
    assert starts == {0}
    train = set()
    for seed in (1, 2, 3):
        fgs, *_ = load_project_config("mareld2.yaml", grid_size=(5, 6), seed=seed,
                                      project_config=project)
        train.add(next(iter(fgs.values())).params["daylight"]["start_tick"])
    assert train != {0}
