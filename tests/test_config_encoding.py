"""Configuration must decode identically regardless of the host locale."""

import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import inference
from lib.config import config_loader
from lib.gpu.config import EnvironmentBuilder, ProjectSpec


@pytest.fixture
def windows_text_encoding(monkeypatch):
    """Emulate legacy Windows text defaults even on UTF-8 development hosts."""
    def windows_open(path, mode="r", **kwargs):
        if "b" not in mode:
            kwargs.setdefault("encoding", "cp1252")
        return builtins.open(path, mode, **kwargs)

    monkeypatch.setattr(config_loader, "open", windows_open, raising=False)
    monkeypatch.setattr(inference, "open", windows_open, raising=False)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_config_preserves_keys_and_unicode(tmp_path, windows_text_encoding, encoding):
    path = tmp_path / "library.yaml"
    path.write_text("species_definitions:\n  räka:\n    display_name: Räka\n", encoding=encoding)
    assert config_loader.load_config(path) == {
        "species_definitions": {"räka": {"display_name": "Räka"}},
    }


@pytest.mark.parametrize("gpu_schema", [False, True])
def test_bundled_project_loads_with_windows_defaults(windows_text_encoding, gpu_schema):
    builder = EnvironmentBuilder(project_path=str(ROOT / "mareld2.yaml"), grid=(4, 5))
    env = ProjectSpec(builder).env if gpu_schema else builder(seed=0)
    assert "gadoids" in env.fgs
    assert (env.H, env.W) == (4, 5)


def test_inference_spawn_settings_decode_utf8(tmp_path, windows_text_encoding):
    project = tmp_path / "project.yaml"
    project.write_text(
        "spawn_templates:\n  colony:\n    Skärgård:\n      n_colonies: 2\n"
        "decision_makers:\n  - group_id: gadoids\n",
        encoding="utf-8-sig",
    )
    library_dir = tmp_path / "fgconfig"
    library_dir.mkdir()
    (library_dir / "fg_library.yaml").write_text(
        "species_definitions:\n  gadoids:\n    spawn:\n      mode: colony\n",
        encoding="utf-8-sig",
    )
    assert inference._load_spawn_templates(project) == {
        "colony": {"Skärgård": {"n_colonies": 2}},
    }
    assert inference._load_spawn_defaults(project) == {"gadoids": "colony"}
