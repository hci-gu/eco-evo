"""train_gpu --init-from: a bare run name resolves to results/<name> first."""
import pytest

from train_gpu import resolve_init_from


def test_bare_name_prefers_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "run1").mkdir(parents=True)
    (tmp_path / "run1").mkdir()
    assert resolve_init_from("run1") == tmp_path.joinpath("results", "run1").relative_to(tmp_path)


def test_bare_name_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run2").mkdir()
    assert str(resolve_init_from("run2")) == "run2"


def test_path_used_as_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "run3").mkdir(parents=True)
    assert str(resolve_init_from("results/run3")) == "results/run3"
    assert resolve_init_from(tmp_path / "results" / "run3") == tmp_path / "results" / "run3"


def test_missing_directory_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="results/nope or nope"):
        resolve_init_from("nope")
