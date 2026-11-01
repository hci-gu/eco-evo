"""A manifest's top-level ``library`` key names the library a run reads.

An explicit library path other than the repository default still wins
(the probes' variant libraries); without the key the default library is
read as before (config_loader.resolve_library_path).
"""
import os

from lib.config.config_loader import (load_config, load_project_config,
                                      resolve_library_path)
from lib.gpu.config import DEFAULT_LIBRARY, EnvironmentBuilder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.join(ROOT, "mareld2.yaml")
LIBRARY = os.path.join(ROOT, "fgconfig", "fg_library.yaml")


def _variant_library(tmp_path, factor=0.5):
    text = open(LIBRARY, encoding="utf-8-sig").read()
    old = "    resting_metabolism: 4.59"
    assert text.count(old) == 1
    path = tmp_path / "variant.yaml"
    path.write_text(text.replace(old, "    resting_metabolism: %g" % (4.59 * factor)),
                    encoding="utf-8")
    return str(path)


def _manifest_with_key(tmp_path, library):
    lines = open(PROJECT, encoding="utf-8-sig").read().split("\n")
    root = ROOT.replace(os.sep, "/") + "/"
    out = []
    for line in lines:
        # Keep the manifest's relative map paths valid from tmp_path.
        for rel in ("maps/", "vindparker.npz"):
            line = line.replace(": " + rel, ": " + root + rel)
        out.append(line)
    out.insert(0, "library: %s" % library)
    path = tmp_path / "manifest.yaml"
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)


def test_resolver_precedence(tmp_path):
    project = {"library": "libs/other.yaml"}
    manifest = str(tmp_path / "p.yaml")
    assert resolve_library_path(project, manifest) == os.path.join(str(tmp_path), "libs", "other.yaml")
    assert resolve_library_path(project, manifest, DEFAULT_LIBRARY) == os.path.join(
        str(tmp_path), "libs", "other.yaml")
    assert resolve_library_path(project, manifest, "/x/explicit.yaml") == "/x/explicit.yaml"
    assert resolve_library_path({}, manifest) == "fgconfig/fg_library.yaml"


def test_loader_reads_the_manifest_library(tmp_path):
    manifest = _manifest_with_key(tmp_path, _variant_library(tmp_path))
    fgs, *_ = load_project_config(manifest, grid_size=(5, 6), seed=1, mode="inference")
    assert abs(fgs["mackerel"].params["resting_metabolism"] - 2.295) < 1e-9
    base, *_ = load_project_config(PROJECT, grid_size=(5, 6), seed=1, mode="inference")
    assert base["mackerel"].params["resting_metabolism"] == 4.59


def test_gpu_builder_resolves_the_key(tmp_path):
    variant = _variant_library(tmp_path)
    manifest = _manifest_with_key(tmp_path, variant)
    assert EnvironmentBuilder(project_path=manifest).library_path == variant
    assert EnvironmentBuilder(project_path=PROJECT).library_path == DEFAULT_LIBRARY
    explicit = str(tmp_path / "explicit.yaml")
    assert EnvironmentBuilder(project_path=manifest, library_path=explicit).library_path == explicit
    assert load_config(EnvironmentBuilder(project_path=manifest).library_path)
