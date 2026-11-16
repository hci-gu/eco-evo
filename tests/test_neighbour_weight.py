"""Section 273: the observation pair's neighbour weight w.

Each neighbour input of an observed group shows c + w (n - c) on in-grid
cells (c the group's value in the observer's own cell, n in the neighbour)
and 0 outside the grid; the own biomass and the own juvenile channel take
the self pair's w, another group's juvenile channel that group's w. w = 1
(or no key) is today's observation, bit for bit. The tensor engine
matches the reference engine.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.config import config_loader  # noqa: E402
from lib.environments.ecosystem_env import observations  # noqa: E402
from lib.environments.ecosystem_env.constants import DIRECTIONS  # noqa: E402
from test_edge_closure import DAYLIGHT, STAGES  # noqa: E402
from test_gpu_ecosystem import _assert_parity, device, make_env  # noqa: E402,F401

PROJECT = ROOT / "mareld2.yaml"
W_A = {"a": 0.3, "b": 0.0, "c": 0.6}
W_B = {"a": 0.1, "b": 0.45}


def _env(weights=None, **kw):
    regen = {fid: {"neighbour_weight": dict(w)} for fid, w in (weights or {}).items()}
    kw.setdefault("min_split", 0.0)
    kw.setdefault("extinction_factor", 0.0)
    env = make_env(regen=regen, **kw)
    env.build_static_caches()
    return env


def _batch(env):
    return observations.build_observation_batch(env).reshape(
        env.N_dm, env.H, env.W, -1)


def test_unit_weights_are_todays_observation():
    plain = _env()
    ones = _env({"a": {"a": 1.0, "b": 1.0, "c": 1.0}, "b": {"a": 1.0}})
    assert not plain._has_nbr_w and not ones._has_nbr_w
    np.testing.assert_array_equal(_batch(plain), _batch(ones))


def test_blend_towards_the_own_cell():
    plain = _env()
    weighted = _env({"a": W_A, "b": W_B})
    assert weighted._has_nbr_w
    base, out = _batch(plain), _batch(weighted)
    H, W = plain.H, plain.W
    inside = {d: observations.shift_field(np.ones((H, W), np.float32), d)
              for d in DIRECTIONS}
    for i, fid in enumerate(plain.dm_ids):
        others = [plain.global_fg_order[int(j)] for j in plain.obs_others_idx[i]]
        names = [fid] + others
        weights = {"a": W_A, "b": W_B}[fid]
        n_other = len(others)
        centre = [0] + [2 + k for k in range(n_other)]
        centre_dim, nbr_dim = 2 + n_other, 1 + n_other
        # The own cell is untouched.
        np.testing.assert_array_equal(out[i, :, :, :centre_dim], base[i, :, :, :centre_dim])
        for q, d in enumerate(DIRECTIONS):
            for k, name in enumerate(names):
                slot = centre_dim + q * nbr_dim + k
                w = weights.get(name, 1.0)
                c, n = base[i, :, :, centre[k]], base[i, :, :, slot]
                want = n if w == 1.0 else inside[d] * (c + np.float32(w) * (n - c))
                np.testing.assert_allclose(out[i, :, :, slot], want, rtol=1e-6, atol=1e-7)
                if w == 0.0:
                    # Nothing beyond the own cell; 0 off the grid as before.
                    np.testing.assert_allclose(out[i, :, :, slot], inside[d] * c, rtol=1e-6)


def test_juvenile_channels_take_the_group_weight():
    env = _env({"a": W_A, "b": W_B}, stages={"a": dict(STAGES)}, daylight=dict(DAYLIGHT))
    i_a, i_b = env.dm_ids.index("a"), env.dm_ids.index("b")
    # a observes b and c, then its own J; b observes a, then a's J.
    assert env.obs_nbr_w[i_a].tolist() == pytest.approx([0.3, 0.0, 0.6, 0.3])
    assert env.obs_nbr_w[i_b].tolist() == pytest.approx([0.45, 0.1, 0.1])


@pytest.mark.parametrize("staged", [False, True])
def test_tensor_engine_matches_reference(device, staged):
    kw = dict(stages={"a": dict(STAGES)}, daylight=dict(DAYLIGHT)) if staged else {}
    env = _env({"a": W_A, "b": W_B}, migration=True, **kw)
    model = _assert_parity(env, device)
    assert model.nbr_blend


def _library():
    return config_loader.load_config(ROOT / "fgconfig" / "fg_library.yaml")


def test_loader_reads_and_checks_the_key():
    lib = _library()
    inter = lib["interaction_definitions"]
    # The library carries the section 275 encounter components, which
    # exclude the weight: the legacy key is checked on a library without them.
    for entry in inter.values():
        for k in config_loader.ENCOUNTER_KEYS:
            (entry or {}).pop(k, None)
    key = "planktivorous_fish_observes_porpoises"
    inter[key] = dict(inter.get(key, {}), observes=True, neighbour_weight=0.25)
    result = config_loader.load_project_config(
        str(PROJECT), grid_size=(8, 8), seed=1, spawn_seed=1,
        library_config=copy.deepcopy(lib))
    fgs = result[0] if isinstance(result, tuple) else result
    assert fgs["planktivorous_fish"].params["neighbour_weight"]["porpoises"] == 0.25
    inter[key]["neighbour_weight"] = 1.5
    with pytest.raises(ValueError, match="neighbour_weight"):
        config_loader.load_project_config(
            str(PROJECT), grid_size=(8, 8), seed=1, spawn_seed=1, library_config=lib)
