"""A decision maker with movement_speed 0 cannot choose to move (section 232.4).

Contracts asserted:

1. The project's zooplankton has movement_speed 0 (it drifts with the
   currents only).
2. CPU: its four move actions are masked, so the action distribution puts
   no mass on them while the rest and eat actions stay available; another
   DM keeps its move actions.
3. GPU engine: TensorEcosystem.action_mask masks the same actions.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.environments.ecosystem_env import decisions  # noqa: E402
from lib.environments.ecosystem_env.constants import EAT_START, MOVE_SLICE, REST_INDEX  # noqa: E402
from lib.gpu.config import EnvironmentBuilder  # noqa: E402
from lib.gpu.ecosystem import TensorEcosystem  # noqa: E402
from test_gpu_ecosystem import DEVICES  # noqa: E402

IMMOBILE = "zooplankton"
MOBILE = "planktivorous_fish"


@pytest.fixture(scope="module")
def env():
    return EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6))(seed=3)


def test_project_zooplankton_has_no_speed(env):
    assert env.fgs[IMMOBILE].speed == 0.0
    assert env.dm_v[env.dm_ids.index(IMMOBILE)] == 0.0


def test_cpu_masks_the_move_actions(env):
    n_actions = EAT_START + env.N_all
    mask = decisions.build_action_mask(env, n_actions)
    i, j = env.dm_ids.index(IMMOBILE), env.dm_ids.index(MOBILE)
    assert not mask[i, MOVE_SLICE].any()
    assert mask[i, REST_INDEX].all()
    assert mask[i, EAT_START:].any()
    assert mask[j, MOVE_SLICE].any()
    logits = np.zeros((env.N_dm, env.H * env.W, n_actions), dtype=env.dtype)
    probs = decisions.masked_softmax(env, logits, mask)   # (N_dm, A, H, W)
    assert float(probs[i, MOVE_SLICE].sum()) == 0.0
    assert float(probs[j, MOVE_SLICE].sum()) > 0.0

@pytest.mark.parametrize("device", DEVICES)
def test_gpu_masks_the_move_actions(env, device):
    model = TensorEcosystem(env, device)
    biomass = torch.stack([torch.as_tensor(env.fgs[f].biomass.reshape(-1))
                           for f in env.global_fg_order]).to(model.device)[None]
    mask = model.action_mask(biomass)
    i, j = env.dm_ids.index(IMMOBILE), env.dm_ids.index(MOBILE)
    assert not mask[0, i, :4].any()
    assert mask[0, i, 4].all()
    assert mask[0, j, :4].any()
