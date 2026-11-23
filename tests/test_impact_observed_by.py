"""Observable impacts seen by named decision makers only (section 300).

``observed_by`` on an observable ``impact_variables`` entry lists the
decision makers that observe the impact. The channel layout is the same
for every DM; a DM outside the list reads the impact's slots (own cell and
the four neighbours) as zero on both engines. Section 300's water-column
habitat term (``shallow_beds``) was its one use; section 300.1 removed the
term from mareld2.yaml and the library, and the mechanism stays.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import impact_observed_by
from lib.environments.ecosystem_env import impacts as impact_mod
from test_gpu_ecosystem import DEVICES, _assert_parity
from test_gpu_impacts import impact_env

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def _blind(env, fid, impact_ids):
    env.fgs[fid].params['unobserved_impacts'] = list(impact_ids)
    env.build_static_caches()
    return env


def _impact_slots(env, i):
    """Feature indices of DM i's impact slots (centre and neighbours)."""
    n_imp = len(env.observable_impact_vars)
    n_dim = int(env.per_dm_in_dim[i]) - (1 if env._has_daylight else 0)
    nbr = (n_dim - 1) // 5                 # center = nbr + 1, total = center + 4 nbr
    centre = nbr + 1
    slots = [centre - n_imp + k for k in range(n_imp)]
    for d in range(4):
        slots += [centre + d * nbr + nbr - n_imp + k for k in range(n_imp)]
    return slots, n_imp


@pytest.mark.parametrize("daylight", [False, True])
def test_unobserved_slots_read_zero_on_both_engines(device, daylight):
    calendar = ({"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 3,
                 "random_start": False} if daylight else None)
    env = _blind(impact_env(observable=["trawl", "noise"], daylight=calendar), "b", ["noise"])
    ia, ib = env.dm_ids.index("a"), env.dm_ids.index("b")
    assert env.dm_impact_obs[ia].tolist() == [1.0, 1.0]
    assert env.dm_impact_obs[ib].tolist() == [1.0, 0.0]
    feats = env.get_observation().features
    slots, n_imp = _impact_slots(env, ib)
    noise_slots = slots[1::n_imp]
    trawl_slots = slots[0::n_imp]
    assert not np.any(feats[ib][:, noise_slots]), "b does not observe noise"
    assert np.any(feats[ib][:, trawl_slots]), "b still observes trawl"
    np.testing.assert_array_equal(feats[ib][:, trawl_slots[0]], env.grid.get_map("trawl").ravel())
    a_slots, _ = _impact_slots(env, ia)
    np.testing.assert_array_equal(feats[ia][:, a_slots[1]], env.grid.get_map("noise").ravel())
    # Same layout as without the restriction.
    full = impact_env(observable=["trawl", "noise"], daylight=calendar)
    assert list(full.per_dm_in_dim) == list(env.per_dm_in_dim)
    # The GPU engine reads the same zeros (observations compared every tick).
    _assert_parity(env, device, ticks=4, base_tol=2e-6)


def _project(observed_by=None, observable=True, extra=None):
    iv = {"impact_id": "x", "observable": observable}
    if observed_by is not None:
        iv["observed_by"] = observed_by
    proj = {"decision_makers": [{"group_id": "fish"}, {"group_id": "seal"},
                                {"group_id": "gone", "muted": True}],
            "impact_variables": [iv] + (extra or [])}
    return proj


def test_observed_by_parsing_and_refusals():
    assert impact_observed_by(_project()) == {"x": None}
    assert impact_observed_by(_project(["fish"])) == {"x": {"fish"}}
    assert impact_observed_by(_project("seal")) == {"x": {"seal"}}
    with pytest.raises(ValueError, match="not active"):
        impact_observed_by(_project(["gone"]))
    with pytest.raises(ValueError, match="not active"):
        impact_observed_by(_project(["nobody"]))
    with pytest.raises(ValueError, match="empty"):
        impact_observed_by(_project([]))
    with pytest.raises(ValueError, match="observable"):
        impact_observed_by(_project(["fish"], observable=False))
    muted = _project(extra=[{"impact_id": "y", "muted": True, "observed_by": ["nobody"]}])
    assert "y" not in impact_observed_by(muted)

