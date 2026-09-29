"""The same ``seed`` must give the same rollout (section 124.4).

The tick draws from NumPy's global RNG - the NDM ``seed_rate``
recruitment noise above all - and ``inference.build_env`` used to seed
only the spawn. Two rollouts with one seed then drifted apart by a few
per cent over thousands of ticks. ``build_env(seed=...)`` now seeds the
global RNG as well.
"""
import numpy as np

from inference import build_env
from lib.diagnostics import viability


def _rollout(seed, ticks=40):
    env = build_env('mareld2.yaml', (12, 12), seed=seed, verbose=False,
                    apply_natural_mortality=True, tick_hours=6)
    assert any(float(getattr(fg, 'seed_rate', 0.0) or 0.0) > 0.0
               for fg in env.fgs.values()), \
        'fixture needs an NDM with seed_rate > 0 to exercise the noise'
    provider = viability.install_behaviour(env, 'eat', seed=seed)
    trace = []
    for _ in range(ticks):
        env.step(provider(env))
        trace.append(np.concatenate(
            [np.asarray(env.fgs[f].biomass, dtype=np.float64).ravel()
             for f in sorted(env.fgs)]))
    return np.array(trace)


def test_same_seed_gives_bit_identical_rollouts():
    first = _rollout(seed=5)
    # Anything else drawing from the global RNG in between must not leak
    # into the next world.
    np.random.rand(1000)
    second = _rollout(seed=5)
    assert np.array_equal(first, second)
