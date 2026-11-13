"""Run porp99b.py with the planktivores blind to porpoises (raw observation 0, own cell and
the four neighbours); BLIND=0 runs the base arm through the same wrapper."""
import os, runpy, sys
import numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.environments.ecosystem_env import observations, impacts
SCRIPT = '/home/simon/eco-evo/research_notes/Tumlarna i fix99/scripts/porp99b.py'
_orig = observations.build_observation_batch
state = {}

def blind(env):
    out = _orig(env)
    if os.environ.get('BLIND', '1') != '1':
        return out
    if 'idx' not in state:
        d = env.dm_ids.index('planktivorous_fish')
        others = [int(x) for x in env.obs_others_idx[d]]
        jv = getattr(env, 'obs_juvenile_idx', None)
        chan = others + ([int(x) for x in jv[d]] if jv is not None else [])
        ip = env.global_fg_order.index('porpoises')
        pos = [k for k, c in enumerate(chan) if c == ip]
        n_other, n_imp = len(chan), len(impacts.observable_impact_layers(env))
        center, nd = 2 + n_other + n_imp, 1 + n_other + n_imp
        state['d'] = d
        state['idx'] = [2 + k for k in pos] + [center + q * nd + 1 + k for q in range(4) for k in pos]
        print('planktivore channels', [env.global_fg_order[c][:6] for c in chan],
              'porpoise slots', state['idx'], flush=True)
    out[state['d']][:, state['idx']] = 0.0
    return out

observations.build_observation_batch = blind
runpy.run_path(SCRIPT, run_name='__main__')
