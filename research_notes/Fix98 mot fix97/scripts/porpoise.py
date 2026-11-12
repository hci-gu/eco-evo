"""Porpoise trajectory under frozen fix98 policies: biomass, fill, actions, losses, intake."""
import json
import os
import sys

import numpy as np

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from inference import load_policies_and_stats  # noqa: E402
from lib.environments.ecosystem_env import movement  # noqa: E402
from lib.runners import training_progress as tp  # noqa: E402
from lib.world import daylight  # noqa: E402

RUN = os.environ['RUN']
LIB = os.environ['LIB']
TICKS = int(os.environ.get('TICKS', 1460))
G = 'porpoises'

config = json.load(open(os.path.join(RUN, 'progress', 'config.json')))
config['library'] = LIB
seed = int(config['seed'])
with tp.evaluation_randomness(seed):
    env = tp.build_inference_env(config, seed)
with tp.evaluation_randomness(seed):
    policies, mean, var = load_policies_and_stats(env, RUN, verbose=False)
    env.policies = policies
    env.rebuild_batched_weights()
    env.obs_mean, env.obs_var = mean, var
    env.softmax_temperature = float(config.get('temperature', 1.0))
    i = env.dm_ids.index(G)
    sh = {}
    _costs = movement.apply_energy_costs

    def hook(e, actions):
        b = np.asarray(e.fgs[G].biomass, dtype=np.float64)
        rest = float((b * actions.rest[i]).sum())
        eat = actions.eat[i]
        eat_by_prey = (b[None] * eat).reshape(eat.shape[0], -1).sum(axis=1)
        move = float((b * actions.move[i].sum(axis=0)).sum())
        tot = max(rest + float(eat_by_prey.sum()) + move, 1e-30)
        sh['rest'], sh['move'] = rest / tot, move / tot
        sh['eat'] = eat_by_prey / tot
        sh['cells'] = int((b > 1e-6).sum())
        return _costs(e, actions)

    movement.apply_energy_costs = hook
    fg = env.fgs[G]
    print('ME', fg.max_energy_reserve, 'u', getattr(fg, 'maintenance_level', None),
          'min_split', getattr(fg, 'min_split_biomass', None))
    prev_int = {}
    for t in range(TICKS):
        env.step()
        if t % 20 == 0 or t < 10:
            B = float(np.asarray(fg.biomass, dtype=np.float64).sum())
            R = float(np.asarray(fg.energy_reserve, dtype=np.float64).sum())
            loss = {k: round(float((getattr(env, k, None) or {}).get(G, 0.0)), 4)
                    for k in ('loss_starvation', 'loss_predation', 'loss_impact', 'loss_natural')}
            intake = dict(env.intake_by_pred_prey.get(G, {}))
            d_int = {p: round(v - prev_int.get(p, 0.0), 4) for p, v in intake.items()}
            prev_int = intake
            m = daylight.calendar_at(env.daylight, t)['month_name']
            print(f"t{t:5d} {m:4s} B {B:7.4f} fill {R / max(B * fg.max_energy_reserve, 1e-30):5.3f} "
                  f"cells {sh.get('cells')} rest {sh.get('rest', 0):.2f} move {sh.get('move', 0):.2f} "
                  f"eat {np.round(sh.get('eat', []), 2).tolist()} loss {loss} intake20 {d_int}", flush=True)
            if B < 1e-4:
                break
