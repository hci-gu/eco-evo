"""Benthivore biomass share by bed-depth class (< 100 / 100-150 / 150-250 / >= 250 m) per probe year,
under a run's frozen policies in its long-run probe world. RUN, LIB, YEARS from the environment."""
import json
import os
import sys

import numpy as np
import yaml

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from inference import load_policies_and_stats  # noqa: E402
from lib.runners import training_progress as tp  # noqa: E402
from lib.world import habitat  # noqa: E402

RUN, LIB = os.environ['RUN'], os.environ['LIB']
YEARS = int(os.environ.get('YEARS', 3))
G = 'benthivorous_fish'
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
    project = yaml.safe_load(open(config['project']))
    grid = tuple(config['grid'])
    depth = np.asarray(habitat.load_depth_map(project, config["project"], grid),
                       dtype=np.float64)
    edges = [0, 100, 150, 250, 1e9]
    cls = np.digitize(depth, edges[1:-1])
    cell_share = [float((cls == k).mean()) for k in range(4)]
    print('cell share by class', np.round(cell_share, 3))
    ty = 1460
    acc = np.zeros((YEARS, 4))
    krill_acc = np.zeros(YEARS)
    prev = dict(env.intake_by_pred_prey.get(G, {})).get('krill', 0.0)
    for t in range(YEARS * ty):
        env.step()
        b = np.asarray(env.fgs[G].biomass, dtype=np.float64).reshape(depth.shape)
        y = t // ty
        for k in range(4):
            acc[y, k] += b[cls == k].sum()
    for y in range(YEARS):
        s = acc[y] / acc[y].sum()
        print(f"year {y + 1}: biomass share < 100 / 100-150 / 150-250 / >= 250 m: "
              + ' '.join(f'{v:.3f}' for v in s)
              + f"   (cells {' '.join(f'{v:.3f}' for v in cell_share)})")
