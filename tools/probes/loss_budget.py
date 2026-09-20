"""Loss budget and action mix of saved policies, on the reference engine.

The training log has neither, so the policies are replayed in the NumPy
``EcosystemEnvironment`` (which books ``loss_starvation`` /
``loss_predation``) in one fixed world per run, reward-independent.

Predicted by section 84.6: with ``--local_reward_norm grid`` the
starvation share of the loss budget should fall back from the 88-100
percent of the ``mean`` arm towards the ``theta=1`` level, and the eat
share should rise.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import train as train_mod  # noqa: E402
from inference import load_policies_and_stats  # noqa: E402


def replay(run, ticks, seed):
    config = json.load(open(f"results/{run}/gpu_run.json", encoding="utf-8"))
    H, W = config["grid"]
    env = train_mod._ProbeEnvBuilder(
        project_path="mareld2.yaml", grid_size=(H, W),
        apply_natural_mortality=config["mortality"] == "on",
        migration=config["migration"] == "on")()
    policies, mean, var = load_policies_and_stats(
        env, os.path.join("results", run), verbose=False)
    env.policies, env.obs_mean, env.obs_var = policies, mean, var

    np.random.seed(seed)
    torch.manual_seed(seed)
    mix = np.zeros((len(env.dm_ids), 3))
    counted = 0
    for _ in range(ticks):
        env.step()
        for d, fid in enumerate(env.dm_ids):
            occupied = env.fgs[fid].biomass > 0
            if not occupied.any():
                continue
            move = float(env.pi_move[d][:, occupied].sum(0).mean())
            rest = float(env.pi_rest[d][occupied].mean())
            mix[d] += (move, rest, max(0.0, 1.0 - move - rest))
        counted += 1
    return env, mix / max(counted, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    print(f"{'run':<22}{'fg':<14}{'biomass':>10}{'occ':>6}"
          f"{'loss':>10}{'starve%':>9}{'pred%':>8}"
          f"{'eat':>7}{'rest':>7}{'move':>7}")
    for run in args.runs:
        env, mix = replay(run, args.ticks, args.seed)
        norm = json.load(open(f"results/{run}/gpu_run.json", encoding="utf-8"))
        label = f"{run.replace('pzbpgp248nsrp', '')}" \
                f"[{norm.get('local_reward_norm')},th{norm.get('local_reward_theta')}]"
        for d, fid in enumerate(env.dm_ids):
            biomass = env.fgs[fid].biomass
            starve = float(env.loss_starvation.get(fid, 0.0))
            predation = float(env.loss_predation.get(fid, 0.0))
            total = starve + predation + float(env.loss_impact.get(fid, 0.0))
            share = 100.0 / total if total > 0 else 0.0
            print(f"{label[:21]:<22}{fid:<14}{float(biomass.sum()):>10.2f}"
                  f"{int((biomass > 0).sum()):>6}{total:>10.1f}"
                  f"{starve * share:>9.0f}{predation * share:>8.0f}"
                  f"{mix[d][2]:>7.2f}{mix[d][1]:>7.2f}{mix[d][0]:>7.2f}")
        print()


if __name__ == "__main__":
    main()
