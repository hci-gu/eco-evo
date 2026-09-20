"""Compare --local_reward arms on the training log alone.

For every run given on the command line this prints, per decision maker
and over the first ``--updates`` updates (so arms of unequal length stay
comparable):

  corr     Pearson correlation between local_occupancy and reward.
           Strongly negative means the reward rises as cells are lost,
           i.e. the survivorship bias of ``--local_reward_norm mean``.
  occ      mean occupied cells over the first and the last fifth.
  reward   the same for the reward.
  eat/rest action shares over the last fifth.
  biomass  mean biomass over the first and the last fifth.
"""

import argparse
import json
import os

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def load(run, limit):
    rows = [json.loads(line) for line in
            open(f"results/{run}/training.jsonl", encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    config = json.load(open(f"results/{run}/gpu_run.json", encoding="utf-8"))
    return rows, config


def ends(values):
    fifth = max(len(values) // 5, 1)
    return float(np.mean(values[:fifth])), float(np.mean(values[-fifth:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--updates", type=int, default=0)
    args = parser.parse_args()

    print(f"{'run':<20}{'fg':<14}{'n':>5}{'corr':>7}"
          f"{'occ0':>8}{'occ1':>8}{'rew0':>10}{'rew1':>10}"
          f"{'eat':>7}{'rest':>7}{'move':>7}{'bio0':>10}{'bio1':>10}")
    for run in args.runs:
        rows, config = load(run, args.updates)
        dms = config["decision_makers"]
        groups = config["groups"]
        label = f"{run}[{config['local_reward_norm']}" \
                f",th{config['local_reward_theta']:g}]" if config["local_reward"] \
            else f"{run}[global]"
        for d, fg in enumerate(dms):
            occupancy = np.array([r["local_occupancy"][d] for r in rows]) \
                if rows[0].get("local_occupancy") else np.zeros(len(rows))
            reward = np.array([r["reward_mean"][d] for r in rows])
            actions = np.array([r["actions"][d] for r in rows])
            biomass = np.array([r["mean_biomass"][groups.index(fg)] for r in rows])
            if occupancy.std() > 0 and reward.std() > 0:
                corr = float(np.corrcoef(occupancy, reward)[0, 1])
            else:
                corr = float("nan")
            occ0, occ1 = ends(occupancy)
            rew0, rew1 = ends(reward)
            bio0, bio1 = ends(biomass)
            fifth = max(len(rows) // 5, 1)
            _, move, rest, eat = actions[-fifth:].mean(0)
            print(f"{label[:19]:<20}{fg:<14}{len(rows):>5}{corr:>7.2f}"
                  f"{occ0:>8.1f}{occ1:>8.1f}{rew0:>10.5f}{rew1:>10.5f}"
                  f"{eat:>7.2f}{rest:>7.2f}{move:>7.2f}{bio0:>10.1f}{bio1:>10.1f}")
        print()


if __name__ == "__main__":
    main()
