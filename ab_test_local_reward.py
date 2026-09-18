"""T2: is the local reward better than the global one? (sections 79.8, 82.3)

Section 82 (T1) answered the diagnostic question: ``--local_reward_norm sum``
with ``theta=0`` optimises spatial extent, not local performance, and the
defaults (``log`` + ``mean``) are free of that gradient. What T1 could not
answer is whether the local reward is *better* than today's global one,
because the arms optimise different scalars and their reward curves are not
comparable.

T2 therefore trains the arms and then scores them on REWARD-INDEPENDENT
metrics in a fixed evaluation rollout that every arm shares: identical
worlds, identical tick count, local reward switched off.

Arms (all ``--local_reward_norm mean --local_reward_metric log``):

    theta0     w_c = 1            the local reward (the colleague's proposal)
    theta05    w_c = A_c**0.5     halfway
    theta1     w_c = A_c          = the global reward by construction (79.6)
    global     no flag at all     sanity: must land close to theta1

Usage:

    python3 ab_test_local_reward.py --stage train   # 12 train_gpu.py runs
    python3 ab_test_local_reward.py --stage eval    # scoring + table
    python3 ab_test_local_reward.py                 # both

Training is delegated to ``train_gpu.py`` as a subprocess, one per
(arm, seed), so each run starts from a clean process. Runs that already
carry a checkpoint are skipped, which makes the script restartable.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.gpu.config import DEFAULT_LIBRARY, EnvironmentBuilder, ProjectSpec  # noqa: E402
from lib.gpu.rollout import RolloutRunner  # noqa: E402
from lib.gpu.random import fold_in  # noqa: E402
from lib.gpu.trainer import TensorARSTrainer  # noqa: E402


ARMS = {
    "theta0": ["--local_reward", "--local_reward_norm", "mean",
               "--local_reward_metric", "log", "--local_reward_theta", "0"],
    "theta05": ["--local_reward", "--local_reward_norm", "mean",
                "--local_reward_metric", "log", "--local_reward_theta", "0.5"],
    "theta1": ["--local_reward", "--local_reward_norm", "mean",
               "--local_reward_metric", "log", "--local_reward_theta", "1"],
    "global": [],
}


def run_directory(root: Path, arm: str, seed: int) -> Path:
    return root / f"t2_{arm}_s{seed}"


def train(args) -> None:
    """One ``train_gpu.py`` subprocess per (arm, seed); skips finished runs."""
    for seed in args.seeds:
        for arm, flags in ARMS.items():
            directory = run_directory(args.results, arm, seed)
            if (directory / "trainer.pth").exists():
                print(f"[skip] {directory} already has a checkpoint", flush=True)
                continue
            command = [sys.executable, "train_gpu.py",
                       "--project", args.project,
                       "--grid", args.grid,
                       "--profile", "info",
                       "--device", args.device,
                       "--generations", str(args.generations),
                       "--ticks", str(args.ticks),
                       "--seed", str(seed),
                       "--output", str(directory)] + flags
            if args.device == "cpu":
                command += ["--execution", "eager"]
            print(f"[train] {arm} seed={seed} -> {directory}", flush=True)
            started = time.time()
            result = subprocess.run(command, stdin=subprocess.DEVNULL)
            if result.returncode != 0:
                raise SystemExit(f"training failed for {arm} seed={seed}")
            print(f"[train] {arm} seed={seed} done in "
                  f"{time.time() - started:.0f} s", flush=True)


class Evaluator:
    """Fixed, reward-independent rollout shared by every arm.

    The worlds come from ``WorldSpawner`` keyed on ``--eval-seed`` only, so
    all arms and seeds see bit-identical initial biomass, reserves and
    seasonal phases. ``local_reward=None`` on the runner means no arm is
    scored with its own fitness function.
    """

    def __init__(self, args):
        builder = EnvironmentBuilder(args.project, DEFAULT_LIBRARY,
                                     tuple(int(v) for v in args.grid.lower().split("x")))
        self.spec = ProjectSpec(builder, seed=args.eval_seed)
        self.args = args
        self.trainer = TensorARSTrainer(self.spec, device=args.device, n_deltas=1,
                                        worlds=args.eval_worlds, seed=args.eval_seed,
                                        execution="eager")
        self.model = self.trainer.model
        self.runner = RolloutRunner(self.model, self.trainer.bank, 1, args.eval_worlds,
                                    execution="eager", obs_normalize=True,
                                    local_reward=None)
        index = torch.arange(args.eval_worlds, device=self.model.device, dtype=torch.int64)
        keys = fold_in(torch.full_like(index, args.eval_seed), 90001)
        self.keys = fold_in(keys, index)
        self.trainer.refresh_worlds(epoch=0)
        self.biomass0 = self.trainer.world_biomass.clone()
        self.reserve0 = self.trainer.spawner.reserves(self.biomass0, self.keys)
        self.phase0 = self.trainer.spawner.phases(self.keys)
        # Occupancy uses the same viability bar as the extinction sweep,
        # so "occupied" means the same thing here as inside the tick.
        self.occupied_bar = self.model.threshold.clamp_min(1e-9)

    def __call__(self, directory: Path) -> dict:
        self.trainer.import_policies(directory)
        self.runner.set_weights(self.trainer.bank.pack(
            [w[None].clone() for w in self.trainer.theta]))
        self.runner.reset(self.biomass0, self.reserve0, self.phase0, self.keys,
                          self.args.eval_ticks, self.trainer.obs_mean.float(),
                          self.trainer.obs_var.float(), self.model.tensor(1.0))
        biomass, occupancy, entropy = [], [], []
        for _ in range(self.args.eval_ticks):
            self.runner.run(1)
            b = self.runner.biomass
            biomass.append(b.sum(-1).double().cpu().numpy())
            occupied = (b > self.occupied_bar).double().sum(-1)
            occupancy.append(occupied.cpu().numpy())
            share = b.double() / b.double().sum(-1, keepdim=True).clamp_min(1e-30)
            shannon = -(share * share.clamp_min(1e-300).log()).sum(-1)
            entropy.append((shannon / math.log(self.model.C)).cpu().numpy())
        return self.summarise(np.array(biomass), np.array(occupancy), np.array(entropy))

    def summarise(self, biomass, occupancy, entropy) -> dict:
        """``[tick, world, group]`` series -> one number per group.

        Averaged over evaluation worlds. ``swing`` is the coefficient of
        variation over the second half of the rollout, i.e. the boom-bust
        amplitude of POPULATION_STABILITY.md measured after the transient.
        """
        start = self.biomass0.sum(-1).double().cpu().numpy()
        half = biomass[biomass.shape[0] // 2:]
        mean = half.mean(0)
        swing = half.std(0) / np.maximum(mean, 1e-30)
        output = {}
        for g, fid in enumerate(self.model.ids):
            final = biomass[-1, :, g]
            output[fid] = dict(
                start_biomass=float(start[:, g].mean()),
                final_biomass=float(final.mean()),
                mean_biomass=float(mean[:, g].mean()),
                min_biomass=float(biomass[:, :, g].min(0).mean()),
                occupancy_start=float(occupancy[0, :, g].mean()),
                occupancy_final=float(occupancy[-1, :, g].mean()),
                entropy_final=float(entropy[-1, :, g].mean()),
                swing=float(swing[:, g].mean()),
                # A group counts as extinct in a world when it ends below a
                # thousandth of what it started with; averaged over worlds
                # this is the fraction of worlds that lost it.
                extinct=float((final < 1e-3 * start[:, g]).mean()),
            )
        return output


KEYS = ("final_biomass", "mean_biomass", "min_biomass", "occupancy_final",
        "entropy_final", "swing", "extinct")


def aggregate(records: dict) -> dict:
    """Mean and standard deviation over seeds, per arm, group and metric."""
    output = {}
    for arm in ARMS:
        seeds = [r for (a, _), r in records.items() if a == arm]
        if not seeds:
            continue
        groups = {}
        for fid in seeds[0]:
            groups[fid] = {k: (float(np.mean([s[fid][k] for s in seeds])),
                               float(np.std([s[fid][k] for s in seeds])))
                           for k in KEYS}
        output[arm] = groups
    return output


def report(summary: dict, groups) -> None:
    for fid in groups:
        print(f"\n=== {fid} ===")
        header = f"{'arm':<9}" + "".join(f"{k:>18}" for k in KEYS)
        print(header)
        for arm, data in summary.items():
            row = f"{arm:<9}"
            for k in KEYS:
                mean, deviation = data[fid][k]
                row += f"{mean:>11.4g}+-{deviation:<5.3g}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("train", "eval", "both"), default="both")
    parser.add_argument("--project", default="mareld2.yaml")
    parser.add_argument("--grid", default="24x24")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--eval-seed", "--eval_seed", dest="eval_seed", type=int, default=777)
    parser.add_argument("--eval-worlds", "--eval_worlds", dest="eval_worlds", type=int, default=4)
    parser.add_argument("--eval-ticks", "--eval_ticks", dest="eval_ticks", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results/t2_local_reward.json"))
    args = parser.parse_args()

    if args.stage in ("train", "both"):
        train(args)
    if args.stage not in ("eval", "both"):
        return
    torch.set_num_threads(1)
    evaluator = Evaluator(args)
    records, missing = {}, []
    for arm in ARMS:
        for seed in args.seeds:
            directory = run_directory(args.results, arm, seed)
            if not any(directory.glob("policy_*.pth")):
                missing.append(str(directory))
                continue
            print(f"[eval] {arm} seed={seed}", flush=True)
            records[(arm, seed)] = evaluator(directory)
    if missing:
        print("[eval] no policies in: " + ", ".join(missing), flush=True)
    if not records:
        raise SystemExit("nothing to evaluate; run --stage train first")
    summary = aggregate(records)
    payload = dict(setup=vars(args) | {"results": str(args.results),
                                       "output": str(args.output)},
                   groups=evaluator.model.ids,
                   decision_makers=evaluator.model.dm_ids,
                   runs={f"{a}_s{s}": r for (a, s), r in records.items()},
                   summary=summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    report(summary, evaluator.model.dm_ids)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
