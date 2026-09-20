"""Diagnose the vertical banding seen in a --local_reward GPU training run.

Replays the saved policies of results/<run> in a single fixed world and
measures, per decision maker:

  * aniso   var(column means) / var(row means) of the biomass field.
            1.0 = isotropic, >>1 = vertical bands, <<1 = horizontal ones.
  * nyq_x   share of the column profile's spectral power sitting in the
            highest spatial frequency (1-cell-wide stripes).
  * nyq_y   the same along y.
  * N/E/S/W the mean action probability per move direction.

A second pass with untrained (randomly initialised) policies is the
control: if the bands only appear after training, they are learned.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from lib.gpu.config import DEFAULT_LIBRARY, EnvironmentBuilder, ProjectSpec  # noqa: E402
from lib.gpu.random import fold_in  # noqa: E402
from lib.gpu.rollout import RolloutRunner  # noqa: E402
from lib.gpu.trainer import TensorARSTrainer  # noqa: E402


def profiles(field, H, W):
    grid = field.reshape(H, W)
    col, row = grid.mean(0), grid.mean(1)
    def nyquist(profile):
        spectrum = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
        return float(spectrum[-1] / max(spectrum.sum(), 1e-30))
    return (float(col.var() / max(row.var(), 1e-30)),
            nyquist(col), nyquist(row))


def rollout(trainer, runner, world, ticks, H, W):
    m = trainer.model
    runner.set_weights(trainer.bank.pack([w[None].clone() for w in trainer.theta]))
    runner.reset(*world, ticks, trainer.obs_mean.float(), trainer.obs_var.float(),
                 m.tensor(1.0))
    directions = np.zeros((m.D, 4))
    series = []
    counted = 0
    for tick in range(ticks):
        obs = m.observations(runner.biomass, runner.reserve, runner.hidden)
        if runner.normalize:
            obs = ((obs - runner.obs_mean[None, :, None]) /
                   runner.obs_var.clamp_min(1e-2).sqrt()[None, :, None]).clamp(-10, 10)
        logits = trainer.bank.forward(obs.reshape(1, m.D, m.C, m.F),
                                      runner.weights, runner.biases)
        actions = m.action_probabilities(logits.reshape(1, m.D, m.C, m.A),
                                         runner.biomass, m.tensor(1.0))
        live = (runner.biomass[:, m.dm_index] > 0).float()
        # actions is [E, D, A, C]: directions on axis 2, cells on axis 3.
        share = ((actions[:, :, :4] * live[:, :, None]).sum(-1)
                 / live.sum(-1).clamp_min(1)[:, :, None])
        directions += share[0].double().cpu().numpy()
        counted += 1
        runner.run(1)
        series.append(runner.biomass[0].double().cpu().numpy()
                      .reshape(-1, H, W).mean(1))
    biomass = runner.biomass[0].double().cpu().numpy()
    out = {}
    for d, fid in enumerate(m.dm_ids):
        g = m.dm_positions[d]
        aniso, nyq_x, nyq_y = profiles(biomass[g], H, W)
        out[fid] = dict(total=float(biomass[g].sum()),
                        occupied=int((biomass[g] > 0).sum()),
                        aniso=aniso, nyq_x=nyq_x, nyq_y=nyq_y,
                        moves=(directions[d] / counted).round(4).tolist(),
                        drift=drift(np.array(series)[:, g]))
    return out, biomass


def drift(profile):
    """Mean per-tick shift of the column profile, in cells.

    Cross-correlate consecutive column profiles over the last quarter of
    the rollout; a travelling wave train shows up as a consistent
    non-zero shift, a standing pattern as 0.
    """
    tail = profile[3 * len(profile) // 4:]
    W = tail.shape[1]
    lags = []
    for a, b in zip(tail[:-1], tail[1:]):
        a, b = a - a.mean(), b - b.mean()
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        correlation = [float((np.roll(a, k) * b).sum()) for k in range(-3, 4)]
        lags.append(range(-3, 4)[int(np.argmax(correlation))])
    return float(np.mean(lags)) if lags else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/pzbpgp248nsrpfix16")
    p.add_argument("--grid", default="16x16")
    p.add_argument("--ticks", type=int, default=700)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dump", default=None)
    args = p.parse_args()
    torch.set_num_threads(8)

    H, W = (int(v) for v in args.grid.lower().split("x"))
    builder = EnvironmentBuilder("mareld2.yaml", DEFAULT_LIBRARY,
                                 (H, W), True, False)
    spec = ProjectSpec(builder, seed=args.seed)
    trainer = TensorARSTrainer(spec, device=args.device, n_deltas=1, worlds=1,
                               seed=args.seed, hidden_dim=48, hidden_layers=2,
                               activation="tanh", execution="eager")
    m = trainer.model
    trainer.refresh_worlds(epoch=0)
    keys = fold_in(fold_in(torch.tensor([args.seed], device=m.device), 20001),
                   torch.tensor([0], device=m.device))
    keys = fold_in(keys, 0)
    biomass0 = trainer.world_biomass.clone()
    world = (biomass0, trainer.spawner.reserves(biomass0, keys),
             trainer.spawner.phases(keys), keys)
    runner = RolloutRunner(m, trainer.bank, 1, 1, execution="eager",
                           obs_normalize=True, local_reward=None)

    control, _ = rollout(trainer, runner, world, args.ticks, H, W)
    trainer.import_policies(args.run)
    trained, field = rollout(trainer, runner, world, args.ticks, H, W)

    print(f"{'fg':<14}{'arm':<9}{'total':>10}{'occ':>6}{'aniso':>9}"
          f"{'nyq_x':>8}{'nyq_y':>8}{'drift':>7}   N/E/S/W")
    for fid in m.dm_ids:
        for name, data in (("untrained", control), ("trained", trained)):
            d = data[fid]
            print(f"{fid:<14}{name:<9}{d['total']:>10.3f}{d['occupied']:>6}"
                  f"{d['aniso']:>9.2f}{d['nyq_x']:>8.2f}{d['nyq_y']:>8.2f}"
                  f"{d['drift']:>7.2f}   "
                  + " ".join(f"{v:.3f}" for v in d["moves"]))
    if args.dump:
        Path(args.dump).write_text(json.dumps(
            dict(untrained=control, trained=trained,
                 field={f: field[m.dm_positions[d]].reshape(H, W).tolist()
                        for d, f in enumerate(m.dm_ids)}), indent=1))


if __name__ == "__main__":
    main()
