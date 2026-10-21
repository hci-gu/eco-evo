"""How many ARS delta pairs does GPU training need?

For each ``--n-deltas`` value in ``--sweep``, takes ``--repeats``
independent ARS steps from the SAME policy weights (restored between
steps, fresh deltas and fresh worlds each time, as training draws them)
and reports, per decision maker, the mean pairwise cosine similarity of
the resulting weight steps, plus seconds and peak VRAM per update.

Read the cosine against the DIMENSIONAL reference, not against 1. An
ARS step is a sum over N random directions in a d-dimensional weight
space, so even with noise-free rewards two independent steps agree only
to

    E[cos] ~= N / (N + d)            (d ~ 3500-3950 for a 2x48 net)

i.e. ~0.004 at N = 16. A measured cosine at that level says the step is
as good as random search can make it, NOT that the reward has no signal.
The signal question is answered by --control same-deltas: the same
deltas in fresh worlds, where the cosine measures how reproducible the
ranking of the deltas is across worlds (section 125).

The trainer is built from ``train_gpu.py``'s own parser, so every flag
after ``--`` means exactly what it means there:

    python3 tools/probes/ars_delta_budget.py --init-from results/RUN \\
        --sweep 4 8 16 32 64 --repeats 4 -- \\
        --project mareld2.yaml --migration on --mortality on \\
        --currents on --profile info --n_eval_ticks 500 \\
        --policynetwork 2 48 tanh --lr 0.03 --sigma 0.05
"""
import argparse
import itertools
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
import train_gpu  # noqa: E402
from lib.gpu.cli import builder_from_args, trainer_options  # noqa: E402
from lib.gpu.config import ProjectSpec  # noqa: E402
from lib.gpu.trainer import TensorARSTrainer  # noqa: E402
from lib.training_profiles import parse_training_args  # noqa: E402


def main():
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    own = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    own.add_argument('--init-from', default=None,
                     help='Directory with policy_<fg>.pth to measure at; '
                          'omitted = freshly initialised policies.')
    own.add_argument('--control', default='none',
                     choices=('none', 'same-all', 'same-worlds', 'same-deltas'),
                     help="What to hold fixed between the repeated steps. "
                          "'none' = training's own noise; 'same-all' is the "
                          "positive control (cos must be 1); 'same-worlds' "
                          "leaves only the delta draw, 'same-deltas' only "
                          "the world draw.")
    own.add_argument('--sweep', type=int, nargs='+', default=[4, 8, 16, 32, 64])
    own.add_argument('--repeats', type=int, default=4)
    a = own.parse_args(argv[:split])

    parser = train_gpu.build_parser()
    args = parse_training_args(parser, argv[split + 1:], destinations={
        "n_eval_ticks": "ticks", "rollouts_per_delta": "worlds"})
    options = trainer_options(args)
    spec = ProjectSpec(builder_from_args(args), seed=args.seed)
    ticks = args.ticks
    print(f"control {a.control}, policies {a.init_from or 'fresh init'}")
    print(f"ticks {ticks}, worlds {args.worlds}, sigma {args.sigma}, "
          f"lr {args.lr}, top_deltas {'N/2' if args.top_deltas is None else args.top_deltas}, "
          f"entropy {args.entropy_coef}, argmax {args.argmax_penalty}, "
          f"temperature {args.temp_end}", flush=True)

    rows = {}
    for n in a.sweep:
        opts = dict(options, n_deltas=n,
                    top_deltas=(None if args.top_deltas is None
                                else min(args.top_deltas, n)))
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        trainer = TensorARSTrainer(spec, device=args.device, **opts)
        trainer.set_temperature(args.temp_end)
        if a.init_from:
            trainer.import_policies(a.init_from)
        theta0 = [w.detach().clone() for w in trainer.theta]
        iteration0 = trainer.iteration.clone()
        norms, rstd = [], []
        obs0 = (trainer.obs_mean.clone(), trainer.obs_var.clone(),
                trainer.obs_count.clone())
        steps, seconds = [], []
        for r in range(a.repeats + 1):
            for w, w0 in zip(trainer.theta, theta0):
                w.copy_(w0)
            trainer.obs_mean.copy_(obs0[0])
            trainer.obs_var.copy_(obs0[1])
            trainer.obs_count.copy_(obs0[2])
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if a.control in ('same-all', 'same-deltas'):
                trainer.iteration.copy_(iteration0)
            epoch = {'none': None, 'same-all': None,
                     'same-worlds': 7919, 'same-deltas': 1000 + r}[a.control]
            metrics = trainer.train_step(None, ticks, world_epoch=epoch)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if r == 0:
                continue  # warmup / graph capture
            seconds.append(dt)
            steps.append([(w - w0).double().cpu().numpy()
                          for w, w0 in zip(trainer.theta, theta0)])
            norms.append([float(np.linalg.norm(x)) for x in steps[-1]])
            rstd.append(metrics['reward_std'].double().cpu().numpy())
        peak_gib = (torch.cuda.max_memory_allocated() / 2**30
                    if args.device.startswith('cuda') else float('nan'))
        trainer.close()
        del trainer
        cos = {}
        for d, fid in enumerate(spec.env.dm_ids):
            vals = []
            for i, j in itertools.combinations(range(len(steps)), 2):
                u, v = steps[i][d], steps[j][d]
                vals.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)))
            cos[fid] = (float(np.mean(vals)), float(np.std(vals)))
        rows[n] = (cos, float(np.median(seconds)))
        dims = {fid: steps[0][d].size for d, fid in enumerate(spec.env.dm_ids)}
        print(f"    |step| " + " ".join(f"{v:.2e}" for v in np.mean(norms, 0))
              + "   reward sd over deltas " + " ".join(
                  f"{v:.3g}" for v in np.mean(rstd, 0)), flush=True)
        print(f"N={n:>3}  s/update {rows[n][1]:7.2f}  "
              f"s/pair {rows[n][1] / n:6.3f}  peak {peak_gib:5.2f} GiB  " + "  ".join(
            f"{f} {c[0]:+.3f}" for f, c in cos.items()), flush=True)

    print("\nmean pairwise cosine of independent ARS steps (+- sd over pairs)")
    dm_ids = list(next(iter(rows.values()))[0])
    print(f"{'N':>4} {'s/upd':>7} " + " ".join(f"{f:>16}" for f in dm_ids))
    for n, (cos, sec) in rows.items():
        print(f"{n:>4} {sec:>7.2f} " + " ".join(
            f"{cos[f][0]:>+8.3f} +-{cos[f][1]:.3f}" for f in dm_ids))
    print("noise floor: two random directions in the same weight space "
          "have cos sd 1/sqrt(dim) = " + ", ".join(
              f"{f} {1 / np.sqrt(dims[f]):.3f} (dim {dims[f]})" for f in dm_ids))
    print("\nreference with noise-free rewards, E[cos] = N/(N+d):")
    for n in rows:
        print(f"  N={n:>3}  " + "  ".join(
            f"{f} {n / (n + dims[f]):.4f}" for f in dm_ids))


if __name__ == '__main__':
    main()
