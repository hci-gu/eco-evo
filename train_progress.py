"""Wrap either trainer with a headless biomass-survival progress graph.

Put wrapper options before -- and the usual trainer options after it.
"""

import argparse
import math
import sys
from pathlib import Path

from lib.gpu.cli import positive_int
from lib.runners.training_progress import TrainingProgress


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--eval-every", type=positive_int, default=20,
                        help="Evaluate every N completed ARS updates (default: 20)")
    parser.add_argument("--eval-ticks", type=positive_int, default=1000,
                        help="Maximum inference ticks per evaluation (default: 1000)")
    parser.add_argument("--biomass-bounds", type=float, nargs=2, default=(0.3, 3.0),
                        metavar=("LOWER", "UPPER"), help="Inclusive multiples of initial biomass")
    parser.add_argument("--eval-seed", type=int, default=20260530,
                        help="Fixed inference world and runtime-noise seed")
    parser.add_argument("--eval-temperature", type=float, default=1.0,
                        help="Fixed inference softmax temperature (default: 1)")
    parser.add_argument("--plot-dir", type=Path, help="Default: <training run directory>/progress")
    if "--" in argv:
        split = argv.index("--")
        own_args, training_args = argv[:split], argv[split + 1:]
    else:
        own_args, training_args = argv, []
    args = parser.parse_args(own_args)
    lower, upper = args.biomass_bounds
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= 1 <= upper and lower < upper):
        parser.error("--biomass-bounds must be finite and satisfy 0 < LOWER <= 1 <= UPPER, LOWER < UPPER")
    if not 0 <= args.eval_seed < 2**32:
        parser.error("--eval-seed must be between 0 and 2**32 - 1")
    if not math.isfinite(args.eval_temperature) or args.eval_temperature <= 0:
        parser.error("--eval-temperature must be positive and finite")
    monitor = TrainingProgress(args.backend, args.eval_every, args.eval_ticks,
                               lower, upper, args.eval_seed, args.eval_temperature, args.plot_dir)
    if args.backend == "gpu":
        from train_gpu import main as train_main
        return train_main(training_args, on_step=monitor)
    from train import main as train_main
    return train_main(training_args, on_step=monitor, confirm=False)


if __name__ == "__main__":
    raise SystemExit(main())
