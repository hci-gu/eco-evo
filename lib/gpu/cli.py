"""Shared, validated options for headless training and comparative benchmarks."""

import argparse
import math
import re

from lib.gpu.config import DEFAULT_LIBRARY, EnvironmentBuilder


def grid_size(value):
    match = re.fullmatch(r"\s*(\d+)\s*[*xX]\s*(\d+)\s*", value)
    if not match or min(map(int, match.groups())) < 3:
        raise argparse.ArgumentTypeError("Grid must be HxW with both dimensions at least 3")
    return tuple(map(int, match.groups()))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return value


def add_common_arguments(parser):
    parser.add_argument("--project", default=None, help="Project YAML; omitted means all library groups")
    parser.add_argument("--library", default=DEFAULT_LIBRARY)
    parser.add_argument("--grid", type=grid_size, default=(60, 60))
    parser.add_argument("--n-deltas", "--n_deltas", dest="n_deltas", type=positive_int, default=10)
    parser.add_argument("--top-deltas", "--top_deltas", dest="top_deltas", type=positive_int)
    parser.add_argument("--worlds", "--rollouts_per_delta", dest="worlds", type=positive_int, default=1)
    parser.add_argument("--ticks", "--n_eval_ticks", dest="ticks", type=positive_int, default=15)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--species", nargs="+", default=["all"])
    parser.add_argument("--no-coevolution", "--no_coevolution", dest="coevolution", action="store_false", default=True)
    parser.add_argument("--coevolution", dest="coevolution", action="store_true")
    parser.add_argument("--policynetwork", nargs="+", default=["2", "30", "sig"], metavar="ARCH")
    parser.add_argument("--uniform-bias-init", "--uniform_bias_init", dest="uniform_bias_init", action="store_true")
    parser.add_argument("--no-obs-normalize", "--no_obs_normalize", dest="obs_normalize", action="store_false", default=True)
    parser.add_argument("--no-integral-reward", "--no_integral_reward", dest="integral_reward", action="store_false", default=True)
    parser.add_argument("--legacy-reward", "--legacy_reward", "--legacyreward", dest="legacy_reward", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--cappa", type=float, default=1.0)
    parser.add_argument("--survival-threshold", "--survival_threshold", dest="survival_threshold", type=float, default=0.01)
    parser.add_argument("--entropy-coef", "--entropy_coef", dest="entropy_coef", type=float, default=0.1)
    parser.add_argument("--argmax-penalty", "--argmax_penalty", dest="argmax_penalty", type=float, default=0.3)
    parser.add_argument("--mortality", choices=("on", "off"), default="off")
    parser.add_argument("--migration", choices=("on", "off"), default="off")
    parser.add_argument("--execution", choices=("eager", "compile", "cuda-graph", "compile-graph"), default="cuda-graph")
    parser.add_argument("--graph-ticks", type=positive_int, default=32)
    parser.add_argument("--pairs-per-batch", type=positive_int, help="Limit simultaneous perturbation pairs to fit VRAM")


def builder_from_args(args):
    return EnvironmentBuilder(args.project, args.library, args.grid,
                              args.migration == "on", args.mortality == "on")


def trainer_options(args):
    if len(args.policynetwork) not in (2, 3):
        raise ValueError("--policynetwork expects LAYERS NODES [sig|relu|tanh]")
    hidden_layers, hidden_dim = map(int, args.policynetwork[:2])
    activation = args.policynetwork[2] if len(args.policynetwork) == 3 else "sig"
    if min(hidden_layers, hidden_dim) < 1 or activation not in ("sig", "sigmoid", "relu", "tanh"):
        raise ValueError("Policy dimensions must be positive and activation must be sig, relu, or tanh")
    if args.sigma <= 0 or args.lr < 0 or not math.isfinite(args.sigma) or not math.isfinite(args.lr):
        raise ValueError("sigma must be positive and lr nonnegative, both finite")
    if args.top_deltas is not None and args.top_deltas > args.n_deltas:
        raise ValueError("top_deltas cannot exceed n_deltas")
    return dict(n_deltas=args.n_deltas, worlds=args.worlds, top_deltas=args.top_deltas,
                sigma=args.sigma, lr=args.lr, seed=args.seed, hidden_dim=hidden_dim,
                hidden_layers=hidden_layers, activation=activation,
                uniform_bias_init=args.uniform_bias_init, obs_normalize=args.obs_normalize,
                integral_reward=args.integral_reward, legacy_reward=args.legacy_reward,
                alpha=args.alpha, beta=args.beta, survival_bonus=args.cappa,
                survival_threshold=args.survival_threshold,
                entropy_coef=args.entropy_coef, argmax_penalty=args.argmax_penalty,
                execution=args.execution, graph_ticks=args.graph_ticks,
                pairs_per_batch=args.pairs_per_batch)


def targets_from_args(args, dm_ids):
    targets = tuple(dm_ids) if args.species == ["all"] else tuple(args.species)
    if not targets or len(set(targets)) != len(targets) or any(f not in dm_ids for f in targets):
        raise ValueError("--species must name active decision makers: " + ", ".join(dm_ids))
    return targets
