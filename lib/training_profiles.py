"""Shared CPU/GPU training recipes and explicit CLI override handling."""

import argparse
import copy


# Fixed temperature and no action-distribution modifiers: let the biological
# rules drive behavior. Keys use the original CPU CLI destinations.
PROFILES = {
    "sanity": {
        "coevolution": True,
        "generations": "10",
        "iter_per_gen": 15,
        "n_deltas": 16,
        "n_eval_ticks": 100,
        "temp_anneal_gens": 8,
        "argmax_penalty": 0.0,
        "entropy_coef": 0.0,
        "temp_start": 1.0,
        "temp_end": 1.0,
        "uniform_bias_init": False,
        "rollouts_per_delta": 3,
    },
    "info": {
        "coevolution": True,
        "generations": "10",
        "iter_per_gen": 20,
        "n_deltas": 16,
        "n_eval_ticks": 150,
        "temp_anneal_gens": 30,
        "argmax_penalty": 0.0,
        "entropy_coef": 0.0,
        "temp_start": 1.0,
        "temp_end": 1.0,
        "uniform_bias_init": False,
        "rollouts_per_delta": 3,
    },
    "deep": {
        "coevolution": True,
        "generations": "80",
        "iter_per_gen": 20,
        "n_deltas": 20,
        "n_eval_ticks": 200,
        "temp_anneal_gens": 60,
        "argmax_penalty": 0.0,
        "entropy_coef": 0.0,
        "temp_start": 1.0,
        "temp_end": 1.0,
        "uniform_bias_init": False,
        "rollouts_per_delta": 3,
    },
}


def add_profile_argument(parser):
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default=None,
        help="Training preset: sanity (quick check), info (standard run), or "
             "deep (long run). Explicit CLI flags override preset values.",
    )


def parse_training_args(parser, argv=None, *, destinations=None):
    """Apply explicit CLI values > profile values > ordinary parser defaults.

    Reparse with suppressed defaults to identify supplied destinations, including
    aliases, --flag=value, abbreviations, and paired boolean flags. Copying the
    parser preserves optional-value const semantics and leaves its defaults alone.
    """
    args = parser.parse_args(argv)
    if args.profile is None:
        return args
    explicit_parser = copy.deepcopy(parser)
    explicit_parser._defaults.clear()
    for action in explicit_parser._actions:
        action.default = argparse.SUPPRESS
    explicit = vars(explicit_parser.parse_args(argv))
    destinations = destinations or {}
    flags = {action.dest: action.option_strings[0]
             for action in parser._actions if action.option_strings}
    applied, overridden = {}, []
    for key, value in PROFILES[args.profile].items():
        dest = destinations.get(key, key)
        if dest in explicit:
            if explicit[dest] != value:
                overridden.append((dest, explicit[dest], value))
        else:
            setattr(args, dest, value)
            applied[dest] = value

    print("==========================================")
    print(f"  PROFILE ACTIVE: --profile {args.profile}")
    print("==========================================")
    if applied:
        print("Profile values applied (no explicit CLI override):")
        for dest, value in applied.items():
            print(f"  {flags[dest]} = {value}")
    else:
        print(f"Profile '{args.profile}': every profile key was "
              "overridden by an explicit CLI flag.")
    if overridden:
        print("\n[i] EXPLICIT CLI FLAGS OVERRIDE PROFILE - the following "
              "profile values were NOT applied because you specified them "
              "explicitly on the command line:")
        for dest, user_value, profile_value in overridden:
            print(f"  {flags[dest]}: using your value {user_value!r} "
                  f"(profile '{args.profile}' would have used {profile_value!r})")
    print("------------------------------------------")
    return args
