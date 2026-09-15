"""Shared CPU/GPU training presets; explicit CLI options override these defaults."""


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


def gpu_profile(name):
    """Translate CPU option destinations to the GPU parser's equivalent names."""
    aliases = {"n_eval_ticks": "ticks", "rollouts_per_delta": "worlds"}
    return {aliases.get(key, key): value for key, value in PROFILES[name].items()}
