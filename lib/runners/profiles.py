"""Shared CPU/GPU training presets; explicit CLI options override these defaults."""


from lib.training_profiles import PROFILES


def gpu_profile(name):
    """Translate CPU option destinations to the GPU parser's equivalent names."""
    aliases = {"n_eval_ticks": "ticks", "rollouts_per_delta": "worlds"}
    return {aliases.get(key, key): value for key, value in PROFILES[name].items()}
