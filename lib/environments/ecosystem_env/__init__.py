from lib.environments.ecosystem_env.constants import MAX_HARVEST_FRAC

__all__ = ["EcosystemEnvironment", "MAX_HARVEST_FRAC"]


def __getattr__(name):
    if name == "EcosystemEnvironment":
        from lib.environments.ecosystem_env.environment import EcosystemEnvironment
        return EcosystemEnvironment
    raise AttributeError(name)
