__all__ = ["EcosystemEnvironment"]


def __getattr__(name):
    if name == "EcosystemEnvironment":
        from lib.environments.ecosystem_env.environment import EcosystemEnvironment
        return EcosystemEnvironment
    raise AttributeError(name)
