"""Spawn-strategier för initial biomass-fördelning per FG.

Modulen är fristående och importeras inte av övriga `lib/` än — den är en
skiss för diskussion innan inkoppling i `_spawn_biomass_distribution`
(`lib/config/config_loader.py`).
"""
from .strategies import (
    get_strategy,
    register_strategy,
    make_weights,
    weights_uniform,
    weights_perlin,
    weights_colony,
    weights_env_driven,
    StrategySpec,
)
from .allocator import distribute_with_floor, summarize

__all__ = [
    "get_strategy",
    "register_strategy",
    "make_weights",
    "weights_uniform",
    "weights_perlin",
    "weights_colony",
    "weights_env_driven",
    "StrategySpec",
    "distribute_with_floor",
    "summarize",
]
