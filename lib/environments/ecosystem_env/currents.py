"""Configuration and compatibility exports for passive cloud-driven currents."""

from dataclasses import asdict, dataclass
import math

from lib.environments.ecosystem_env.movement import (
    apply_currents,
    current_direction_fractions as direction_fractions,
)


@dataclass(frozen=True)
class CurrentConfig:
    strength: float = 0.1
    period: int = 20
    seed: int = 0
    scale: float = 12.0

    def __post_init__(self):
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("Current strength must be finite and between 0 and 1")
        if not isinstance(self.period, int) or self.period < 1:
            raise ValueError("Current period must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Current seed must be an integer in [0, 2**32)")
        if not math.isfinite(self.scale) or self.scale < 1:
            raise ValueError("Current scale must be finite and at least one cell")

    def metadata(self):
        return asdict(self)


def add_current_arguments(parser):
    parser.add_argument("--currents", choices=("on", "off"), default="off",
                        help="Cloud-driven east/south drift of non-decision makers (default: off)")
    parser.add_argument("--current-strength", "--current_strength", type=float, default=0.1,
                        help="Maximum biomass fraction transported per tick (default: 0.1)")
    parser.add_argument("--current-period", "--current_period", type=int, default=20,
                        help="Ticks to scroll the noise field one cell in +X and +Y (default: 20)")
    parser.add_argument("--current-seed", "--current_seed", type=int, default=0,
                        help="Independent current seed, combined with the rollout seed (default: 0)")
    parser.add_argument("--current-scale", "--current_scale", type=float, default=12.0,
                        help="Coarse noise lattice spacing in cells, at least 1 (default: 12)")


def current_options(args):
    config = CurrentConfig(args.current_strength, args.current_period,
                           args.current_seed, args.current_scale)
    return config if args.currents == "on" else None
