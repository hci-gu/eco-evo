"""Opt-in, fixed-horizon fitness for keeping an ecosystem within biomass bounds."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PopulationStability:
    lower: float = 0.10
    upper: float = 3.0
    warning_lower: float = 0.20
    warning_upper: float = 2.5

    # Healthy tick fitness lies in [-4, 1], strictly above failure fitness.
    energy_floor = -3.0
    energy_cap = 1.0
    failure_reward = -5.0

    def __post_init__(self):
        values = (self.lower, self.warning_lower, self.warning_upper, self.upper)
        if not all(math.isfinite(v) for v in values) or not (
            0 < self.lower < self.warning_lower <= 1 <= self.warning_upper < self.upper
        ):
            raise ValueError("Population bounds must satisfy 0 < min < warning-min <= 1 "
                             "<= warning-max < max, with finite values")

    def metadata(self):
        return asdict(self)


def add_population_arguments(parser):
    parser.add_argument("--population-stability", "--population_stability", action="store_true",
                        help="Enable ecosystem failure bounds and warning penalties; requires "
                             "integral total-energy reward (the default).")
    for name, default, description in (
        ("min", 0.10, "Fail below this fraction of each decision maker's starting biomass"),
        ("max", 3.0, "Fail at or above this multiple of starting biomass"),
        ("warning-min", 0.20, "Start the lower warning penalty below this fraction"),
        ("warning-max", 2.5, "Start the upper warning penalty above this multiple"),
    ):
        flag = "population-" + name
        parser.add_argument("--" + flag, "--" + flag.replace("-", "_"), type=float,
                            default=default, help=description + f" (default: {default})")


def population_options(args):
    if not args.population_stability:
        return None
    config = PopulationStability(args.population_min, args.population_max,
                                 args.population_warning_min, args.population_warning_max)
    validate_reward(config, args.integral_reward, args.legacy_reward)
    return config


def validate_reward(config, integral_reward, legacy_reward):
    if config is not None and (legacy_reward or not integral_reward):
        raise ValueError("Population stability requires integral total-energy reward; "
                         "remove --legacy-reward/--legacyreward and --no-integral-reward")


class StabilityScore:
    """CPU score accumulator, also usable by visualisation baseline rollouts.

    Any initially present decision maker can fail the entire world. Failed
    worlds remain failed even if biomass later recovers. The failure tick is
    included in the tail penalty, and the denominator is always the horizon.
    """

    def __init__(self, env, config):
        self.config = config
        self.ids = list(env.dm_ids)
        self.b0 = np.array([env.fgs[f].biomass.sum() for f in self.ids], dtype=np.float64)
        self.ec = np.array([env.fgs[f].params.get("energy_content", 0.0) or 0.0
                            for f in self.ids], dtype=np.float64)
        r0 = np.array([env.fgs[f].energy_reserve.sum() for f in self.ids], dtype=np.float64)
        if not np.isfinite(self.b0).all() or (self.b0 < 0).any():
            raise ValueError("Initial biomass must be finite and nonnegative")
        self.e0 = self.b0 * self.ec + r0
        self.epsilon = np.maximum(1e-6 * self.e0, 1e-9)
        self.total = np.zeros(len(self.ids), dtype=np.float64)
        self.failed = False
        self.valid_ticks = 0

    def step(self, env):
        if self.failed:
            return
        c = self.config
        b = np.array([env.fgs[f].biomass.sum() for f in self.ids], dtype=np.float64)
        r = np.array([env.fgs[f].energy_reserve.sum() for f in self.ids], dtype=np.float64)
        present = self.b0 > 0
        ratio = np.divide(b, self.b0, out=np.ones_like(b), where=present)
        energy = b * self.ec + r
        self.failed = bool(np.any(present & ((ratio < c.lower) | (ratio >= c.upper)))
                           or not np.isfinite(energy).all() or not np.isfinite(b).all()
                           or (energy < 0).any())
        if self.failed:
            return
        low = np.clip((c.warning_lower - ratio) / (c.warning_lower - c.lower), 0, 1)
        high = np.clip((ratio - c.warning_upper) / (c.upper - c.warning_upper), 0, 1)
        warning = np.max(np.where(present, np.maximum(low, high), 0), initial=0)
        base = np.log((energy + self.epsilon) / (self.e0 + self.epsilon))
        self.total += np.clip(base, c.energy_floor, c.energy_cap) - warning
        self.valid_ticks += 1

    def results(self, horizon):
        tail = horizon - self.valid_ticks if self.failed else 0
        return dict(zip(self.ids, (self.total + tail * self.config.failure_reward) / horizon))


def evaluate_stability(env, fg_list, n_ticks, config, collect_observations=False):
    if n_ticks < 1:
        raise ValueError("Population stability rollouts require at least one tick")
    score = StabilityScore(env, config)
    for _ in range(n_ticks):
        observation = env.get_observation()
        actions = env.policy_controller.forward(observation)
        env.step(actions)
        score.step(env)
        if score.failed:
            break
    fitness = score.results(n_ticks)
    samples = None
    if collect_observations and getattr(env, "_obs_sum", None) is not None:
        samples = (env._obs_sum.copy(), env._obs_sumsq.copy(), env._obs_count)
    diagnostics = None
    if getattr(env, "_action_entropy_sum", None) is not None and env._action_entropy_count > 0:
        diagnostics = {}
        count = env._action_entropy_count
        active = getattr(env, "_action_active_ticks", None)
        for fid in fg_list:
            i = env.dm_ids.index(fid)
            denominator = max(1, int(active[i]) if active is not None else count)
            diagnostics[fid] = {
                "entropy": float(env._action_entropy_sum[i] / denominator),
                "max_entropy": float(env._action_max_entropy),
                "move_frac": float(env._action_move_frac[i] / denominator),
                "rest_frac": float(env._action_rest_frac[i] / denominator),
                "eat_frac": float(env._action_eat_frac[i] / denominator),
                "present_frac": float(denominator / max(count, 1)),
            }
    return {f: float(fitness[f]) for f in fg_list}, samples, diagnostics
