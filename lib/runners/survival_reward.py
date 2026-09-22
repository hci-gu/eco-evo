"""Per-species survival first, with a bounded late-life energy tie-breaker."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class SurvivalReward:
    lower: float = 0.3

    def __post_init__(self):
        if not math.isfinite(self.lower) or not 0 < self.lower <= 1:
            raise ValueError("Survival reward biomass floor must be in (0, 1]")

    def metadata(self):
        return asdict(self)


def add_survival_reward_arguments(parser):
    parser.add_argument("--survival-reward", action="store_true",
                        help="Prioritise per-species viable ticks; late-life reserve quality breaks ties")
    parser.add_argument("--survival-reward-floor", type=float, default=.3,
                        help="First biomass dip below this fraction of start ends viability (default: 0.3)")


def survival_reward_options(args):
    return SurvivalReward(args.survival_reward_floor) if args.survival_reward else None


def validate_survival_reward(config, integral, legacy, stability, local,
                             entropy_coef=0., argmax_penalty=0.):
    if config is not None and (not integral or legacy or stability is not None or local):
        raise ValueError("--survival-reward replaces fitness; do not combine with legacy, "
                         "local, population-stability or no-integral reward options")
    if config is not None and (entropy_coef != 0 or argmax_penalty != 0):
        raise ValueError("--survival-reward requires zero entropy-coef and argmax-penalty; "
                         "use --profile info to keep survival as the primary objective")


class SurvivalScore:
    """Count a contiguous viable prefix; failure/recovery never restarts it.

    Q is mean reserve fullness R/(B*max_reserve) over the final ceil(T/3)
    viable ticks. Empty starts and failure on tick one score zero. Every
    extra viable tick outweighs any Q difference in the same world/horizon.
    The ecosystem keeps running after a species fails its scoring threshold.
    """
    def __init__(self, env, config):
        self.config = config
        self.ids = list(env.dm_ids)
        self.b0 = np.array([env.fgs[f].biomass.sum() for f in self.ids], dtype=np.float64)
        self.capacity = np.array([env.fgs[f].max_energy_reserve for f in self.ids], dtype=np.float64)
        self.alive = self.b0 > 0
        self.quality = [[] for _ in self.ids]

    def step(self, env):
        for i, fid in enumerate(self.ids):
            if not self.alive[i]:
                continue
            fg = env.fgs[fid]
            b, r = float(fg.biomass.sum()), float(fg.energy_reserve.sum())
            self.alive[i] = (math.isfinite(b) and math.isfinite(r) and r >= 0
                             and b > 0 and b >= self.config.lower * self.b0[i])
            if self.alive[i]:
                q = np.clip(r / max(b * self.capacity[i], 1e-30), 0, 1)
                self.quality[i].append(float(q))

    def results(self, horizon):
        if horizon < 1:
            raise ValueError("Survival reward requires a positive horizon")
        result = {}
        for fid, values in zip(self.ids, self.quality):
            ticks = len(values)
            tail = (ticks + 2) // 3
            q = float(np.mean(values[-tail:])) if ticks else 0.
            result[fid] = (ticks + .5 * q) / horizon
        return result


def evaluate_survival(env, fg_list, ticks, config, collect_observations=False):
    score = SurvivalScore(env, config)
    if ticks < 1:
        raise ValueError("Survival reward requires a positive horizon")
    for _ in range(ticks):
        observation = env.get_observation()
        env.step(env.policy_controller.forward(observation))
        score.step(env)
    fitness = score.results(ticks)
    samples = None
    if collect_observations and getattr(env, "_obs_sum", None) is not None:
        samples = (env._obs_sum.copy(), env._obs_sumsq.copy(), env._obs_count)
    diagnostics = None
    if getattr(env, "_action_entropy_sum", None) is not None:
        diagnostics = {}
        for fid in fg_list:
            i = env.dm_ids.index(fid)
            count = max(1, int(env._action_active_ticks[i]))
            diagnostics[fid] = dict(
                entropy=float(env._action_entropy_sum[i] / count),
                max_entropy=float(env._action_max_entropy),
                move_frac=float(env._action_move_frac[i] / count),
                rest_frac=float(env._action_rest_frac[i] / count),
                eat_frac=float(env._action_eat_frac[i] / count),
                present_frac=float(env._action_active_ticks[i] / ticks))
    return {f: fitness[f] for f in fg_list}, samples, diagnostics


class TensorSurvivalScore:
    """Fixed-address device buffers; prefix sums allow exact late-life means."""
    def __init__(self, shape, capacity, device, config):
        self.config = config
        self.alive = torch.zeros(shape, dtype=torch.bool, device=device)
        self.ticks = torch.zeros(shape, dtype=torch.int64, device=device)
        self.quality_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.prefix = torch.zeros((capacity + 1, *shape), dtype=torch.float64, device=device)
        self.buffers = [self.alive, self.ticks, self.quality_sum, self.prefix]

    def reset(self, b0):
        self.alive.copy_(b0 > 0)
        self.ticks.zero_()
        self.quality_sum.zero_()
        self.prefix.zero_()

    def step(self, b, r, b0, reserve_capacity, tick):
        valid = (self.alive & torch.isfinite(b) & torch.isfinite(r) & (r >= 0)
                 & (b > 0) & (b >= self.config.lower * b0))
        self.ticks.add_(valid.long())
        q = (r / (b * reserve_capacity).clamp_min(1e-30)).clamp(0, 1)
        self.quality_sum.add_(torch.where(valid, q, 0.))
        self.prefix.index_copy_(0, (tick + 1).reshape(1), self.quality_sum[None])
        self.alive.copy_(valid)

    def results(self, horizon):
        tail = (self.ticks + 2) // 3
        start = self.ticks - tail
        before = self.prefix.gather(0, start[None])[0]
        q = ((self.quality_sum - before) / tail.clamp_min(1)).clamp(0, 1)
        return (self.ticks.double() + .5 * q) / horizon
