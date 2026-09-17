"""Per-cell source tracking for the local reward (``--local_reward``).

Motivation
----------
The default fitness is a single global quantity,
``mean_t log(E_total(t) / E_total(0))`` with
``E_total = sum_c (B(c)*energy_content + R(c))``. It is dominated by the
large cells: a policy that manages a thousand-ton cell well and a
one-kilogram cell badly scores the same as the reverse.

The local reward asks a different question, per cell and per tick:

    A(c, t) = B(c,t)*energy_content + R(c,t)      (start of tick)
    B(c, t+1) = sum_d share_d(c) * Q(dest(c,d), t+1)
    reward(c) = B(c, t+1) / A(c, t)

where the sum runs over the plus-shaped destination set
``{stay, N, E, S, W}`` reachable by one move action, and ``share_d(c)``
is the fraction of the destination cell's post-movement biomass that
came from ``c``. Every cell contributes one scale-invariant ratio, so
the reward measures *how well the population in c fared*, independent
of how big that population was.

Why the proportional allocation is exact
----------------------------------------
``environment.step()`` runs

    impact_mortality -> predation -> apply_energy_costs -> apply_movement
    -> apply_currents -> population_change -> accessibility_mask
    -> extinction_threshold

Impacts and predation happen *before* the movement, so those losses are
already booked in the source cell -- nothing to track. Everything
*after* the movement is cell-wise multiplicative (``population_change``
scales B and R by a per-cell factor, the accessibility mask and the
extinction sweep multiply by 0/1, the reserve clip is per cell), and
``apply_currents`` only touches non decision makers. A destination
cell's end-of-tick energy can therefore be split between its sources in
proportion to the biomass each of them delivered; this is an identity,
not an approximation.

Edge cases
----------
* ``suppress_subthreshold_splits`` cancels move outflows that would land
  in a still-sub-threshold cell and returns them to the source. The
  contributions are read *after* that correction, otherwise the mass
  balance does not close.
* Outflow that leaves the grid (always, at the border) has no in-grid
  destination. Its share is removed from the denominator instead of
  being counted as a loss -- ``A`` is scaled by the in-grid biomass
  fraction ``frac_in``, so a border cell is not punished for a move the
  tracker cannot follow.
* With ``--migration on`` the immigrating biomass has no source cell.
  Because the shares are normalised by the destination's *total*
  post-movement biomass, the immigrated part is simply attributed to
  nobody. Consequently ``sum_c B(c,t+1) < sum_c Q(c,t+1)`` when
  migration is on; the mass-balance identity only holds exactly with
  ``--migration off``.
"""

import numpy as np

from lib.environments.ecosystem_env.constants import EAST, NORTH, SOUTH, WEST

METRICS = ("log", "ratio")
NORMS = ("mean", "sum")


class LocalRewardConfig:
    """Aggregation settings for the per-cell local reward.

    Attributes:
        metric: ``"log"`` uses ``log(B/A)`` (symmetric, geometric mean
            growth, risk averse); ``"ratio"`` uses the raw quotient
            ``B/A`` as literally specified.
        norm: ``"mean"`` divides the weighted sum by the total weight,
            ``"sum"`` keeps the unnormalised sum over occupied cells.
            With ``sum`` the fitness grows with the number of occupied
            cells, which rewards thin spreading; ``mean`` removes that
            gradient.
        theta: exponent of the cell weight ``w_c = A_c**theta``.
            ``0.0`` weights every occupied cell equally (the local
            reward proper), ``1.0`` reduces to the global energy reward.
        clip_lo / clip_hi: the ratio ``B/A`` is clipped to
            ``[clip_lo, clip_hi]`` before the metric is applied, which
            bounds the contribution of a nearly empty cell.
        min_energy_factor: a cell only takes part when
            ``A_c >= min_energy_factor * extinction_threshold *
            energy_content``, i.e. when it held a viable population at
            the start of the tick.
    """

    __slots__ = ("metric", "norm", "theta", "clip_lo", "clip_hi",
                 "min_energy_factor")

    def __init__(self, metric="log", norm="mean", theta=0.0, clip_lo=0.2,
                 clip_hi=2.0, min_energy_factor=1.0):
        metric = str(metric)
        norm = str(norm)
        if metric not in METRICS:
            raise ValueError(
                "local reward metric must be one of %s, got %r"
                % (list(METRICS), metric))
        if norm not in NORMS:
            raise ValueError(
                "local reward norm must be one of %s, got %r"
                % (list(NORMS), norm))
        clip_lo = float(clip_lo)
        clip_hi = float(clip_hi)
        if not (0.0 < clip_lo <= clip_hi):
            raise ValueError(
                "local reward clip range must satisfy 0 < clip_lo <= "
                "clip_hi, got (%r, %r)" % (clip_lo, clip_hi))
        self.metric = metric
        self.norm = norm
        self.theta = float(theta)
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.min_energy_factor = float(min_energy_factor)

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, data):
        """Build a config from a plain dict (or pass through None)."""
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        return cls(**dict(data))

    def __repr__(self):
        return "LocalRewardConfig(%s)" % ", ".join(
            "%s=%r" % (k, v) for k, v in self.as_dict().items())


def local_reward_options(args):
    """``LocalRewardConfig`` from parsed CLI args, or ``None`` when off.

    Mirrors ``population_options`` / ``current_options`` so train.py can
    build every optional reward the same way.
    """
    if not getattr(args, "local_reward", False):
        return None
    clip = getattr(args, "local_reward_clip", None) or (0.2, 2.0)
    return LocalRewardConfig(
        metric=getattr(args, "local_reward_metric", "log"),
        norm=getattr(args, "local_reward_norm", "mean"),
        theta=getattr(args, "local_reward_theta", 0.0),
        clip_lo=clip[0],
        clip_hi=clip[1],
        min_energy_factor=getattr(
            args, "local_reward_min_energy_factor", 1.0),
    )


def add_local_reward_arguments(parser):
    """Register the ``--local_reward`` family on an argument parser.

    Shared by ``train.py`` and ``lib/gpu/cli.py`` so the CPU and the GPU
    training entry points cannot drift apart on flag names, choices or
    defaults. Mirrors ``add_current_arguments`` /
    ``add_population_arguments``.
    """
    # Instead of the single global quantity log(E_total/E0), score every
    # occupied cell separately: A(c,t) is the cell's energy at the start
    # of the tick, B(c,t+1) the end-of-tick energy of the biomass that
    # started in c (wherever it moved or split to), and reward(c) = B/A.
    # The ratio is scale invariant, so a small cell's good decision
    # counts as much as a large cell's.
    parser.add_argument("--local_reward", "--localreward", dest="local_reward",
                        action="store_true", default=False,
                        help="Use the per-cell source-tracked reward: per tick, "
                             "aggregate reward(c) = B(c,t+1)/A(c,t) over the cells "
                             "occupied at the start of the tick, where A is the "
                             "cell's energy (B*energy_content + R) and B the "
                             "end-of-tick energy of exactly that population, "
                             "tracked through the move/split into {c,N,E,S,W}. "
                             "Fitness is the mean over rollout ticks. Scale "
                             "invariant per cell, so small populations count as "
                             "much as large ones. Mutually exclusive with "
                             "--legacy_reward and --population-stability.")
    parser.add_argument("--local_reward_metric", choices=["log", "ratio"],
                        default="log",
                        help="Per-cell transform before aggregation. 'log' (default) "
                             "uses log(B/A), which is symmetric in gain/loss and "
                             "optimises the geometric mean growth rate (risk "
                             "averse). 'ratio' uses the raw quotient B/A, which is "
                             "asymmetric and mildly rewards boom-bust.")
    parser.add_argument("--local_reward_norm", choices=["mean", "sum"],
                        default="mean",
                        help="Aggregation over cells. 'mean' (default) divides by "
                             "the total cell weight. 'sum' keeps the raw sum, which "
                             "grows with the number of occupied cells and therefore "
                             "also rewards spreading thin just above the extinction "
                             "threshold. Default: mean.")
    parser.add_argument("--local_reward_theta", type=float, default=0.0,
                        help="Cell weight exponent: w_c = A_c**theta. 0.0 (default) "
                             "weights every occupied cell equally (the local reward "
                             "proper); 1.0 with --local_reward_norm mean reduces to "
                             "the global energy-weighted growth rate, which makes "
                             "theta a single knob for A/B testing local vs global.")
    parser.add_argument("--local_reward_clip", type=float, nargs=2,
                        default=[0.2, 2.0], metavar=("LO", "HI"),
                        help="Clip range for the per-cell ratio B/A before the "
                             "metric is applied. Bounds the contribution of a "
                             "nearly empty cell, where float32 residue would "
                             "otherwise dominate the quotient. Default: 0.2 2.0.")
    parser.add_argument("--local_reward_min_energy_factor", type=float, default=1.0,
                        help="A cell participates only when A_c >= factor * "
                             "extinction_threshold_factor * min_split_biomass * "
                             "energy_content, i.e. when it held a viable population "
                             "at the start of the tick. Default: 1.0.")


def is_enabled(env):
    return getattr(env, "local_reward", None) is not None


def attach(env, config):
    """Enable tracking on an already built env (used by the runners).

    ``config`` may be a ``LocalRewardConfig``, a plain dict or ``None``;
    the env builders stay untouched so every rollout path (training,
    probe, inference) can opt in the same way.
    """
    config = LocalRewardConfig.from_dict(config)
    env.local_reward = config
    reset(env)
    return config


def reset(env):
    """(Re)initialise the accumulators. Called from the env ctor."""
    env.local_reward_sum = {fid: 0.0 for fid in env.fgs}
    env.local_reward_occupied = {fid: 0.0 for fid in env.fgs}
    env.local_reward_ticks = 0
    env.local_reward_last = None
    env._local_start_energy = None
    env._local_contrib = None


def _energy_content(env):
    ec = getattr(env, "_local_dm_ec", None)
    if ec is None or len(ec) != env.N_dm:
        ec = np.array(
            [float(env.fgs[fid].params.get("energy_content", 0.0) or 0.0)
             for fid in env.dm_ids],
            dtype=env.dtype,
        )
        env._local_dm_ec = ec
    return ec


def _cell_energy(env):
    """Q(c) = B(c)*energy_content + R(c) for every DM, shape (N_dm,H,W)."""
    biomass = np.stack([env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    reserve = np.stack(
        [env.fgs[fid].energy_reserve for fid in env.dm_ids], axis=0)
    return biomass * _energy_content(env)[:, None, None] + reserve


def _gather_destinations(values):
    """Neighbour values re-indexed on the SOURCE cell.

    ``out[:, d, y, x]`` is ``values`` at the cell that a direction-``d``
    move from ``(y, x)`` lands in, and 0 where that destination is off
    the grid. The offsets mirror ``movement._transfer_to_neighbours``.
    """
    out = np.zeros((values.shape[0], 4) + values.shape[1:],
                   dtype=values.dtype)
    out[:, NORTH, 1:, :] = values[:, :-1, :]      # (y,x) -> (y-1,x)
    out[:, EAST, :, :-1] = values[:, :, 1:]       # (y,x) -> (y,x+1)
    out[:, SOUTH, :-1, :] = values[:, 1:, :]      # (y,x) -> (y+1,x)
    out[:, WEST, :, 1:] = values[:, :, :-1]       # (y,x) -> (y,x-1)
    return out


def _in_grid_mask(env):
    """Boolean (1,4,H,W): does a direction-d move stay on the grid?"""
    mask = np.zeros((1, 4, env.H, env.W), dtype=bool)
    mask[:, NORTH, 1:, :] = True
    mask[:, EAST, :, :-1] = True
    mask[:, SOUTH, :-1, :] = True
    mask[:, WEST, :, 1:] = True
    return mask


def begin_tick(env):
    """Snapshot ``A(c, t)``: the state the policy actually observed.

    Must run before ``apply_impact_mortality`` so that a population
    wiped out by impacts or predation registers as a loss for its own
    cell.
    """
    if env.N_dm == 0:
        env._local_start_energy = None
        return
    env._local_start_energy = _cell_energy(env)
    env._local_contrib = None


def record_movement(env, b_stay, b_out, b_total):
    """Store the post-movement biomass flow, indexed on the source cell.

    ``b_stay`` is what stayed in the cell (rest + eat + the non-moving
    share of a move action plus any cancelled split), ``b_out[i,d]``
    what left towards direction ``d``, and ``b_total`` the resulting
    per-cell biomass. All three are read after
    ``suppress_subthreshold_splits`` so the flow sums to ``b_total``.
    """
    env._local_contrib = (
        np.array(b_stay, dtype=env.dtype, copy=True),
        np.array(b_out, dtype=env.dtype, copy=True),
        np.array(b_total, dtype=env.dtype, copy=True),
    )


def tracked_end_energy(env):
    """``B(c, t+1)`` per DM plus the in-grid biomass fraction.

    Returns ``(tracked, frac_in)``, both shaped (N_dm, H, W), or
    ``(None, None)`` when there is nothing to track.
    """
    contrib = getattr(env, "_local_contrib", None)
    if env.N_dm == 0 or contrib is None:
        return None, None

    b_stay, b_out, b_total = contrib
    q_end = _cell_energy(env)

    # Per-unit-biomass end-of-tick energy of every destination cell.
    # Everything after the movement is cell-wise multiplicative, so
    # ``q_end / b_total`` is exactly the energy each delivered ton is
    # worth at the end of the tick.
    with np.errstate(divide="ignore", invalid="ignore"):
        unit = np.where(b_total > 0.0, q_end / b_total, 0.0)
    unit = unit.astype(env.dtype, copy=False)

    unit_dest = _gather_destinations(unit)
    tracked = b_stay * unit + (b_out * unit_dest).sum(axis=1)

    b_move_total = b_out.sum(axis=1)
    b_move_in = (b_out * _in_grid_mask(env)).sum(axis=1)
    b_src = b_stay + b_move_total
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_in = np.where(b_src > 0.0, (b_stay + b_move_in) / b_src, 1.0)
    return tracked.astype(env.dtype, copy=False), frac_in.astype(
        env.dtype, copy=False)


def _min_energy(env, config):
    """Per-DM lower bound on ``A_c`` for a cell to take part."""
    thr = np.asarray(env._dm_split_thr, dtype=np.float64)
    floor = thr * np.asarray(_energy_content(env), dtype=np.float64)
    floor = floor * float(config.min_energy_factor)
    return np.maximum(floor, 1e-12)


def end_tick(env):
    """Accumulate this tick's aggregated local reward per DM."""
    config = getattr(env, "local_reward", None)
    if config is None or env.N_dm == 0:
        return
    start = getattr(env, "_local_start_energy", None)
    tracked, frac_in = tracked_end_energy(env)
    if start is None or tracked is None:
        return

    a_min = _min_energy(env, config)[:, None, None]
    a_eff = np.asarray(start, dtype=np.float64) * np.asarray(
        frac_in, dtype=np.float64)
    active = (np.asarray(start, dtype=np.float64) >= a_min) & (a_eff > 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(active,
                         np.asarray(tracked, dtype=np.float64)
                         / np.where(a_eff > 0.0, a_eff, 1.0),
                         1.0)
    ratio = np.clip(ratio, config.clip_lo, config.clip_hi)
    term = np.log(ratio) if config.metric == "log" else ratio

    if config.theta == 0.0:
        weight = np.where(active, 1.0, 0.0)
    else:
        weight = np.where(active,
                          np.power(np.maximum(a_eff, 0.0), config.theta),
                          0.0)

    weighted = (weight * term).sum(axis=(1, 2))
    total_weight = weight.sum(axis=(1, 2))
    occupied = active.sum(axis=(1, 2))

    if config.norm == "mean":
        value = np.where(total_weight > 0.0,
                         weighted / np.where(total_weight > 0.0,
                                             total_weight, 1.0),
                         0.0)
    else:
        value = weighted

    for i, fid in enumerate(env.dm_ids):
        env.local_reward_sum[fid] += float(value[i])
        env.local_reward_occupied[fid] += float(occupied[i])
    env.local_reward_ticks += 1
    # Per-cell diagnostics for this tick, in the same spirit as
    # ``loss_predation`` / ``loss_starvation``: the offline probes and
    # the mass-balance test read them instead of recomputing the shares.
    env.local_reward_last = {
        "start": start,
        "tracked": tracked,
        "frac_in": frac_in,
        "active": active,
        "ratio": ratio,
        "value": value,
        "occupied": occupied,
    }
    env._local_contrib = None
    env._local_start_energy = None


def fitness(env, fg_id):
    """Rollout fitness: the mean per-tick aggregate for ``fg_id``."""
    ticks = int(getattr(env, "local_reward_ticks", 0) or 0)
    if ticks <= 0:
        return 0.0
    return float(env.local_reward_sum.get(fg_id, 0.0)) / float(ticks)


def occupancy(env, fg_id):
    """Mean number of participating cells per tick (diagnostic)."""
    ticks = int(getattr(env, "local_reward_ticks", 0) or 0)
    if ticks <= 0:
        return 0.0
    return float(env.local_reward_occupied.get(fg_id, 0.0)) / float(ticks)
