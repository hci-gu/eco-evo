"""Warm-up before a training rollout (section 200).

A world starts from sampled totals that ignore the season: phytoplankton
far above the model's own level for the date, zooplankton 8-30x its
winter stock. The first month or two of every rollout was the world
settling, and the zooplankton policy partly learned in that transient.

With ``--warmup-days D`` each world first runs D days with the
UNPERTURBED policies (the current theta), shared by every candidate
evaluated on that world, and the reward starts from the state it
reaches. Shared, not per candidate, so the +/- candidates still start
from the same state (common random numbers, section 150) and nothing a
candidate does goes unscored. The calendar runs on through the warm-up:
the scored rollout starts D days after the sampled start day, which is
still uniform over the year with ``start_day_of_year: random``.

Both trainers use it: the GPU trainer once per world and update, the CPU
trainer inside every rollout task (the +/- tasks of a pair share seed
and base weights, and the warm-up runs on a stream seeded from that seed,
so they reach the same state).
"""
import math

import numpy as np

from lib.world.tick_time import ticks_per_day


# ---- Re-anchoring after the warm-up (section 274) --------------------------
#
# The warm-up settles the plankton and the spatial patterns, but it also
# moved the levels: fix101's porpoises lost half their tonnes in days 5-20
# of a fresh start while the planktivores left their spawn cells, so every
# scored rollout began at about 0.4 of the drawn total. The groups the
# manifest lists under ``simulation_settings.warmup.reanchor`` are scaled
# back to their start totals after the warm-up, keeping the settled layout,
# each cell's reserve fill and the juvenile share. A group whose level an
# outside exchange sets (an open pool or a migrating window) is refused by
# the loader: re-anchoring it would restart its relaxation instead.

def carries_tissue_reserve(fg):
    """True when ``energy_reserve`` belongs to the group's tonnes: a
    decision maker's reserve (MJ) or a refuge's accessible part (t). A
    nutrient producer keeps its pool there - water, not tissue - and any
    other non decision maker's slot means nothing."""
    return bool(getattr(fg, "is_decision_maker", False)) or getattr(fg, "exposed_share", None) is not None


def reanchor_snapshot(env):
    """The start state of every re-anchored group: {fid: (total, biomass,
    reserve, juvenile)} (copies; juvenile None without a stage)."""
    out = {}
    for fid, fg in env.fgs.items():
        if not getattr(fg, "warmup_reanchor", False):
            continue
        reserve = getattr(fg, "energy_reserve", None)
        juvenile = getattr(fg, "juvenile", None)
        out[fid] = (float(np.asarray(fg.biomass, dtype=np.float64).sum()),
                    np.array(fg.biomass, copy=True),
                    None if reserve is None else np.array(reserve, copy=True),
                    None if juvenile is None else np.array(juvenile, copy=True))
    return out


def reanchor(env, snapshot):
    """Scale each group in ``snapshot`` back to its start total.

    Biomass, juveniles and (for a tissue reserve) the reserve take the same
    factor, so the layout, every cell's fill and the juvenile share are the
    warmed ones. A group that died out in the warm-up gets its start fields
    back. Returns {fid: factor} (None = restored)."""
    factors = {}
    for fid, (target, biomass, reserve, juvenile) in snapshot.items():
        fg = env.fgs[fid]
        current = float(np.asarray(fg.biomass, dtype=np.float64).sum())
        if target <= 0.0:
            continue
        if current <= 0.0:
            # In place, so references the env holds to the arrays follow.
            fg.biomass[...] = biomass
            if reserve is not None:
                fg.energy_reserve[...] = reserve
            if juvenile is not None:
                fg.juvenile = juvenile.copy()
            factors[fid] = None
            continue
        k = target / current
        fg.biomass[...] = np.asarray(fg.biomass, dtype=np.float64) * k
        if carries_tissue_reserve(fg) and getattr(fg, "energy_reserve", None) is not None:
            fg.energy_reserve[...] = np.asarray(fg.energy_reserve, dtype=np.float64) * k
        if getattr(fg, "juvenile", None) is not None:
            fg.juvenile = (np.asarray(fg.juvenile, dtype=np.float64) * k).astype(
                fg.juvenile.dtype, copy=False)
        factors[fid] = k
    return factors


def warmup_ticks(days, tick_hours=None):
    """Whole ticks in ``days`` at the run's tick length (0 = off)."""
    days = float(days or 0.0)
    if not math.isfinite(days) or days < 0.0:
        raise ValueError("--warmup-days must be a finite number >= 0")
    return int(round(days * ticks_per_day(tick_hours)))


def add_warmup_argument(parser):
    """Register ``--warmup-days`` (shared by train.py and train_gpu.py)."""
    parser.add_argument(
        "--warmup-days", "--warmup_days", dest="warmup_days", type=float,
        default=0.0,
        help=("Days each training world runs with the unperturbed policies "
              "before the reward starts (default 0 = off). Lets the "
              "plankton settle from the sampled start totals; shared by "
              "the +/- candidates of a world. Section 200. The groups in "
              "the manifest's simulation_settings.warmup.reanchor are then "
              "scaled back to their start totals, and the probes start "
              "after the same warm-up (section 274)."))


def run_warmup(env, ticks, seed=None):
    """Advance a CPU env ``ticks`` ticks with its current policies.

    The tick reads the env's batched copy of the weights when it has one, so
    a caller that swaps the policies' weights calls
    ``env.rebuild_batched_weights()`` before this and again after swapping
    back.

    The CPU tick draws its FG order and the seeding noise from the global
    ``np.random``, which is not re-seeded per rollout task. With ``seed``
    (the delta pair's world seed) the warm-up runs on its own stream, so
    the +/- tasks of a pair reach the same state on any worker; the
    previous global state is restored afterwards, so the scored rollout
    draws exactly as it did without a warm-up.

    The observation-normalisation and action-statistic accumulators are
    cleared afterwards, so the scored rollout reports only its own ticks
    (the GPU trainer never merges warm-up observations either).
    """
    saved = None
    snapshot = reanchor_snapshot(env) if int(ticks) > 0 else {}
    if seed is not None:
        saved = np.random.get_state()
        np.random.seed((int(seed) ^ 0x5EED200) & 0x7FFFFFFF)
    try:
        for _ in range(int(ticks)):
            observation = env.get_observation()
            actions = env.policy_controller.forward(observation)
            env.step(actions)
    finally:
        if saved is not None:
            np.random.set_state(saved)
    # Section 274: the re-anchored groups start the scored part at their
    # start totals, on the settled layout.
    reanchor(env, snapshot)
    env._obs_sum = None
    env._obs_sumsq = None
    env._obs_count = 0
    env._action_entropy_sum = None


def _reset_books(env):
    """Zero the loss, exchange and intake books (as the env starts them),
    so a probe or inference rollout after a warm-up reports its own ticks."""
    from lib.environments.ecosystem_env.loss_accounting import LOSS_CAUSES
    for _name, attr in LOSS_CAUSES:
        setattr(env, attr, {fid: 0.0 for fid in env.fgs})
    env.gain_exchange = {fid: 0.0 for fid in env.fgs}
    env.loss_natural_parts = {}
    env._extinction_events = {fid: 0 for fid in env.fgs}
    env.intake_by_pred_prey = {fid: {} for fid in env.fgs}
    if hasattr(env, "intake_juvenile_by_pred_prey"):
        env.intake_juvenile_by_pred_prey = {}


def rebase_calendar(env):
    """Make the env's current tick its tick 0: ``light_start`` moves on by
    the ticks run, ``tick_count`` restarts. Everything on the calendar
    reads ``light_start + tick_count``, so the date is unchanged, and the
    tensor engine (which uploads ``light_start`` and counts its own ticks)
    then reads the same day as the CPU engine. The currents' phase, keyed on
    ``tick_count``, jumps once here."""
    period = getattr(env, "light_period", None)
    if period:
        env.light_start = (int(getattr(env, "light_start", 0)) + int(env.tick_count)) % int(period)
    env.tick_count = 0


def settle(env, ticks, seed=None, policies=None, obs=None):
    """The training warm-up for a probe or inference world (section 274).

    Runs ``ticks`` ticks with ``policies`` (default the env's own) and the
    observation statistics ``obs`` = (mean, var) (default the env's),
    re-anchors the listed groups, makes the settled state the env's tick 0
    and clears the books. The scored start is the start day plus the
    warm-up, as in training. A no-op for ``ticks`` <= 0.
    """
    ticks = int(ticks or 0)
    if ticks <= 0:
        return
    saved_policies = saved_obs = None
    if policies is not None:
        saved_policies = env.policies
        env.policies = policies
    if obs is not None:
        saved_obs = (getattr(env, "obs_mean", None), getattr(env, "obs_var", None))
        env.obs_mean, env.obs_var = obs
    if policies is not None and hasattr(env, "rebuild_batched_weights"):
        env.rebuild_batched_weights()
    try:
        run_warmup(env, ticks, seed=seed)
    finally:
        if saved_policies is not None:
            env.policies = saved_policies
            if hasattr(env, "rebuild_batched_weights"):
                env.rebuild_batched_weights()
        if saved_obs is not None:
            env.obs_mean, env.obs_var = saved_obs
    rebase_calendar(env)
    _reset_books(env)
