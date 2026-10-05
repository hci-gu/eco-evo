"""Exposure-weighted natural mortality M1 (section 139).

A decision maker's residual mortality M1 (``natural_mortality``, per
tick) is normally one constant loss that its behaviour cannot touch.
For zooplankton that misses two things the literature is clear on:

* part of M1 is VISUAL predation by groups that are not FGs (fish
  larvae, Norway pout, 0-group fish): it only reaches the visible
  biomass and follows the light, like the modelled fish predators;
* part is TACTILE predation (chaetognaths, jellyfish, ctenophores)
  that hunts where the zooplankton hides by day, so hiding does not
  protect against it and may raise it (Tonnesson & Tiselius 2005;
  Ohman, Frost & Cohen 1983).

Per tick and cell, with pi the fraction resting (hiding) this tick:

    M1(t, c) = M1 * [ (1 - s_v - s_t)
                      + s_v * V / V_ref
                      + s_t * D / D_ref ]
    V = (1 - pi (1 - phi)) * m(t)     visual exposure
    D = (1 - pi) + rho * pi           tactile exposure

``phi`` is the FG's own ``visibility_floor``, ``m(t)`` the daylight
attack-rate multiplier (annual mean 1; 1 without the calendar) for
``m1_visual_dark_ratio`` / ``m1_visual_threshold_deg``, ``rho`` the
``depth_risk_ratio``. ``V_ref`` and ``D_ref`` are V and D at
``hide_reference`` (pi_ref) and the annual-mean light, so a population
that hides pi_ref of the time loses exactly ``natural_mortality`` on
average over the year and the derived growth_rate (growth_budget)
needs no change. Hiding more than pi_ref costs tactile risk; hiding
when it is dark buys no visual protection - which is what makes hiding
by day and feeding at night (diel vertical migration) the best answer.

Off unless an FG sets ``m1_visual_share`` or ``m1_tactile_share``; the
tick is then bit-identical to the constant M1.
"""
import math

import numpy as np

from lib.world import daylight


DEFAULT_HIDE_REFERENCE = 0.5
DEFAULT_VISUAL_DARK_RATIO = 0.1


def _share(params, key):
    raw = params.get(key, 0.0)
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number in [0, 1], got {raw!r}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1], got {value}")
    return value


def settings(params):
    """Validated exposure settings of one FG, or None when inert."""
    visual = _share(params, "m1_visual_share")
    tactile = _share(params, "m1_tactile_share")
    if visual == 0.0 and tactile == 0.0:
        return None
    if visual + tactile > 1.0 + 1e-9:
        raise ValueError("m1_visual_share + m1_tactile_share must be <= 1, "
                         f"got {visual} + {tactile}")
    rho = float(params.get("depth_risk_ratio", 1.0) or 1.0)
    # 0 / empty means the default: the FG editor writes empty fields as
    # 0.0, and a population that never hides is not a useful reference.
    reference = float(params.get("hide_reference", 0.0) or
                      DEFAULT_HIDE_REFERENCE)
    if not rho > 0.0:
        raise ValueError(f"depth_risk_ratio must be > 0, got {rho}")
    if not 0.0 <= reference <= 1.0:
        raise ValueError(f"hide_reference must be in [0, 1], got {reference}")
    dark = params.get("m1_visual_dark_ratio", DEFAULT_VISUAL_DARK_RATIO)
    dark = float(DEFAULT_VISUAL_DARK_RATIO if dark in (None, "") else dark)
    threshold = params.get("m1_visual_threshold_deg", None)
    threshold = float(daylight.DEFAULT_THRESHOLD_DEG
                      if threshold in (None, "") else threshold)
    return {"visual": visual, "tactile": tactile, "rho": rho,
            "reference": reference, "dark_ratio": dark,
            "threshold": threshold}


def build(env):
    """Per-DM exposure tables: ``env.m1_exposure`` = {fg_id: dict}."""
    env.m1_exposure = {}
    for i, fid in enumerate(env.dm_ids):
        cfg = settings(env.fgs[fid].params)
        if cfg is None:
            continue
        phi = float(env.dm_visibility_floor[i])
        cfg["index"] = i
        cfg["floor"] = phi
        cfg["v_ref"] = 1.0 - cfg["reference"] * (1.0 - phi)
        cfg["d_ref"] = (1.0 - cfg["reference"]) + cfg["rho"] * cfg["reference"]
        cfg["light"] = None
        calendar = getattr(env, "daylight", None)
        if calendar is not None and cfg["visual"] > 0.0:
            cfg["light"] = daylight.attack_multiplier(
                float(calendar["latitude_deg"]), int(calendar["tick_hours"]),
                cfg["dark_ratio"], cfg["threshold"]).astype(np.float64)
        env.m1_exposure[fid] = cfg
    env._has_m1_exposure = bool(env.m1_exposure)


def components(env, fg_id, rate):
    """Per-cell M1 split ``(visual, tactile, other)`` for this tick.

    ``rate`` is the FG's per-tick ``natural_mortality`` (already scaled
    by ``--mortality_multiplier``). Returns None for an FG without
    exposure weighting. The three arrays sum to the per-cell rate.
    """
    if not getattr(env, "_has_m1_exposure", False):
        return None
    cfg = env.m1_exposure.get(fg_id)
    if cfg is None:
        return None
    hidden = getattr(env, "pi_rest", None)
    if hidden is None:
        pi = np.float64(cfg["reference"])
    else:
        pi = np.asarray(hidden[cfg["index"]], dtype=np.float64)
    light = 1.0
    if cfg["light"] is not None:
        from lib.environments.ecosystem_env import interactions
        light = float(cfg["light"][interactions.light_index(env)])
    visible = (1.0 - pi * (1.0 - cfg["floor"])) * light
    deep = (1.0 - pi) + cfg["rho"] * pi
    visual = rate * cfg["visual"] * visible / cfg["v_ref"]
    tactile = rate * cfg["tactile"] * deep / cfg["d_ref"]
    other = rate * (1.0 - cfg["visual"] - cfg["tactile"])
    shape = np.shape(visual) if np.ndim(visual) else np.shape(deep)
    return (np.broadcast_to(visual, shape),
            np.broadcast_to(tactile, shape),
            np.broadcast_to(np.float64(other), shape))
