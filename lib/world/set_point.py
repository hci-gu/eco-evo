"""Light-entrained reserve set point (section 271).

A DM with ``appetite: reserve_room`` may carry a ``reserve_set_point`` block:
the top of its reserve room is then a defended level s_T(t) instead of the
full reserve, room = max(s_T B ME - R, 0) (``predation._reserve_room_cap``,
GPU mirror). Below s_T it eats at full appetite as before; above it the room
is closed and eating gains nothing.

s_T follows an endogenous annual clock entrained by photoperiod (salmonid
appetite under excess food: Metcalfe & Thorpe 1992, Bull 1996, Tveiten et
al. 1996, Saether et al. 1996; herring condition following the light at a
constant temperature: Berg et al. 2024; research_notes/"Ljusstyrd aptit"
summary.md and design.md). Its only inputs are the day length L at the site
and the sign of dL/dt:

- the clock's progress g runs from 0 at the onset, L_on on FALLING days, to
  1 at the low, L_low on RISING days, linearly in the photoperiod change
  travelled since the onset ((L_on - L) down to the winter solstice, then
  (L - L_ws) up again);
- recovery: g falls linearly in L from 1 at L_low to 0 at L_top (rising);
- g = 0 from L_top through the summer solstice to L_on.

The member species' target is 1 - depth g; the group's (biomass-weighted,
the other members keep 1) is 1 - member_share depth g. At a fixed latitude
(L, sign dL/dt) carries the day of year: the user accepted that tension for
an entrained clock (2026-11-12, provisional). Nothing here reads a month.

The clock also sets the growth window (form 2, user 2026-11-12): while the
target falls the reserve is released and the material law's surplus term
max(I - C, 0) holds growth near zero anyway (frozen fix98 check), so the
growth branch runs only on the days the target is not falling (multiplier 1,
else 0; ``growth_window_by_day``), and the growth budget pays the somatic
production P inside that window: g = P / (1460 mean_t[w(t) (s_T(t) - u)])
(``budget_gate``). This reopens section 250's refusal of a season for an
open population, for the clock's own window only; the window is the model's,
not a field month count.

The block is inert without the daylight calendar (no latitude): the room is
the full reserve, growth runs all year and the growth budget ignores it, on
both engines.
"""
import math

import numpy as np

from lib.world import daylight

KEY = "reserve_set_point"
FIELDS = ("member_share", "depth", "onset_day_length_h",
          "low_day_length_h", "top_day_length_h")


def settings(spec):
    """The parsed block ``{member_share, depth, onset, low, top, observed}``
    or None without one. Raises ValueError on a malformed block."""
    raw = (spec or {}).get(KEY)
    if raw in (None, "", {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{KEY} must be a mapping, got {raw!r}")
    out = {}
    for key in FIELDS:
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"{KEY}.{key} is required and must be a number") from None
        if not math.isfinite(value):
            raise ValueError(f"{KEY}.{key} must be finite")
        out[key] = value
    if not 0.0 < out["member_share"] <= 1.0:
        raise ValueError(f"{KEY}.member_share must be in (0, 1]")
    if not 0.0 <= out["depth"] < 1.0:
        raise ValueError(f"{KEY}.depth must be in [0, 1)")
    if not out["low_day_length_h"] < out["top_day_length_h"]:
        raise ValueError(f"{KEY}: low_day_length_h must be below top_day_length_h")
    for key in ("onset_day_length_h", "low_day_length_h", "top_day_length_h"):
        if not 0.0 <= out[key] <= 24.0:
            raise ValueError(f"{KEY}.{key} must be in [0, 24] h")
    out["observed"] = bool(raw.get("observed", True))
    if (spec or {}).get("appetite") not in ("reserve_room",):
        raise ValueError(f"{KEY} needs appetite: reserve_room")
    return out


def _day_series(latitude_deg):
    hours = np.asarray(daylight.day_length_hours(latitude_deg), dtype=np.float64)
    slope = (np.roll(hours, -1) - np.roll(hours, 1)) / 2.0
    return hours, slope


def progress_by_day(latitude_deg, onset, low, top):
    """The clock's progress g per day of year (0-based, 365 days)."""
    hours, slope = _day_series(latitude_deg)
    l_ws, l_ss = float(hours.min()), float(hours.max())
    if not l_ws < onset <= l_ss:
        raise ValueError(f"{KEY}: onset_day_length_h {onset} h must lie in "
                         f"({l_ws:.2f}, {l_ss:.2f}] h at latitude {latitude_deg}")
    if not l_ws <= low < top <= l_ss:
        raise ValueError(f"{KEY}: need {l_ws:.2f} <= low_day_length_h < "
                         f"top_day_length_h <= {l_ss:.2f} h at latitude {latitude_deg}")
    down, up = onset - l_ws, low - l_ws
    path = down + up
    g = np.zeros_like(hours)
    for d, (length, rate) in enumerate(zip(hours, slope)):
        if rate < 0.0:
            g[d] = 0.0 if length >= onset else (onset - length) / path
        elif length <= low:
            g[d] = (down + (length - l_ws)) / path
        elif length < top:
            g[d] = 1.0 - (length - low) / (top - low)
    return np.clip(g, 0.0, 1.0)


def group_target_by_day(cfg, latitude_deg):
    """The group's target s_T per day of year."""
    g = progress_by_day(latitude_deg, cfg["onset_day_length_h"],
                        cfg["low_day_length_h"], cfg["top_day_length_h"])
    return 1.0 - cfg["member_share"] * cfg["depth"] * g


def tick_table(cfg, latitude_deg, tick_hours):
    """The group's target per year tick on the daylight calendar, (T,)."""
    by_day = group_target_by_day(cfg, latitude_deg)
    hours = int(tick_hours)
    ticks = daylight.ticks_per_year(hours)
    days = (np.arange(ticks) * hours) // 24
    return by_day[np.minimum(days, by_day.size - 1)]


def growth_window_by_day(cfg, latitude_deg):
    """1 on the days the group target is not falling, else 0 (365 days)."""
    target = group_target_by_day(cfg, latitude_deg)
    falling = np.roll(target, -1) < target - 1e-12
    return np.where(falling, 0.0, 1.0)


def growth_window_table(cfg, latitude_deg, tick_hours):
    """The growth window per year tick on the daylight calendar, (T,)."""
    by_day = growth_window_by_day(cfg, latitude_deg)
    hours = int(tick_hours)
    ticks = daylight.ticks_per_year(hours)
    days = (np.arange(ticks) * hours) // 24
    return by_day[np.minimum(days, by_day.size - 1)]


def budget_gate(spec, latitude_deg, s_closed, maintenance):
    """``(gate, window_target, window_share)`` for the growth budget, or None
    without a block or a latitude. gate = mean_t[w(t) (min(s_T(t), s*) -
    u)], so g = P / (gate 1460) pays P inside the window; window_target is
    the mean target there (the reference fill), window_share the share of
    the year the window covers."""
    cfg = settings(spec)
    if cfg is None or latitude_deg is None:
        return None
    target = np.minimum(group_target_by_day(cfg, float(latitude_deg)), float(s_closed))
    window = growth_window_by_day(cfg, float(latitude_deg))
    share = float(window.mean())
    if share <= 0.0:
        raise ValueError(f"{KEY}: the target never stops falling - no growth window")
    gate = float((window * np.maximum(target - maintenance, 0.0)).mean())
    return gate, float(target[window > 0].mean()), share


def mean_target(spec, latitude_deg):
    """The annual mean of the group's target, or None without a block or a
    latitude (section 271: the growth budget's ideal reserve)."""
    cfg = settings(spec)
    if cfg is None or latitude_deg is None:
        return None
    return float(group_target_by_day(cfg, float(latitude_deg)).mean())
