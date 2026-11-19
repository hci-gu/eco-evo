"""Spawning in the box (section 279).

A decision maker with ``appetite: reserve_room`` may carry a ``spawning``
list: one entry per member species (or member set) whose mature fish build
their gonads AND spawn inside the box. research_notes/"Lek i rutan"
summary.md has the evidence per species and design.md the form.

Each entry describes the species' mature fish in the box:

- ``share``: their tonnes per tonne of the group (species share x the share
  of its mature fish that spawn in the box x the mature share).
- ``gamete_energy``: MJ shed as eggs and milt per tonne of those fish and
  year.
- ``reserve_share``: the part of the gamete energy paid out of the stored
  reserve; the rest is replaced by food as it is shed.
- ``spawn_start`` / ``spawn_end``: the group's spawning window, each a day
  length in hours and whether the days are lengthening (``rising``). Only
  the day length and its sign are read (the provisional clock exception of
  section 271.4); no month key.
- ``spawning_days``: how long one fish spawns (absent = the whole window).
  Each fish's start is spread evenly over the window.
- ``fast_cut``: the share of its normal intake a spawning fish does NOT eat
  (0 = it feeds through spawning), and ``fast_loss`` the energy a fasting
  fish then loses per tonne and day (MJ/t/d, field metabolism less
  activity).

Two engine effects, energy only (the booked somatic production P is net of
the spawning loss, so no tonnes are removed):

1. The gametes leave the reserve while they are shed: per tonne of the
   group and day, share x gamete_energy / spawning_days x the share of the
   fish spawning that day (``shed``; integrates to share x gamete_energy a
   year).
2. The fish that spawn do not refill what they lose: the top of the group's
   reserve room is lowered by the spawning fish's accumulated deficit,
   share x (fast_cut x fast_loss + reserve_share x gamete_energy /
   spawning_days) x E[days into its spawning] / ME (``cut``). The engines
   keep a single reserve per cell, so a cut in the intake of the spawners
   alone would be eaten back by the others; closing the room is what a
   pooled reserve can carry. A fish's deficit ends with its spawning
   (refill at once after it; the user's choice, 279).

The cut rides on the room table of section 271 (``env.set_point_table``,
both engines); the growth window there stays the set point's own. The growth
budget pays P at the lowered fill (``growth_budget``, the set point's gate
with the cut). Inert without the daylight calendar (no latitude).
"""
import math

import numpy as np

from lib.world import daylight

KEY = "spawning"
YEAR_DAYS = 365
_TAU_SAMPLES = 4000
_FIELDS = ("share", "gamete_energy", "reserve_share", "fast_cut", "fast_loss")


def _number(raw, key, where):
    try:
        value = float(raw.get(key))
    except (TypeError, ValueError):
        raise ValueError(f"{where}.{key} is required and must be a number") from None
    if not math.isfinite(value):
        raise ValueError(f"{where}.{key} must be finite")
    return value


def _point(raw, key, where):
    point = raw.get(key)
    if not isinstance(point, dict):
        raise ValueError(f"{where}.{key} must be a mapping "
                         "{day_length_h: hours, rising: true|false}")
    hours = _number(point, "day_length_h", f"{where}.{key}")
    if not 0.0 <= hours <= 24.0:
        raise ValueError(f"{where}.{key}.day_length_h must be in [0, 24] h")
    rising = point.get("rising")
    if not isinstance(rising, bool):
        raise ValueError(f"{where}.{key}.rising must be true or false")
    return hours, rising


def settings(spec):
    """The parsed list of entries, or None without one. Raises ValueError
    on a malformed block."""
    raw = (spec or {}).get(KEY)
    if raw in (None, "", [], {}):
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{KEY} must be a list of entries, got {raw!r}")
    if (spec or {}).get("appetite") != "reserve_room":
        raise ValueError(f"{KEY} needs appetite: reserve_room (the cut closes "
                         "the reserve room, section 279)")
    out = []
    for n, entry in enumerate(raw):
        where = f"{KEY}[{n}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a mapping")
        item = {key: _number(entry, key, where) for key in _FIELDS}
        item["name"] = str(entry.get("name", f"entry {n}"))
        if not 0.0 < item["share"] <= 1.0:
            raise ValueError(f"{where}.share must be in (0, 1]")
        if item["gamete_energy"] < 0.0 or item["fast_loss"] < 0.0:
            raise ValueError(f"{where}: gamete_energy and fast_loss must be >= 0")
        for key in ("reserve_share", "fast_cut"):
            if not 0.0 <= item[key] <= 1.0:
                raise ValueError(f"{where}.{key} must be in [0, 1]")
        item["spawn_start"] = _point(entry, "spawn_start", where)
        item["spawn_end"] = _point(entry, "spawn_end", where)
        days = entry.get("spawning_days")
        if days in (None, ""):
            item["spawning_days"] = None
        else:
            item["spawning_days"] = _number(entry, "spawning_days", where)
            if item["spawning_days"] <= 0.0:
                raise ValueError(f"{where}.spawning_days must be > 0")
        out.append(item)
    total = sum(item["share"] for item in out)
    if total > 1.0 + 1e-9:
        raise ValueError(f"{KEY}: the entries' shares sum to {total:.4g} > 1")
    return out


def day_of(latitude_deg, hours, rising):
    """The 0-based day of year on which the day length crosses ``hours`` on
    lengthening (``rising``) or shortening days at the site."""
    length = np.asarray(daylight.day_length_hours(latitude_deg), dtype=np.float64)
    nxt = np.roll(length, -1)
    l_min, l_max = float(length.min()), float(length.max())
    if not l_min <= hours <= l_max:
        raise ValueError(f"{KEY}: day length {hours} h is outside "
                         f"[{l_min:.2f}, {l_max:.2f}] h at latitude {latitude_deg}")
    if rising:
        hit = np.flatnonzero((length <= hours) & (nxt > hours))
    else:
        hit = np.flatnonzero((length >= hours) & (nxt < hours))
    if hit.size == 0:          # exactly the extreme: the solstice itself
        return int(np.argmin(length) if rising else np.argmax(length))
    return int((hit[0] + 1) % YEAR_DAYS)


def _occupancy(width, days):
    """Per day offset x in [0, 365) from the window start: the share of the
    fish spawning (P) and E[days into the spawning; 0 outside] (E), each
    fish's start uniform on [0, width - days]."""
    taus = (np.linspace(0.0, width - days, _TAU_SAMPLES) if width > days
            else np.zeros(1))
    x = np.arange(YEAR_DAYS, dtype=np.float64) + 0.5
    elapsed = x[:, None] - taus[None, :]
    inside = (elapsed >= 0.0) & (elapsed < days)
    share = inside.mean(axis=1)
    depth = np.where(inside, elapsed, 0.0).mean(axis=1)
    return share, depth


def entry_by_day(entry, latitude_deg, max_reserve):
    """``(cut, shed)`` per day of year (365,) for one entry: the room cut as
    a fraction of ME and the gamete energy shed, MJ per tonne of the group
    and day."""
    start = day_of(latitude_deg, *entry["spawn_start"])
    end = day_of(latitude_deg, *entry["spawn_end"])
    width = float((end - start) % YEAR_DAYS) or float(YEAR_DAYS)
    days = entry["spawning_days"]
    days = width if days is None else float(days)
    if days > width + 1e-9:
        raise ValueError(f"{KEY} '{entry['name']}': spawning_days {days:g} is "
                         f"longer than the window ({width:g} d)")
    share, depth = _occupancy(width, days)
    gamete_rate = entry["gamete_energy"] / days
    deficit_rate = (entry["fast_cut"] * entry["fast_loss"]
                    + entry["reserve_share"] * gamete_rate)
    cut = entry["share"] * deficit_rate * depth / float(max_reserve)
    shed = entry["share"] * gamete_rate * share
    order = (np.arange(YEAR_DAYS) - start) % YEAR_DAYS
    return cut[order], shed[order]


def group_by_day(spec, latitude_deg):
    """``(cut, shed)`` per day of year summed over the entries, or None
    without a block or a latitude."""
    entries = settings(spec)
    if entries is None or latitude_deg is None:
        return None
    max_reserve = float((spec or {}).get("max_energy_reserve") or 0.0)
    if max_reserve <= 0.0:
        raise ValueError(f"{KEY} needs max_energy_reserve > 0")
    cut = np.zeros(YEAR_DAYS)
    shed = np.zeros(YEAR_DAYS)
    for entry in entries:
        c, s = entry_by_day(entry, float(latitude_deg), max_reserve)
        cut += c
        shed += s
    return cut, shed


def cut_by_day(spec, latitude_deg):
    """The room cut per day of year, or None (growth budget, section 279)."""
    tables = group_by_day(spec, latitude_deg)
    return None if tables is None else tables[0]


def tick_tables(spec, latitude_deg, tick_hours):
    """``(cut, shed)`` per year tick: the cut as a fraction of ME and the
    gamete energy shed in MJ per tonne of the group and TICK."""
    tables = group_by_day(spec, latitude_deg)
    if tables is None:
        return None
    cut, shed = tables
    hours = int(tick_hours)
    ticks = daylight.ticks_per_year(hours)
    days = np.minimum((np.arange(ticks) * hours) // 24, YEAR_DAYS - 1)
    return cut[days], shed[days] * hours / 24.0
