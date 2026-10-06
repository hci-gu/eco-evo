"""Water temperature on the daylight calendar: Q10 metabolism (section 143).

The model had no temperature: every group paid the same
``resting_metabolism`` all year, so an ectotherm paid its summer cost in
January. Herring then ran a ~9 MJ/t/tick deficit through the winter and
lost ~13 % of the stock to starvation, which no herring stock shows
(``reports/Vintermetabolism sill och tumlare.md``).

The model
---------
A decision maker with ``metabolism_q10`` pays

    resting_metabolism(t) = resting_metabolism * m(t)
    m(t) = q10 ** ((T(t) - t_ref) / 10)

where T(t) is the water temperature of the group's layer at the tick's
mid-point, interpolated from 12 monthly means (circular, mid-month
anchors, as the daylight light climate). Every action cost (rest, eat,
move) is a multiple of ``resting_metabolism``, so all of them follow.

``metabolism_t_ref`` is the temperature the library's
``resting_metabolism`` was calibrated at. It is either a number (deg C)
or the keyword ``annual_mean``: the library value is the annual mean
cost, i.e. m(t) is normalised to average exactly 1 over the year (the
convention of the daylight attack-rate multiplier). That is slightly
different from ``t_ref`` = the annual mean temperature, since q10**x is
convex.

Endotherms (porpoises) get no term: their field metabolic rate is flat
over 0-20 C (Rojano-Donate et al. 2018). Only decision makers carry a
metabolism, so a non decision maker with ``metabolism_q10`` is an error.
Consumption is NOT scaled here; attack-rate temperature dependence is a
separate, weaker effect (E ~0.4 eV, Rall et al. 2012) left for later.

The calendar
------------
The term needs the daylight calendar (``simulation_settings.daylight``)
and reads the same year tick: one start day, one random start per world,
no second calendar to keep in step. It adds no observation channel, so
checkpoints stay valid.

Settings
--------
The site's temperatures live in the project manifest, next to the
daylight block:

    simulation_settings:
      temperature:
        enabled: true
        layers:                     # 12 monthly means (deg C), Jan..Dec
          mean_0_50m: [6.3, 5.4, ...]
          at_50m: [7.7, 7.2, ...]
        group_layers:               # which layer each group lives in
          pelagic_fish: mean_0_50m

and the biology in fg_library.yaml (both dimensionless or deg C, i.e.
tick-independent):

    pelagic_fish:
      metabolism_q10: 2.2
      metabolism_t_ref: 14.0      # or annual_mean
"""

import math

import numpy as np

from lib.world import daylight
from lib.world.tick_time import HOURS_PER_DAY, resolve_tick_hours

ANNUAL_MEAN = "annual_mean"


def _monthly(values, what):
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be 12 numbers (Jan..Dec)")
    if len(out) != 12:
        raise ValueError(f"{what} must have 12 monthly values, got {len(out)}")
    if not all(math.isfinite(v) and -5.0 <= v <= 40.0 for v in out):
        raise ValueError(f"every {what} value must be a water temperature "
                         "in [-5, 40] deg C")
    return out


def parse_settings(project):
    """``simulation_settings.temperature`` -> dict, or None when disabled.

    Returns ``{"layers": {name: [12 floats]}, "group_layers": {fg_id:
    name}}``. Raises ValueError on an enabled block that cannot be
    honoured - including one without an enabled daylight calendar, whose
    year tick it reads - rather than silently running at constant cost.
    """
    settings = ((project or {}).get("simulation_settings") or {})
    block = settings.get("temperature") if isinstance(settings, dict) else None
    if not isinstance(block, dict) or not block.get("enabled", False):
        return None
    if daylight.parse_settings(project) is None:
        raise ValueError("simulation_settings.temperature needs the daylight "
                         "calendar: enable simulation_settings.daylight too")
    layers = block.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("simulation_settings.temperature.layers must map a "
                         "layer name to 12 monthly temperatures")
    layers = {str(name): _monthly(values, f"temperature layer '{name}'")
              for name, values in layers.items()}
    group_layers = block.get("group_layers") or {}
    if not isinstance(group_layers, dict):
        raise ValueError("simulation_settings.temperature.group_layers must "
                         "map a group id to a layer name")
    group_layers = {str(fid): str(name) for fid, name in group_layers.items()}
    unknown = sorted({n for n in group_layers.values() if n not in layers})
    if unknown:
        raise ValueError("temperature group_layers name unknown layers: "
                         + ", ".join(unknown))
    return {"layers": layers, "group_layers": group_layers}


def species_settings(params):
    """(q10, t_ref) of a species, or None if its metabolism is constant.

    ``t_ref`` is a float (deg C) or ``ANNUAL_MEAN``. A q10 of exactly 1
    is inert and returns None, so such a group stays bit-identical.
    """
    raw = params.get("metabolism_q10", None)
    if raw in (None, ""):
        return None
    try:
        q10 = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"metabolism_q10 must be a number, got {raw!r}")
    if q10 == 0.0:
        return None                      # the FG editor's empty field
    if not math.isfinite(q10) or q10 < 0.0:
        raise ValueError(f"metabolism_q10 must be > 0, got {q10}")
    if q10 == 1.0:
        return None
    t_ref = params.get("metabolism_t_ref", None)
    if t_ref in (None, ""):
        raise ValueError("metabolism_q10 needs metabolism_t_ref: a "
                         f"temperature (deg C) or '{ANNUAL_MEAN}'")
    if t_ref == ANNUAL_MEAN:
        return q10, ANNUAL_MEAN
    try:
        t_ref = float(t_ref)
    except (TypeError, ValueError):
        raise ValueError("metabolism_t_ref must be deg C or "
                         f"'{ANNUAL_MEAN}', got {t_ref!r}")
    if not math.isfinite(t_ref):
        raise ValueError(f"metabolism_t_ref must be finite, got {t_ref}")
    return q10, t_ref


def tick_temperatures(monthly_c, tick_hours):
    """T per year tick (deg C) at each tick's mid-point."""
    hours = resolve_tick_hours(tick_hours)
    n_ticks = daylight.ticks_per_year(hours)
    day = (np.arange(n_ticks) * hours + 0.5 * hours) / HOURS_PER_DAY
    return daylight.monthly_value(monthly_c, day)


def metabolism_multiplier(monthly_c, tick_hours, q10, t_ref):
    """m(t) per year tick, shape (ticks_per_year,).

    With ``t_ref`` = ``ANNUAL_MEAN`` the annual mean is exactly 1.
    """
    temps = tick_temperatures(monthly_c, tick_hours)
    if t_ref == ANNUAL_MEAN:
        raw = np.power(float(q10), temps / 10.0)
        return raw / float(raw.mean())
    return np.power(float(q10), (temps - float(t_ref)) / 10.0)


def runtime_config(settings, fg_id):
    """The dict the loader stores as ``params['temperature']`` on ``fg_id``,
    or None when the group has no layer."""
    if settings is None:
        return None
    layer = settings["group_layers"].get(fg_id)
    if layer is None:
        return None
    return {"layer": layer, "monthly_c": list(settings["layers"][layer])}
