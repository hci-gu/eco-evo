"""Daylight: a calendar, the sun's elevation, and what light drives.

Two things depend on it: the attack rate of visual predators (section
137) and the growth of light-limited primary producers (section 138,
see "Light-limited growth" below).

Visual predators (herring on zooplankton is the case that motivated
this) detect prey far better in light than in darkness. The light
dependence is put on the PREDATOR side, as a per-(predator, prey)
modulation of the Holling attack rate ``a``; the prey's response (diel
vertical migration, i.e. hiding by day) is left to the policies, which
get the light level as one extra observation channel. Section 137.

The model
---------
For a pair with ``dark_ratio`` rho and ``light_threshold_deg`` theta:

    f(elev) = 1 / (1 + exp(-(elev - theta) / TRANSITION_DEG))
    F(t)    = mean of f over the tick (sub-stepped, f is very nonlinear)
    m(t)    = (rho + (1 - rho) * F(t)) / annual mean of the numerator
    a(t)    = a_library * m(t)

``f`` uses the sun's elevation rather than an irradiance in lux: a
fish's visual range grows roughly with log(light) and saturates at
twilight levels (Aksnes & Giske 1993; Aksnes & Utne 1997), so over a
day ``f`` is ~1 in daylight, ~0 at night and changes during twilight -
which the elevation describes directly. Cloud cover moves the
transition by a degree or two and is ignored.

The library's ``max_intake_rate`` is read as the ANNUAL MEAN attack
rate: m(t) averages to exactly 1 over the year, so the calibrated
budgets (growth_budget, budget_gate) still hold on average and only the
distribution over the day and the season is new.

The calendar
------------
Tick k of the year starts at hour ``k * tick_hours`` after 1 January
00:00 local SOLAR time (noon = sun at its highest; longitude and the
equation of time are ignored). 365 * 24 = 8760 h is divisible by every
allowed tick length (1-6 h), so a year is a whole number of ticks and
the light schedule is exactly periodic. A world starts at ``start_tick``
and tick t of the run is year tick ``(start_tick + t) % ticks_per_year``.

Settings live in the project manifest:

    simulation_settings:
      daylight:
        enabled: true
        latitude_deg: 58.1
        start_day_of_year: random   # or 1..365

and per pair in fg_library.yaml (both optional, both dimensionless and
tick-independent):

    pelagic_fish_preys_on_zooplankton:
      dark_ratio: 0.1            # a_dark / a_light; absent or 1 = no effect
      light_threshold_deg: -8.0  # sun elevation of half detection

Light-limited growth (section 138)
----------------------------------
A non-decision maker with ``light_saturation`` (I_k, umol photons
m-2 s-1; one value, or 12 monthly values for photoacclimation) grows at

    r(t) = growth_rate * P(t) / P_ref
    P(t) = mean over the tick and over the mixed layer [0, z_mix] of
           tanh(I(z) / I_k)                         (Jassby & Platt 1976)
    I(z) = I_0 * exp(-Kd z)
    I_0  = clear-sky PAR (Haurwitz 1945 global radiation x PAR fraction
           x umol per J) x the month's cloud transmission

P_ref is the daily mean of P on ``light_reference_day`` (default 105,
mid April): growth_rate is the literature rate measured in April
(section 131), so the factor is 1 on the day it was measured - NOT
1 on average over the year, unlike the attack-rate multiplier. At
night P = 0 and the population does not grow at all. z_mix (12 monthly
values) is what makes winter dark for the algae: they are mixed far
below the euphotic zone, which the sun's elevation alone does not see.
The site's light climate lives next to the latitude in the manifest:

        light_attenuation_per_m: 0.2           # Kd(PAR)
        mixed_layer_depth_m: [12 values, Jan..Dec]
        cloud_transmission: [12 values]        # actual / clear-sky
"""

from functools import lru_cache
import math

import numpy as np

from lib.world.tick_time import (
    DAYS_PER_YEAR,
    HOURS_PER_DAY,
    resolve_tick_hours,
)

HOURS_PER_YEAR = DAYS_PER_YEAR * HOURS_PER_DAY

# Width (degrees of sun elevation) of the logistic transition from dark
# to light. 10 % -> 90 % detection spans about +-4.4 deg around the
# threshold, i.e. roughly the 30-60 min of twilight at 58 N.
TRANSITION_DEG = 2.0

# Threshold used when a modulated pair does not give its own, and for
# the light observation channel: the end of civil twilight.
DEFAULT_THRESHOLD_DEG = -6.0

# Sub-steps per hour when averaging f over a tick (10 min).
SUBSTEPS_PER_HOUR = 6

# Earth's axial tilt, for the solar declination.
AXIAL_TILT_DEG = 23.44

# Clear-sky global radiation, Haurwitz (1945):
#   G = 1098 W m-2 * sin(elev) * exp(-0.057 / sin(elev)).
HAURWITZ_W_M2 = 1098.0
HAURWITZ_EXP = 0.057
# PAR share of global radiation and its photon flux per joule.
PAR_FRACTION = 0.45
UMOL_PER_JOULE_PAR = 4.57

# Default day of year at which a light-limited growth_rate applies.
DEFAULT_LIGHT_REFERENCE_DAY = 105

# Mid-points of the depth integration over the mixed layer.
MIXED_LAYER_LAYERS = 40

MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MONTH_START_DAY = np.cumsum((0,) + MONTH_DAYS[:-1])
_MONTH_MID_DAY = _MONTH_START_DAY + np.array(MONTH_DAYS) / 2.0
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def ticks_per_year(tick_hours):
    """Whole ticks in a 365-day year; exact for every allowed tick."""
    hours = resolve_tick_hours(tick_hours)
    if HOURS_PER_YEAR % hours:
        raise ValueError(
            f"A {hours} h tick does not divide the year; the daylight "
            "calendar needs a whole number of ticks per year")
    return HOURS_PER_YEAR // hours


def _declination_rad(day_number):
    """Solar declination for a continuous day number (1 = 1 Jan 00:00)."""
    return np.radians(AXIAL_TILT_DEG) * np.sin(
        2.0 * np.pi * (284.0 + np.asarray(day_number, dtype=np.float64))
        / DAYS_PER_YEAR)


def solar_elevation_deg(latitude_deg, hours_since_new_year):
    """Sun elevation in degrees at local solar time.

    ``hours_since_new_year`` may be an array. Declination from the
    standard approximation 23.44 * sin(2 pi (284 + n) / 365) with a
    continuous day number n, so it moves smoothly within a day.
    """
    t = np.asarray(hours_since_new_year, dtype=np.float64)
    declination = _declination_rad(1.0 + t / HOURS_PER_DAY)
    hour_angle = np.radians(15.0 * (np.mod(t, HOURS_PER_DAY) - 12.0))
    phi = math.radians(float(latitude_deg))
    sin_elev = (math.sin(phi) * np.sin(declination)
                + math.cos(phi) * np.cos(declination) * np.cos(hour_angle))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))


def detection(elevation_deg, threshold_deg):
    """f(elev): the fraction of full visual detection, in (0, 1)."""
    z = (np.asarray(elevation_deg, dtype=np.float64)
         - float(threshold_deg)) / TRANSITION_DEG
    return 0.5 * (1.0 + np.tanh(0.5 * z))   # logistic, overflow-free


@lru_cache(maxsize=64)
def _light_schedule_cached(latitude_deg, tick_hours, threshold_deg):
    hours = resolve_tick_hours(tick_hours)
    n_ticks = ticks_per_year(hours)
    substeps = hours * SUBSTEPS_PER_HOUR
    offsets = (np.arange(substeps) + 0.5) * (hours / substeps)
    times = np.arange(n_ticks)[:, None] * hours + offsets[None, :]
    light = detection(solar_elevation_deg(latitude_deg, times),
                      threshold_deg).mean(axis=1)
    light.setflags(write=False)
    return light


def light_schedule(latitude_deg, tick_hours,
                   threshold_deg=DEFAULT_THRESHOLD_DEG):
    """F per year tick, shape (ticks_per_year,), values in [0, 1]."""
    return _light_schedule_cached(float(latitude_deg),
                                  int(resolve_tick_hours(tick_hours)),
                                  float(threshold_deg))


def attack_multiplier(latitude_deg, tick_hours, dark_ratio,
                      threshold_deg=DEFAULT_THRESHOLD_DEG):
    """m(t) per year tick for one pair; its annual mean is exactly 1.

    ``dark_ratio`` >= 1 (or invalid) means no light dependence and
    returns all ones - exactly, not via the normalisation, so an
    unmodulated pair stays bit-identical.
    """
    n_ticks = ticks_per_year(tick_hours)
    rho = float(dark_ratio)
    if not math.isfinite(rho) or rho >= 1.0:
        return np.ones(n_ticks, dtype=np.float64)
    rho = max(0.0, rho)
    raw = rho + (1.0 - rho) * light_schedule(latitude_deg, tick_hours,
                                             threshold_deg)
    mean = float(raw.mean())
    if mean <= 0.0:
        # Polar night all year with rho = 0: the pair can never feed.
        return np.zeros(n_ticks, dtype=np.float64)
    return raw / mean


def clear_sky_par(elevation_deg):
    """Surface PAR (umol photons m-2 s-1) under a clear sky."""
    sin_e = np.sin(np.radians(np.asarray(elevation_deg, dtype=np.float64)))
    safe = np.maximum(sin_e, 1e-6)
    ghi = np.where(sin_e > 0.0,
                   HAURWITZ_W_M2 * safe * np.exp(-HAURWITZ_EXP / safe), 0.0)
    return ghi * PAR_FRACTION * UMOL_PER_JOULE_PAR


def monthly_value(values, day_of_year_0):
    """Interpolate 12 mid-month values to a (0-based, fractional) day.

    Circular, so December runs smoothly into January.
    """
    values = np.asarray(values, dtype=np.float64)
    xp = np.concatenate(([_MONTH_MID_DAY[-1] - DAYS_PER_YEAR],
                         _MONTH_MID_DAY,
                         [_MONTH_MID_DAY[0] + DAYS_PER_YEAR]))
    fp = np.concatenate(([values[-1]], values, [values[0]]))
    return np.interp(np.mod(day_of_year_0, DAYS_PER_YEAR), xp, fp)


def _photosynthesis(latitude_deg, hours, kd, zmix, cloud, ik):
    """Mean tanh(I(z)/I_k) over the mixed layer at the given hours."""
    day = np.asarray(hours, dtype=np.float64) / HOURS_PER_DAY
    surface = (clear_sky_par(solar_elevation_deg(latitude_deg, hours))
               * monthly_value(cloud, day))
    depth = monthly_value(zmix, day)
    fractions = (np.arange(MIXED_LAYER_LAYERS) + 0.5) / MIXED_LAYER_LAYERS
    light = surface[..., None] * np.exp(-kd * depth[..., None] * fractions)
    saturation = (monthly_value(ik, day) if isinstance(ik, tuple)
                  else np.full_like(day, ik))
    return np.tanh(light / saturation[..., None]).mean(axis=-1)


@lru_cache(maxsize=16)
def _growth_schedule_cached(latitude_deg, tick_hours, kd, zmix, cloud, ik,
                            reference_day):
    hours = resolve_tick_hours(tick_hours)
    n_ticks = ticks_per_year(hours)
    substeps = hours * SUBSTEPS_PER_HOUR
    offsets = (np.arange(substeps) + 0.5) * (hours / substeps)
    times = np.arange(n_ticks)[:, None] * hours + offsets[None, :]
    p = _photosynthesis(latitude_deg, times, kd, zmix, cloud, ik).mean(axis=1)
    day_steps = HOURS_PER_DAY * SUBSTEPS_PER_HOUR
    ref_times = ((reference_day - 1) * HOURS_PER_DAY
                 + (np.arange(day_steps) + 0.5) / SUBSTEPS_PER_HOUR)
    p_ref = float(_photosynthesis(latitude_deg, ref_times, kd, zmix, cloud,
                                  ik).mean())
    if p_ref <= 0.0:
        raise ValueError(f"no light on the reference day {reference_day}: "
                         "light-limited growth cannot be normalised")
    out = p / p_ref
    out.setflags(write=False)
    return out


def growth_light_schedule(latitude_deg, tick_hours, light_climate,
                          light_saturation,
                          reference_day=DEFAULT_LIGHT_REFERENCE_DAY):
    """Growth multiplier P(t)/P_ref per year tick, shape (ticks_per_year,).

    ``light_climate`` is the dict ``parse_settings`` returns under
    ``"light_climate"``. 1 means the literature growth_rate; the daily
    mean on ``reference_day`` is exactly 1.
    """
    return _growth_schedule_cached(
        float(latitude_deg), int(resolve_tick_hours(tick_hours)),
        float(light_climate["light_attenuation_per_m"]),
        tuple(float(v) for v in light_climate["mixed_layer_depth_m"]),
        tuple(float(v) for v in light_climate["cloud_transmission"]),
        (tuple(float(v) for v in light_saturation)
         if isinstance(light_saturation, (list, tuple))
         else float(light_saturation)),
        int(reference_day))


def growth_light_settings(params):
    """(I_k, reference day) of a species, or None if not light-limited.

    I_k is a float, or a tuple of 12 monthly values (Jan..Dec).
    """
    raw = params.get("light_saturation", None)
    if isinstance(raw, (list, tuple)):
        try:
            ik = tuple(float(v) for v in raw)
        except (TypeError, ValueError):
            raise ValueError("light_saturation must be a number or 12 "
                             "monthly numbers")
        if len(ik) != 12 or not all(math.isfinite(v) and v > 0 for v in ik):
            raise ValueError("monthly light_saturation needs 12 positive "
                             f"values, got {list(raw)}")
    else:
        try:
            ik = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(ik) or ik <= 0.0:
            return None
    day = params.get("light_reference_day", None)
    try:
        day = DEFAULT_LIGHT_REFERENCE_DAY if day in (None, "") else int(day)
    except (TypeError, ValueError):
        day = DEFAULT_LIGHT_REFERENCE_DAY
    if day == 0:
        day = DEFAULT_LIGHT_REFERENCE_DAY   # the FG editor's empty field
    if not 1 <= day <= DAYS_PER_YEAR:
        raise ValueError(f"light_reference_day must be 1-365, got {day}")
    return ik, day


def _parse_light_climate(block):
    """The optional site light climate, validated, or None if absent."""
    keys = ("light_attenuation_per_m", "mixed_layer_depth_m",
            "cloud_transmission")
    if not any(k in block for k in keys):
        return None
    missing = [k for k in keys if k not in block]
    if missing:
        raise ValueError("daylight light climate incomplete, missing: "
                         + ", ".join(missing))
    try:
        kd = float(block["light_attenuation_per_m"])
    except (TypeError, ValueError):
        raise ValueError("light_attenuation_per_m must be a number")
    if not kd > 0.0:
        raise ValueError(f"light_attenuation_per_m must be > 0, got {kd}")
    out = {"light_attenuation_per_m": kd}
    for key, lo, hi in (("mixed_layer_depth_m", 0.0, None),
                        ("cloud_transmission", 0.0, 1.0)):
        values = block[key]
        try:
            values = [float(v) for v in values]
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be 12 numbers (Jan..Dec)")
        if len(values) != 12:
            raise ValueError(f"{key} must have 12 monthly values, got "
                             f"{len(values)}")
        if any(not (v > lo and (hi is None or v <= hi)) for v in values):
            bound = f"in ({lo:g}, {hi:g}]" if hi is not None else f"> {lo:g}"
            raise ValueError(f"every {key} value must be {bound}")
        out[key] = values
    return out


def parse_settings(project):
    """``simulation_settings.daylight`` -> dict, or None when disabled.

    Returns ``{"latitude_deg": float, "start_day_of_year": int | None,
    "light_climate": dict | None}`` where a None start means random and
    the light climate (Kd, monthly mixed-layer depth and cloud
    transmission) is only needed by light-limited growth. Raises ValueError on an enabled
    block that cannot be honoured, rather than silently running
    without light.
    """
    settings = ((project or {}).get("simulation_settings") or {})
    block = settings.get("daylight") if isinstance(settings, dict) else None
    if not isinstance(block, dict) or not block.get("enabled", False):
        return None
    try:
        latitude = float(block.get("latitude_deg"))
    except (TypeError, ValueError):
        raise ValueError("simulation_settings.daylight.latitude_deg is "
                         "required when daylight is enabled")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"daylight latitude_deg {latitude} outside "
                         "[-90, 90]")
    start = block.get("start_day_of_year", "random")
    if start in (None, "", "random"):
        start_day = None
    else:
        try:
            start_day = int(start)
        except (TypeError, ValueError):
            raise ValueError("daylight start_day_of_year must be 1-365 "
                             f"or 'random', got {start!r}")
        if not 1 <= start_day <= DAYS_PER_YEAR:
            raise ValueError("daylight start_day_of_year must be 1-365, "
                             f"got {start_day}")
    return {"latitude_deg": latitude, "start_day_of_year": start_day,
            "light_climate": _parse_light_climate(block)}


def start_tick(start_day_of_year, tick_hours, rng=None):
    """Year tick a world starts at.

    A fixed day starts at the tick containing that day's midnight; a
    random start (``start_day_of_year`` None) is uniform over every
    tick of the year, drawn from ``rng`` (a numpy Generator) or the
    global numpy RNG.
    """
    hours = resolve_tick_hours(tick_hours)
    n_ticks = ticks_per_year(hours)
    if start_day_of_year is None:
        if rng is not None:
            return int(rng.integers(0, n_ticks))
        return int(np.random.randint(0, n_ticks))
    return int(((int(start_day_of_year) - 1) * HOURS_PER_DAY) // hours)


def runtime_config(settings, tick_hours, rng=None):
    """The dict the loader stores as ``params['daylight']`` on every FG."""
    if settings is None:
        return None
    hours = resolve_tick_hours(tick_hours)
    return {
        "latitude_deg": float(settings["latitude_deg"]),
        "tick_hours": int(hours),
        "start_tick": start_tick(settings["start_day_of_year"], hours, rng),
        # The GPU trainer draws its own per-world start from the world
        # keys when the start is random; a fixed day is used as is.
        "random_start": settings["start_day_of_year"] is None,
        "light_climate": settings.get("light_climate"),
    }


def month_of_day(day_of_year_0):
    """0-based month (0 = January) of a 0-based day of the year."""
    day = int(day_of_year_0) % DAYS_PER_YEAR
    return int(np.searchsorted(_MONTH_START_DAY, day, side="right")) - 1


@lru_cache(maxsize=16)
def _day_length_cached(latitude_deg):
    # Declination at each day's solar noon: day number d0 + 1.5.
    declination = _declination_rad(np.arange(DAYS_PER_YEAR) + 1.5)
    phi = math.radians(latitude_deg)
    cos_h0 = np.clip(-math.tan(phi) * np.tan(declination), -1.0, 1.0)
    hours = 2.0 * np.degrees(np.arccos(cos_h0)) / 15.0
    hours.setflags(write=False)
    return hours


def day_length_hours(latitude_deg):
    """Hours of sun per day, shape (365,), indexed by 0-based day.

    The sun's centre above the geometric horizon (no refraction, no
    twilight), with the declination at the day's solar noon, so the
    value is constant within a day and steps at midnight. 0 in polar
    night, 24 under the midnight sun.
    """
    return _day_length_cached(float(latitude_deg))


def day_length_fraction(latitude_deg, day_of_year_0):
    """A day's sun hours on the year's own scale: 0 = shortest, 1 = longest.

    At the equator (no seasonal range) every day is 0.5.
    """
    hours = day_length_hours(latitude_deg)
    lo, hi = float(hours.min()), float(hours.max())
    if hi - lo < 1e-9:
        return 0.5
    return (float(hours[int(day_of_year_0) % DAYS_PER_YEAR]) - lo) / (hi - lo)


def calendar_at(config, tick):
    """Where tick ``tick`` of a run falls in the year, and its light.

    ``config`` is the loader's ``params['daylight']`` (runtime_config);
    the year tick is ``(start_tick + tick) % ticks_per_year``, exactly
    as the environment's ``light_index``. Returns ``{"year_tick",
    "day_of_year" (1-based), "month" (0-based), "month_name", "light",
    "day_length_h", "day_length_frac"}`` where ``light`` is F, the value
    of the light observation channel this tick, and the day length is
    that day's sun hours, also on the year's 0..1 scale
    (``day_length_fraction``).
    """
    hours = resolve_tick_hours(config["tick_hours"])
    n_ticks = ticks_per_year(hours)
    year_tick = (int(config.get("start_tick", 0)) + int(tick)) % n_ticks
    day0 = (year_tick * hours) // HOURS_PER_DAY
    month = month_of_day(day0)
    latitude = config["latitude_deg"]
    light = light_schedule(latitude, hours)[year_tick]
    return {"year_tick": int(year_tick), "day_of_year": int(day0) + 1,
            "month": month, "month_name": MONTH_NAMES[month],
            "light": float(light),
            "day_length_h": float(day_length_hours(latitude)[day0]),
            "day_length_frac": day_length_fraction(latitude, day0)}


def pair_settings(inter_def):
    """(dark_ratio, threshold_deg) of one interaction, or None if inert."""
    if not isinstance(inter_def, dict):
        return None
    raw = inter_def.get("dark_ratio", None)
    if raw in (None, ""):
        return None
    try:
        rho = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rho) or rho >= 1.0:
        return None
    threshold = inter_def.get("light_threshold_deg", None)
    try:
        threshold = (DEFAULT_THRESHOLD_DEG if threshold in (None, "")
                     else float(threshold))
    except (TypeError, ValueError):
        threshold = DEFAULT_THRESHOLD_DEG
    return max(0.0, rho), threshold
