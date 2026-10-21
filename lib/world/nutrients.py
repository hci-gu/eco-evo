"""A nutrient pool for light-limited producers (section 172).

Why
---
Section 171: the phytoplankton had no spring bloom. Its growth was
logistic toward a constant K, so nothing ended a bloom and the summer,
with the most light, kept the highest stock - the opposite of the
Skagerrak, where nitrate is used up by April and July is the minimum
(SHARK A17: surface NO2+NO3 5 umol/l in Jan-Feb, 0.1 from May to Sep).

The model
---------
A producer with ``nutrient_half_saturation`` carries a dissolved
nutrient pool N per cell, in the SAME units as its biomass (t/km2 of
producer it could become), and grows at

    dB = min(r L(t) B N / (N + k_N Y M(t)),  N,  K - B)
    N -= dB

instead of logistically: ``r`` is its derived growth_rate, ``L(t)`` the
light factor of section 138, ``k_N`` the half-saturation (umol N/l),
``Y`` = ``nutrient_yield`` (t/km2 of producer per umol/l of nutrient in
a 1 m water column) and ``M(t)`` the mixed-layer depth (m) of the
daylight light climate. K stays only as a ceiling. The pool is supplied
from below, as in Evans & Parslow (1985):

    dN/dt = Y [ m (N0(t) - c) + max(dM/dt, 0) N0(t) - max(-dM/dt, 0) c ]
    c     = N / (Y M)            (the mixed-layer concentration)

``m`` is the diffusive exchange across the pycnocline (m/day), N0(t)
the concentration just below the mixed layer (12 monthly values), a
deepening layer entrains N0 water and a shoaling one leaves its own
water behind. Per tick that is affine in N, ``N' = N (1 - a(t)) +
Y b(t)``, and both tables are built once per year tick here.

When the pool runs dry a producer also loses ``stress_loss_max`` x
c(1 - f)/(f + c) of its biomass per tick (section 175: lysis and
aggregate sinking after nutrient exhaustion, ERSEM's shape; exported,
not given back to the pool). It is what ends the bloom and empties the
summer, and lets the autumn entrainment start a second one.

Grazing returns a fraction ``grazing_regeneration`` of the producer
biomass it removes to the pool (excretion and sloppy feeding,
remineralised in the mixed layer); the rest leaves the cycle through
the grazers. That fraction is the only closure: the budget is open
(supply in, export out), which is what fixes the summer level.

The engine carries N in the producer's ``energy_reserve`` slot, which a
non decision maker never used: the currents and the boundary already
move it with the water, conservatively, on both engines. Nothing that
reads a reserve as energy reads a non decision maker's (observations,
local reward, the mass balance and the movement clamp are all per
decision maker); the places that would scale or zero it - predation,
impact mortality, the empty-cell sweep - skip a nutrient producer. The
per-group "energy" in the inference and training plots
(``energy_reserve.sum()`` for every group) therefore shows the pool for
a nutrient producer; for a non decision maker that number had no
meaning before.

Settings
--------
The site lives in the manifest, next to the daylight block whose light
climate supplies M(t):

    simulation_settings:
      nutrients:
        enabled: true
        sub_mixed_layer_umol_l: [12 values]   # N0(t), Jan..Dec
        surface_umol_l: [12 values]           # initial c, by start day
        exchange_m_per_day: 0.1               # m, or 12 monthly values

and the biology in fg_library.yaml (all tick-independent):

    phytoplankton:
      nutrient_half_saturation: 1.0   # umol N/l
      nutrient_yield: 0.795           # t/km2 per (umol N/l x m)
      grazing_regeneration: 0.3       # fraction of grazed biomass

A producer without ``nutrient_half_saturation`` stays logistic, and
without the manifest block nothing changes at all.

Subsurface layer (section 180)
------------------------------
SHARK A17 has a subsurface chlorophyll maximum from April to August
(15-30 m, 2-4x the surface; in June-July ~90 % of the 0-30 m chlorophyll
lies below the mixed layer), sitting on the nitracline. With

    nutrients:
      subsurface_layer:
        bottom_m: 30                      # > every mixed-layer depth
        light_attenuation_per_m: 0.17     # Kd below the mixed layer
        reservoir_umol_l: [12 values]     # nitrate at the reservoir (50 m)
        reservoir_exchange_m_per_day: 0.3 # m, or 12 monthly values

the water column has two layers, the mixed layer [0, M(t)] and the
subsurface layer [M(t), bottom_m] of thickness H(t), each with its own
producer (``water_layer: mixed`` - the default - or ``subsurface``) and
its own pool, and the nutrients are CHAINED: reservoir -> subsurface
pool -> mixed-layer pool. Per tick (``layer_tables``), with f_up =
max(dM, 0)/H the share of the subsurface layer a deepening mixed layer
entrains and f_dn = max(-dM, 0)/M the share a shoaling one leaves
behind (water AND biomass move), then the diffusive exchange e across
the pycnocline (``exchange_m_per_day``, now between the two layers) and
e_b between the subsurface layer and the reservoir:

    N1  = N (1 - f_dn) + Nd f_up             B1  likewise, Bd1 likewise
    Nd1 = Nd (1 - f_up) + N f_dn
    N'  = N1 (1 - e/M) + Nd1 e/H
    Nd' = Nd1 (1 - e/H - e_b/H) + N1 e/M + Y e_b N_res(t)

which is linear, so it is a 2x2 map per tick. Total new nitrogen is the
one flux e_b (N_res - c_d); each producer's own table then only gives
the half-saturation (k_N Y M for the mixed layer, k_N Y H below) and
keep 1, supply 0. ``sub_mixed_layer_umol_l`` is then the initial
subsurface concentration. Both producers must share one nutrient_yield
(one currency).
"""

import math

import numpy as np

from lib.world import daylight
from lib.world.tick_time import HOURS_PER_DAY, resolve_tick_hours


def _monthly(values, what, lo=0.0):
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be 12 numbers (Jan..Dec)")
    if len(out) != 12:
        raise ValueError(f"{what} must have 12 monthly values, got {len(out)}")
    if not all(math.isfinite(v) and v >= lo for v in out):
        raise ValueError(f"every {what} value must be >= {lo:g}")
    return out


def parse_settings(project):
    """``simulation_settings.nutrients`` -> dict, or None when disabled.

    Raises ValueError on an enabled block that cannot be honoured -
    including one without the daylight light climate, whose mixed-layer
    depths it needs - rather than silently running without nutrients.
    """
    settings = ((project or {}).get("simulation_settings") or {})
    block = settings.get("nutrients") if isinstance(settings, dict) else None
    if not isinstance(block, dict) or not block.get("enabled", False):
        return None
    light = daylight.parse_settings(project)
    if light is None or not light.get("light_climate"):
        raise ValueError("simulation_settings.nutrients needs the daylight "
                         "calendar with its light climate (the mixed-layer "
                         "depths): enable simulation_settings.daylight with "
                         "mixed_layer_depth_m")
    for key in ("sub_mixed_layer_umol_l", "surface_umol_l",
                "exchange_m_per_day"):
        if key not in block:
            raise ValueError(f"simulation_settings.nutrients.{key} is required")
    raw = block["exchange_m_per_day"]
    if isinstance(raw, (list, tuple)):
        # 12 monthly values (summer stratification vs winter storms).
        exchange = _monthly(raw, "nutrients exchange_m_per_day")
    else:
        try:
            exchange = float(raw)
        except (TypeError, ValueError):
            raise ValueError("nutrients exchange_m_per_day must be a number "
                             "or 12 monthly numbers")
        if not (math.isfinite(exchange) and exchange >= 0.0):
            raise ValueError(f"nutrients exchange_m_per_day must be >= 0, "
                             f"got {exchange}")
    subsurface = _parse_subsurface(block.get("subsurface_layer"),
                                   light["light_climate"])
    return {
        "subsurface": subsurface,
        "sub_mixed_layer_umol_l": _monthly(
            block["sub_mixed_layer_umol_l"], "nutrients sub_mixed_layer_umol_l"),
        "surface_umol_l": _monthly(block["surface_umol_l"],
                                   "nutrients surface_umol_l"),
        "exchange_m_per_day": exchange,
        "mixed_layer_depth_m": list(
            light["light_climate"]["mixed_layer_depth_m"]),
    }


def _rate(raw, what):
    """A rate in m/day: one number or 12 monthly numbers."""
    if isinstance(raw, (list, tuple)):
        return _monthly(raw, what)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number or 12 monthly numbers")
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{what} must be >= 0, got {value}")
    return value


def _parse_subsurface(block, light_climate):
    """``nutrients.subsurface_layer`` -> dict, or None (section 180)."""
    if not isinstance(block, dict) or not block.get("enabled", True):
        return None
    for key in ("bottom_m", "light_attenuation_per_m", "reservoir_umol_l",
                "reservoir_exchange_m_per_day"):
        if key not in block:
            raise ValueError(f"nutrients.subsurface_layer.{key} is required")
    try:
        bottom = float(block["bottom_m"])
        kd = float(block["light_attenuation_per_m"])
    except (TypeError, ValueError):
        raise ValueError("subsurface_layer bottom_m and "
                         "light_attenuation_per_m must be numbers")
    deepest = max(float(v) for v in light_climate["mixed_layer_depth_m"])
    if not (math.isfinite(bottom) and bottom > deepest):
        raise ValueError(f"subsurface_layer.bottom_m ({bottom}) must lie "
                         f"below the deepest mixed layer ({deepest} m)")
    if not (math.isfinite(kd) and kd > 0.0):
        raise ValueError(f"subsurface_layer.light_attenuation_per_m must be "
                         f"> 0, got {kd}")
    return {
        "bottom_m": bottom,
        "light_attenuation_per_m": kd,
        "reservoir_umol_l": _monthly(block["reservoir_umol_l"],
                                     "subsurface_layer reservoir_umol_l"),
        "reservoir_exchange_m_per_day": _rate(
            block["reservoir_exchange_m_per_day"],
            "subsurface_layer reservoir_exchange_m_per_day"),
    }


def species_layer(params):
    """'mixed' (default) or 'subsurface': the layer a producer lives in."""
    layer = params.get("water_layer", None) or "mixed"
    if layer not in ("mixed", "subsurface"):
        raise ValueError(f"water_layer must be 'mixed' or 'subsurface', "
                         f"got {layer!r}")
    return layer


def runtime_config(settings, tick_hours):
    """The dict the loader stores as ``params['nutrients']`` on every FG."""
    if settings is None:
        return None
    return dict(settings, tick_hours=int(resolve_tick_hours(tick_hours)))


def species_settings(params):
    """(k_N, Y, rho) of a producer, or None if it grows logistically."""
    raw = params.get("nutrient_half_saturation", None)
    if raw in (None, "", 0, 0.0):
        return None
    values = {}
    for key, default in (("nutrient_half_saturation", None),
                         ("nutrient_yield", None),
                         ("grazing_regeneration", 0.0)):
        value = params.get(key, default)
        if value in (None, ""):
            raise ValueError(f"nutrient_half_saturation needs {key} "
                             "(section 172)")
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {value!r}")
    k_n = values["nutrient_half_saturation"]
    yield_ = values["nutrient_yield"]
    rho = values["grazing_regeneration"]
    if not (math.isfinite(k_n) and k_n > 0.0):
        raise ValueError(f"nutrient_half_saturation must be > 0, got {k_n}")
    if not (math.isfinite(yield_) and yield_ > 0.0):
        raise ValueError(f"nutrient_yield must be > 0, got {yield_}")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"grazing_regeneration must be in [0, 1], got {rho}")
    return k_n, yield_, rho


def stress_settings(params):
    """(L_s, c) of a producer's nutrient-stress loss, or None (section 175).

    Extra loss per tick ``L_s * c (1 - f) / (f + c)`` with f = N/(N +
    half) the growth's nutrient limitation: 0 with nutrients replete,
    L_s when they are exhausted. ``c`` (``stress_loss_shape``, default
    0.1) is ERSEM's lysis curve (0.045 / 0.125 / 0.5 per day at f = 1 /
    0.3 / 0) without its replete baseline, which a dilution growth rate
    already contains.
    """
    raw = params.get("stress_loss_max", None)
    if raw in (None, "", 0, 0.0):
        return None
    try:
        l_s = float(raw)
        shape = float(params.get("stress_loss_shape", 0.1) or 0.1)
    except (TypeError, ValueError):
        raise ValueError("stress_loss_max and stress_loss_shape must be numbers")
    if not 0.0 < l_s < 1.0:
        raise ValueError(f"stress_loss_max must be in (0, 1) per tick, got {l_s}")
    if not (math.isfinite(shape) and shape > 0.0):
        raise ValueError(f"stress_loss_shape must be > 0, got {shape}")
    return l_s, shape


def stress_loss(limitation, l_s, shape):
    """Per-tick stress loss at nutrient limitation ``limitation`` (0..1)."""
    return l_s * shape * (1.0 - limitation) / (limitation + shape)


def _depth(config, hours_since_new_year):
    return daylight.monthly_value(config["mixed_layer_depth_m"],
                                  np.asarray(hours_since_new_year) / HOURS_PER_DAY)


def layer_thickness(config, hours_since_new_year, layer="mixed"):
    """Thickness (m) of a producer's layer: M(t), or bottom_m - M(t)."""
    depth = _depth(config, hours_since_new_year)
    if layer == "subsurface":
        sub = config.get("subsurface")
        if not sub:
            raise ValueError("water_layer 'subsurface' needs "
                             "nutrients.subsurface_layer in the manifest "
                             "(section 180)")
        return sub["bottom_m"] - depth
    return depth


def tick_tables(config, yield_, k_n, layer="mixed"):
    """Per year tick, shape (ticks_per_year,) each:

    ``keep`` = 1 - a(t), ``supply`` = Y b(t) (t/km2 per tick) and
    ``half`` = k_N Y M(t), the half-saturation as a pool size, so the
    tick is ``N' = N * keep + supply`` and growth uses N / (N + half).
    With the subsurface layer (section 180) the transport is
    ``layer_tables``' instead: keep 1, supply 0, and ``half`` is k_N Y
    times the producer's own layer thickness.
    """
    hours = resolve_tick_hours(config["tick_hours"])
    n_ticks = daylight.ticks_per_year(hours)
    start = np.arange(n_ticks) * hours
    if config.get("subsurface") or layer == "subsurface":
        thickness = layer_thickness(config, start + 0.5 * hours, layer)
        return {"keep": np.ones(n_ticks),
                "supply": np.zeros(n_ticks),
                "half": float(k_n) * float(yield_) * thickness}
    m_start = _depth(config, start)
    m_end = _depth(config, start + hours)
    m_mid = _depth(config, start + 0.5 * hours)
    d_m = m_end - m_start
    n0 = daylight.monthly_value(config["sub_mixed_layer_umol_l"],
                                (start + 0.5 * hours) / HOURS_PER_DAY)
    raw = config["exchange_m_per_day"]
    if isinstance(raw, (list, tuple)):
        per_day = daylight.monthly_value(raw, (start + 0.5 * hours) / HOURS_PER_DAY)
    else:
        per_day = float(raw)
    exchange = per_day * hours / HOURS_PER_DAY
    a = (exchange + np.maximum(-d_m, 0.0)) / m_mid
    b = n0 * (exchange + np.maximum(d_m, 0.0))
    return {"keep": np.clip(1.0 - a, 0.0, 1.0),
            "supply": float(yield_) * b,
            "half": float(k_n) * float(yield_) * m_mid}


def layer_tables(config, yield_):
    """The two-layer transport of section 180, per year tick.

    Returns (ticks_per_year,) arrays: ``n_mm, n_md, n_dm, n_dd`` and
    ``n_s`` for the pools (mixed N, subsurface Nd):
        N'  = n_mm N + n_md Nd
        Nd' = n_dm N + n_dd Nd + n_s
    and ``b_mm, b_md, b_dm, b_dd`` for the biomass (no supply term).
    Water conserves: n_mm + n_dm = 1 - f_dn e_b/H, n_md + n_dd = 1 -
    (1 - f_up) e_b/H, the rest goes to the reservoir.
    """
    sub = config["subsurface"]
    hours = resolve_tick_hours(config["tick_hours"])
    n_ticks = daylight.ticks_per_year(hours)
    start = np.arange(n_ticks) * hours
    mid = (start + 0.5 * hours) / HOURS_PER_DAY
    bottom = sub["bottom_m"]
    m_start = _depth(config, start)
    d_m = _depth(config, start + hours) - m_start
    h_start = bottom - m_start
    m_mid = _depth(config, start + 0.5 * hours)
    h_mid = bottom - m_mid
    f_up = np.clip(np.maximum(d_m, 0.0) / h_start, 0.0, 1.0)
    f_dn = np.clip(np.maximum(-d_m, 0.0) / m_start, 0.0, 1.0)

    def per_tick(raw):
        per_day = (daylight.monthly_value(raw, mid)
                   if isinstance(raw, (list, tuple)) else float(raw))
        return per_day * hours / HOURS_PER_DAY

    e = per_tick(config["exchange_m_per_day"])
    e_b = per_tick(sub["reservoir_exchange_m_per_day"])
    a_m = np.clip(e / m_mid, 0.0, 1.0)              # ML -> layer
    a_d = np.clip(e / h_mid, 0.0, 1.0)              # layer -> ML
    a_b = np.clip(e_b / h_mid, 0.0, 1.0 - a_d)      # layer -> reservoir
    n_res = daylight.monthly_value(sub["reservoir_umol_l"], mid)
    return {
        "n_mm": (1.0 - f_dn) * (1.0 - a_m) + f_dn * a_d,
        "n_md": f_up * (1.0 - a_m) + (1.0 - f_up) * a_d,
        "n_dm": (1.0 - f_dn) * a_m + f_dn * (1.0 - a_d - a_b),
        "n_dd": f_up * a_m + (1.0 - f_up) * (1.0 - a_d - a_b),
        "n_s": float(yield_) * e_b * n_res,
        "b_mm": 1.0 - f_dn,
        "b_md": f_up,
        "b_dm": f_dn,
        "b_dd": 1.0 - f_up,
    }


def initial_pool(config, yield_, year_tick, layer="mixed"):
    """N at a world's start, as a pool: the surface climatology of that
    day in the mixed layer; ``sub_mixed_layer_umol_l`` over the
    subsurface layer (section 180)."""
    hours = resolve_tick_hours(config["tick_hours"])
    t = float(year_tick) * hours
    key = "sub_mixed_layer_umol_l" if layer == "subsurface" else "surface_umol_l"
    c = daylight.monthly_value(config[key], t / HOURS_PER_DAY)
    return (float(yield_) * float(layer_thickness(config, t, layer))
            * float(c))


def concentration(config, yield_, pool, year_tick, layer="mixed"):
    """Concentration (umol/l) of a layer's pool at a year tick."""
    hours = resolve_tick_hours(config["tick_hours"])
    depth = float(layer_thickness(config, float(year_tick) * hours
                                  + 0.5 * hours, layer))
    return np.asarray(pool) / (float(yield_) * depth)
