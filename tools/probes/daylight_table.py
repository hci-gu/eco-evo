"""Daylight calendar per month: light level, attack rates, producer growth.

For every light-dependent pair in fg_library.yaml (``dark_ratio`` < 1)
prints the monthly mean of the attack-rate multiplier m(t) and the
implied daylight / darkness attack rates, at the project's latitude.
The library's ``max_intake_rate`` is the ANNUAL MEAN a (m averages to
1 over the year), so the months where m is well below 1 are the ones
to check against tools/probes/budget_gate.py. Section 137.

For every light-limited producer (``light_saturation`` set) it also
prints the monthly mean growth factor P(t)/P_ref, which is 1 on the
species' reference day (default mid April) - section 138.

With a ``simulation_settings.temperature`` block it prints, for every
group with ``metabolism_q10``, the monthly mean temperature of its
layer and the Q10 multiplier on its resting_metabolism - section 143.

Reads the manifest's latitude even when the calendar is disabled there,
so the effect can be inspected before switching it on.

    python3 tools/probes/daylight_table.py
    python3 tools/probes/daylight_table.py --tick-length 1 --latitude 63
"""
import argparse
import os
import sys

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from lib.world import daylight, temperature  # noqa: E402
from lib.world.tick_time import LIBRARY_TICK_HOURS  # noqa: E402

MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _load(path):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def _monthly(values, tick_hours):
    per_day = 24 // tick_hours
    days = np.asarray(values).reshape(365, per_day)
    out, first = [], 0
    for n in MONTH_DAYS:
        out.append(days[first:first + n].mean())
        first += n
    return out


def _print_temperature(project, species, hours):
    """Monthly Q10 metabolism multipliers per group (section 143)."""
    block = ((project.get("simulation_settings") or {}).get("temperature")
             or {})
    if not block.get("layers"):
        print("\nNo simulation_settings.temperature: metabolism is constant.")
        return
    state = "ON" if block.get("enabled") else "off (shown as if on)"
    layers = {str(k): v for k, v in block["layers"].items()}
    group_layers = block.get("group_layers") or {}
    print(f"\nWater temperature {state}: Q10 multiplier on resting_metabolism")
    for fid, spec in species.items():
        settings = temperature.species_settings(spec or {})
        if settings is None:
            continue
        q10, t_ref = settings
        layer = group_layers.get(fid)
        if layer not in layers:
            print(f"  {fid}: metabolism_q10 {q10:g} but no layer assigned")
            continue
        temps = _monthly(temperature.tick_temperatures(layers[layer], hours),
                         hours)
        m = temperature.metabolism_multiplier(layers[layer], hours, q10, t_ref)
        monthly = _monthly(m, hours)
        ref = t_ref if t_ref == temperature.ANNUAL_MEAN else f"{t_ref:g} C"
        print(f"\n{fid}: Q10 {q10:g}, t_ref {ref}, layer {layer}")
        print("  monthly mean T (C):  "
              + "  ".join(f"{mo} {v:4.1f}" for mo, v in zip(MONTHS, temps)))
        print("  monthly mean m:      "
              + "  ".join(f"{mo} {v:4.2f}" for mo, v in zip(MONTHS, monthly)))
        print(f"  annual mean m {m.mean():.2f}; lowest {m.min():.2f}, "
              f"highest {m.max():.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default="mareld2.yaml")
    parser.add_argument("--library", default="fgconfig/fg_library.yaml")
    parser.add_argument("--tick-length", type=int, default=LIBRARY_TICK_HOURS,
                        help="tick length in hours (1-6); 24 must divide by it "
                             "for the monthly table")
    parser.add_argument("--latitude", type=float, default=None,
                        help="override the manifest's latitude_deg")
    args = parser.parse_args(argv)

    project = _load(args.project)
    block = ((project.get("simulation_settings") or {}).get("daylight")
             or {})
    latitude = (args.latitude if args.latitude is not None
                else block.get("latitude_deg"))
    if latitude is None:
        parser.error("no latitude: set simulation_settings.daylight."
                     "latitude_deg or pass --latitude")
    latitude = float(latitude)
    hours = int(args.tick_length)
    if 24 % hours:
        parser.error("the monthly table needs a tick length dividing 24 h")

    state = "ON" if block.get("enabled") else "off (shown as if on)"
    print(f"Daylight calendar {state}: latitude {latitude:g} deg, "
          f"{hours} h ticks")
    print("Light level F (the observation channel, threshold "
          f"{daylight.DEFAULT_THRESHOLD_DEG:g} deg), monthly mean:")
    light = _monthly(daylight.light_schedule(latitude, hours), hours)
    print("  " + "  ".join(f"{m} {v:4.2f}" for m, v in zip(MONTHS, light)))

    library = _load(args.library)
    species = library.get("species_definitions", {})

    settings = daylight.parse_settings(
        {"simulation_settings": {"daylight": dict(block, enabled=True,
                                                  latitude_deg=latitude)}})
    climate = settings["light_climate"]
    for fid, spec in species.items():
        light = daylight.growth_light_settings(spec or {})
        if light is None:
            continue
        ik, ref_day = light
        ik_text = (f"{ik:g}" if not isinstance(ik, tuple) else "monthly " + "/".join(f"{v:g}" for v in ik))
        print(f"\n{fid}: light-limited growth, I_k {ik_text} umol/m2/s, "
              f"factor 1 on day {ref_day}")
        if climate is None:
            print("  no light climate in the manifest (Kd, mixed layer, "
                  "clouds): the simulator refuses to run")
            continue
        g = daylight.growth_light_schedule(latitude, hours, climate, ik,
                                           ref_day)
        monthly = _monthly(g, hours)
        print("  monthly mean growth factor: "
              + "  ".join(f"{m} {v:4.2f}" for m, v in zip(MONTHS, monthly)))
        print(f"  annual mean {g.mean():.2f}; highest tick {g.max():.2f} "
              "(nights are 0)")
    _print_temperature(project, species, hours)
    pairs = [(key, d) for key, d in
             (library.get("interaction_definitions") or {}).items()
             if "_preys_on_" in key and d.get("preys_on")
             and daylight.pair_settings(d) is not None]
    if not pairs:
        print("\nNo pair has dark_ratio < 1: no attack rate depends on light.")
        return 0

    for key, inter in pairs:
        rho, threshold = daylight.pair_settings(inter)
        pred = key.split("_preys_on_")[0]
        a = inter.get("max_intake_rate",
                      species.get(pred, {}).get("max_intake_rate"))
        m = daylight.attack_multiplier(latitude, hours, rho, threshold)
        light_raw = rho + (1.0 - rho) * daylight.light_schedule(
            latitude, hours, threshold)
        a_light = 1.0 / float(light_raw.mean())       # in units of a_mean
        print(f"\n{key}: dark_ratio {rho:g}, threshold {threshold:g} deg")
        if a is not None:
            a = float(a)
            print(f"  a (annual mean, library at {LIBRARY_TICK_HOURS} h) "
                  f"= {a:g};  a_light = {a * a_light:.4g},  "
                  f"a_dark = {a * a_light * rho:.4g}")
        else:
            print(f"  a_light = {a_light:.3f} x a,  "
                  f"a_dark = {a_light * rho:.3f} x a")
        monthly = _monthly(m, hours)
        print("  monthly mean m: "
              + "  ".join(f"{mo} {v:4.2f}" for mo, v in zip(MONTHS, monthly)))
        print(f"  lowest month {MONTHS[int(np.argmin(monthly))]} "
              f"({min(monthly):.2f} x a), highest "
              f"{MONTHS[int(np.argmax(monthly))]} ({max(monthly):.2f} x a)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
