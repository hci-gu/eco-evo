#!/usr/bin/env python3
"""Can a feeding-time window alone hold a fish at 0.3 of Cmax? (gut buffering)

Research diagnostic. A fish forages in one window of tau x 24 h per day and
ingests at rate r (in units of Cmax per day) while foraging, limited by the
free stomach space; outside the window it only digests. The stomach
evacuates by the square-root model of the ICES/SMS gastric evacuation
family (Andersen 1998; Temming & Herrmann 2003; dS/dt = -rho sqrt(S)),
scaled so that a stomach held at its capacity Smax evacuates at Cmax:
rho = Cmax / sqrt(Smax). Smax is expressed in days of Cmax (capacity /
daily Cmax): for cod the maximum meal or stomach capacity is a few % of
body weight against a daily Cmax of ~2-2.5 % BW/d (values from the
literature tables of this folder; scanned here over 0.5-4).

The fish runs to a periodic steady state; the daily ration (= mean
evacuation) over Cmax is the feeding level F.

Two foraging regimes:
  * encounter-theory rate: r = x Cmax/day with x = 44-266 per foraging
    time (engine_feeding_level_stdout.txt; the fish fills its tick ceiling
    in minutes) - the window binds only if the stomach is too small to
    carry a day's food;
  * field-implied rates r = 0.5-4 Cmax/day while foraging.
"""
import numpy as np


def feeding_level(tau, r, smax_days, dt_h=0.05, days=40):
    cmax = 1.0  # per day
    smax = smax_days * cmax
    rho = cmax / np.sqrt(smax)
    S = 0.0
    n = int(24 / dt_h)
    win = int(round(tau * n))
    dt = dt_h / 24.0
    evac_total = 0.0
    for d in range(days):
        evac_day = 0.0
        for i in range(n):
            ev = rho * np.sqrt(max(S, 0.0)) * dt
            S -= ev
            evac_day += ev
            if i < win:
                S = min(smax, S + r * dt)
        if d == days - 1:
            evac_total = evac_day
    return evac_total / cmax


taus = [0.1, 0.17, 0.33, 0.5, 1.0]
smaxes = [0.5, 1.0, 2.0, 3.0]
for r, label in [(266.0, "encounter theory, benthivores -> infauna (x 266)"),
                 (44.0, "encounter theory, predators -> epifauna (x 44)"),
                 (4.0, "field-implied, 4 Cmax/day while foraging"),
                 (1.0, "field-implied, 1 Cmax/day while foraging"),
                 (0.5, "field-implied, 0.5 Cmax/day while foraging")]:
    print(f"r = {label}")
    print("  Smax (days of Cmax) \\ tau " + "".join(f"{t:>7.2f}" for t in taus))
    for sm in smaxes:
        print(f"  {sm:24.1f}   " + "".join(f"{feeding_level(t, r, sm):7.2f}" for t in taus))
    print()
print("Reading: with encounter-theory rates the daily ration is set by the stomach, not by")
print("the window: a stomach that holds >= 1 day of Cmax gives F >= ~0.6-0.9 even with a")
print("2.4-h window. F ~0.3 needs a low intake rate while foraging (field-implied regime).")
