"""Does eating pay the planktivores in winter, before any predation? (mechanism check, library values)

Usage: python3 -I winter_feeding_payoff.py <repo_root>

Per cell of maps/depth.npz, per tonne of planktivores at eat share 1, the energy gain from the two
pelagic prey pairs (zooplankton, krill) under shared handling and fixed attack (sections 242, 265:
demand_j = lambda_j / (1 + sum_k h_k lambda_k), lambda = a_eff B), net of SDA, against the eat
premium (feeding_cost - 1) x resting cost at the month's temperature. Months January and July.

Library values (fg_library.yaml, mareld2.yaml, read 2026-11-11, copied here, not parsed):
  planktivores: RM 8.0 MJ/t/tick at t_ref 14 C, metabolism_q10 1.5, feeding_cost 1.65,
    ingestion_q10 2.2 (a and 1/h, annual mean 1), sda_fraction 0.11, layer mean_0_50m.
  -> zooplankton: a 0.0122, h 52.0, assimilation 0.8, Type II, dark_ratio 0.03 at 0 deg, E 4500.
  -> krill: a 0.023, h 57.8, assimilation 0.8, Type II, dark_ratio 0.57 at 0 deg, E 5000 (AGENTS.md),
    depth_access_table (renormalised to equal-cell mean 1 by the loader).
  krill: outside_density 5.0 t/km2 box mean, season Jan 1.0 / Jul 1.1, outside_depth_table (mean 1).
  zooplankton biomass: January 1.5, July 10.6 t/km2 (research_notes/"Sillens vinterfoda" Q1:
    model at herring cells; A17 0-25 m Jan 1.2-1.6 g WW m-2, Jul 6.9-9.6) - uniform over cells.
Simplifications [J]: prey fully visible (no zooplankton hide, no krill refuge), no interference
(planktivores w 0.05 ignored), daylight multiplier from day length at 58.15 N with the sun's
elevation threshold 0 deg (m = 1 by day, dark_ratio by night, annual mean 1), no predation term.
So the gain is an UPPER bound on what eating pays. Nothing fitted; no model output read.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
depth = np.load(os.path.join(ROOT, "maps", "depth.npz"))["depth"].astype(float).ravel()
depth = depth[np.isfinite(depth) & (depth > 0)]

T = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]   # mean_0_50m, mareld2.yaml
LAT = 58.15


def day_fraction(month):
    doy = 30.4 * month - 15.2                                   # mid-month, month 1..12
    dec = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365.0)
    x = -math.tan(math.radians(LAT)) * math.tan(dec)
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x)) * 2 / 15.0 / 24.0      # sun above 0 deg


DAYF = [day_fraction(m) for m in range(1, 13)]


def light_mult(month, dark):
    norm = np.mean([d + (1 - d) * dark for d in DAYF])
    d = DAYF[month - 1]
    return (d + (1 - d) * dark) / norm


QIN = [2.2 ** ((t - 14.0) / 10.0) for t in T]
QIN = [q / np.mean(QIN) for q in QIN]                             # annual mean 1


def interp_table(rows):
    xs, ys = zip(*rows)
    v = np.interp(depth, xs, ys)
    return v


krill_shape = interp_table([(49.9, 0.025), (50.0, 0.38), (99.9, 0.38), (100.0, 0.8), (150.0, 1.0),
                            (250.0, 1.2), (530.0, 1.4)])
krill_shape /= krill_shape.mean()
acc = interp_table([(99.9, 0.0707), (100.0, 0.0211), (149.9, 0.0211), (150.0, 0.00377),
                    (249.9, 0.00377), (250.0, 0.00171)])
acc /= acc.mean()

ZOO = {1: 1.5, 7: 10.6}
KSEASON = {1: 1.0, 7: 1.1}
print(f"cells {depth.size}; bed depth p10/p50/p90 {np.percentile(depth, [10, 50, 90]).round(0)} m;"
      f" share < 100 m {np.mean(depth < 100):.3f}")
print(f"day fraction Jan {DAYF[0]:.3f}, Jul {DAYF[6]:.3f}")
CLS = [("< 100 m", depth < 100), ("100-150 m", (depth >= 100) & (depth < 150)),
       ("150-250 m", (depth >= 150) & (depth < 250)), (">= 250 m", depth >= 250)]
for month in (1, 7):
    t = T[month - 1]
    rm = 8.0 * 1.5 ** ((t - 14.0) / 10.0)
    premium = 0.65 * rm
    q = QIN[month - 1]
    a_z = 0.0122 * q * light_mult(month, 0.03)
    h_z = 52.0 / q
    a_k = 0.023 * q * light_mult(month, 0.57) * acc
    h_k = 57.8 / q
    Bz = ZOO[month]
    Bk = 5.0 * KSEASON[month] * krill_shape
    lz, lk = a_z * Bz, a_k * Bk
    den = 1 + h_z * lz + h_k * lk
    gain = (lz * 4500 + lk * 5000) * (0.8 - 0.11) / den
    gz = lz * 4500 * (0.8 - 0.11) / den
    print(f"\n== month {month}: T {t} C, resting cost {rm:.2f}, eat premium {premium:.2f} MJ/t/tick; "
          f"ingestion factor {q:.3f}; light mult zoo {light_mult(month, 0.03):.3f}, krill "
          f"{light_mult(month, 0.57):.3f}")
    print(f"   zooplankton {Bz} t/km2 (alone: {a_z * Bz * 4500 * 0.69 / (1 + h_z * a_z * Bz):.1f} MJ/t/tick)")
    for name, m in CLS:
        if m.sum() == 0:
            continue
        g = gain[m]
        print(f"   {name:10s} cells {m.mean():.3f}: net gain mean {g.mean():6.1f} (zoo part {gz[m].mean():5.1f}),"
              f" gain / premium {g.mean() / premium:5.2f}; cells where eating pays {np.mean(g > premium):.2f}")
    print(f"   all cells: gain mean {gain.mean():.1f}, median {np.median(gain):.1f}; share of cells where "
          f"eating pays {np.mean(gain > premium):.2f}")
