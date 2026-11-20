"""How much the measured near-bed light over < 100 m beds flattens the rule-A light term against the light convention
the library's class tables use.

Usage: python3 -I -W ignore shallow_light_factor.py <repo> > ../tables/shallow_light_factor_stdout.txt
Runs the < 100 m and the 100-150 m classes. Optional 2nd argument: a factor on every glider E (2.5 = the
convention-vs-glider level gap of research_notes/"Ljus under 65 m" section 3, taken as a glider artefact;
4-6 = the Swedish near-bed column under the Danish side's clearer surface layer, Kd490 0.235 vs 0.148 over ~15-20 m).
Optional 3rd / 4th arguments: keep only casts west of a longitude / over beds at least this deep (placement check). The glider part is computed from the casts (needs
research_notes/"Ljus under 65 m"/scripts/voto_k.py and its data, read in place).

Convention (copied from research_notes/"Ljus under 65 m"/scripts/deep_light.py, itself from "Ljus under 50 m"
box_light.py): clear-sky lux x 0.5 cloud, 58.15 N, 0-2 m K 0.30, 2-20 m 1.56/Secchi(month), 20-50 m 0.109, 50-65 m
0.091, below 65 m K_deep (0.09 live library; 0.16 = the "class" reading of research_notes/"Djupljus i synparen"
sensitivity.py for < 100 m beds). Light at bed - 5 m for the model's < 100 m cells (maps/depth.npz).

Glider: per daytime cast (sun > 0) over an EMODnet bed < 100 m that reached within 15 m of the bed, the PAR in the
bed - 5 m bin (or the deepest bin within 15 m of the bed); below-threshold bins set to the threshold ("cap", upper
bound) or to 0 ("zero", lower bound). lx = 50 x umol.

Light term (rule A, fish prey): f = r2 + (1 - r2) g, g = E/(E + Ke); daily mean H = D mean_day(f) + (1 - D) r2;
reported as H/r2 (1 = dark). The 150-250 m class is ~1 in both (glider: below threshold in 98-100 % of casts;
convention at K 0.09: 1e-4 to 1e-6 lx by day), so the shallow : 150-250 m ratio of the light term moves by
F = H/r2(glider) / H/r2(convention) on matched months.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO = sys.argv[1]
LEVEL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # multiplies every glider E (level-gap sensitivity)
LONMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 99.0  # keep casts west of this longitude only
BEDMIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # and over beds at least this deep
sys.path.insert(0, os.path.join(REPO, "research_notes", "Ljus under 65 m", "scripts"))
import voto_k as vk  # noqa: E402

LAT = 58.15
SECCHI = {1: 5.0, 2: 7.0, 3: 7.0, 4: 8.5, 5: 8.0, 6: 7.0, 7: 10.0, 8: 9.5, 9: 8.0, 10: 9.0, 11: 9.0, 12: 11.5}
LX = 50.0


def surface_lux(h):
    pts = [(-18, 6e-4), (-12, 8e-3), (-6, 3.0), (0, 500.0), (5, 1.3e5 * math.sin(math.radians(5)))]
    if h >= 5:
        return 1.3e5 * math.sin(math.radians(h))
    if h <= -18:
        return 6e-4
    for (h0, e0), (h1, e1) in zip(pts[:-1], pts[1:]):
        if h0 <= h <= h1:
            f = (h - h0) / (h1 - h0)
            return math.exp(math.log(e0) + f * (math.log(e1) - math.log(e0)))
    return 6e-4


def month_sun(month):
    days = np.arange(1, 366)
    mdays = days[(np.minimum(((days - 1) / (365 / 12)).astype(int), 11) + 1) == month][::3]
    hours = np.arange(0.125, 24, 0.25)
    dec = np.radians(23.44) * np.sin(2 * np.pi * (284 + mdays) / 365.0)
    ha = np.radians(15 * (hours - 12))
    phi = math.radians(LAT)
    s = math.sin(phi) * np.sin(dec)[:, None] + math.cos(phi) * np.cos(dec)[:, None] * np.cos(ha)[None, :]
    return np.degrees(np.arcsin(np.clip(s, -1, 1))).ravel()


def tau(z, month, kdeep):
    lay = [(0, 2, 0.30), (2, 20, 1.56 / SECCHI[month]), (20, 50, 0.109), (50, 65, 0.091), (65, 1e4, kdeep)]
    return sum(k * (min(z, b) - t) for t, b, k in lay if z > t)


def conv_H(month, zb_list, kdeep, ke, r2):
    el = month_sun(month)
    day = el > 0
    E0 = np.array([surface_lux(h) for h in el[day]]) * 0.5
    gs = []
    for zb in zb_list:
        E = E0 * math.exp(-tau(zb, month, kdeep))
        gs.append(np.mean(E / (E + ke)))
    g = float(np.mean(gs))
    D = float(day.mean())
    return 1 + D * (1 / r2 - 1) * g, g, D


def main():
    for lo, hi in ((0, 100), (100, 150)):
        run(lo, hi)


def run(BLO, BHI):
    print(f"######## bed class {BLO}-{BHI} m ########")
    depth = np.load(os.path.join(REPO, "maps", "depth.npz"))["depth"]
    zcells = depth[(depth >= BLO) & (depth < BHI)] - 5.0
    zsample = np.percentile(zcells, np.arange(5, 100, 10))  # 10 quantiles of the < 100 m cells
    meta = json.load(open(os.path.join(vk.DATA, "meta.json")))
    gl = defaultdict(list)  # month -> list of (E cap, E zero) umol
    for ds, m in meta.items():
        if "MPE" not in m["attrs"].get("irradiance", ""):
            continue
        casts, _ = vk.load(ds)
        dark, noise, _ = vk.darkfloor(casts, vk.CH_PAR)
        for c in casts:
            if c["sun"] <= 0:
                continue
            bed = vk.bed_of(c)
            if not np.isfinite(bed) or not (max(BLO, 0.1, BEDMIN) <= bed < BHI) or c["lon"] > LONMAX:
                continue
            kb = int((bed - 5.0) // 5)
            if c["kmax"] < kb or kb not in dark:
                kb = c["kmax"]
                if 5 * kb + 2.5 < bed - 15 or kb not in dark:
                    continue
            row = c["bins"].get(kb)
            if row is None or not np.isfinite(row.get(vk.CH_PAR, np.nan)):
                continue
            E = vk.corrected(c, vk.CH_PAR, dark, noise, True)
            thr = max(3 * dark[kb], 10 * max(noise[kb], 1e-12)) * 1e4
            mo = int(str(c["t"])[5:7])
            if kb in E:
                gl[mo].append((E[kb] * 1e4 * LEVEL, E[kb] * 1e4 * LEVEL))
            else:
                gl[mo].append((thr * LEVEL, 0.0))
    months = sorted(mo for mo in gl if len(gl[mo]) >= 10)
    print(f"{BLO}-{BHI} m cells: bed - 5 m quantiles used for the convention: {', '.join(f'{z:.0f}' for z in zsample)} m")
    print(f"glider months with >= 10 near-bed casts over {BLO}-{BHI} m beds: {months} "
          f"(n {[len(gl[mo]) for mo in months]})\n")
    out = {}
    for ke in (0.002, 0.02, 0.1):
        for r2 in (0.035, 0.06, 0.09):
            rows = []
            for mo in months:
                el = month_sun(mo)
                D = float((el > 0).mean())
                ec = np.array([x[0] for x in gl[mo]]) * LX
                ez = np.array([x[1] for x in gl[mo]]) * LX
                gc = float(np.mean(ec / (ec + ke)))
                gz = float(np.mean(ez / (ez + ke)))
                hc = 1 + D * (1 / r2 - 1) * gc
                hz = 1 + D * (1 / r2 - 1) * gz
                h9, g9, _ = conv_H(mo, zsample, 0.09, ke, r2)
                h16, g16, _ = conv_H(mo, zsample, 0.16, ke, r2)
                rows.append((mo, D, gc, gz, g9, g16, hc, hz, h9, h16))
            # all-month convention for reference
            hall9 = np.mean([conv_H(mo, zsample, 0.09, ke, r2)[0] for mo in range(1, 13)])
            hall16 = np.mean([conv_H(mo, zsample, 0.16, ke, r2)[0] for mo in range(1, 13)])
            A = np.array([r[6:] for r in rows])
            m = A.mean(axis=0)
            res = dict(months=months, H_glider_cap=m[0], H_glider_zero=m[1], H_conv09=m[2], H_conv16=m[3],
                       H_conv09_allmonths=hall9, H_conv16_allmonths=hall16,
                       F_cap_vs09=m[0] / m[2], F_zero_vs09=m[1] / m[2], F_cap_vs16=m[0] / m[3],
                       F_zero_vs16=m[1] / m[3])
            out[f"Ke{ke}_r2{r2}"] = res
            if r2 == 0.06 or ke == 0.02:
                print(f"Ke {ke} lx, r2 {r2}:")
                if r2 == 0.06:
                    for r in rows:
                        print(f"   month {r[0]:2d} D {r[1]:.2f}  mean_day g: glider cap {r[2]:.3f} zero {r[3]:.3f} | "
                              f"convention K_deep 0.09 {r[4]:.3f}, 0.16 {r[5]:.3f}   H/r2: {r[6]:.2f} / {r[7]:.2f} | "
                              f"{r[8]:.2f} / {r[9]:.2f}")
                print(f"   matched-month mean H/r2: glider cap {m[0]:.2f}, zero {m[1]:.2f}; convention 0.09 {m[2]:.2f}, "
                      f"0.16 {m[3]:.2f} (all 12 months: {hall9:.2f} / {hall16:.2f})")
                print(f"   F = glider / convention: cap {m[0] / m[2]:.3f} (vs 0.16: {m[0] / m[3]:.3f}); "
                      f"zero {m[1] / m[2]:.3f} (vs 0.16: {m[1] / m[3]:.3f})\n")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables",
                                     f"shallow_light_factor_{BLO}_{BHI}" + ("" if LEVEL == 1.0 else f"_level{LEVEL:g}")
                                     + ("" if LONMAX > 90 else f"_lon{LONMAX:g}") + ("" if BEDMIN == 0 else f"_bed{BEDMIN:g}")
                                     + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
