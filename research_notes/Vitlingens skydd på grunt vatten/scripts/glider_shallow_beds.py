"""Daytime light near the bed over shallow (< 100 m) beds in the box, from the VOTO gliders, against the deeper classes.

Usage: python3 -I glider_shallow_beds.py <repo> > ../tables/glider_shallow_beds_stdout.txt
       (writes ../tables/glider_shallow_beds.json)

Reuses research_notes/"Ljus under 65 m"/scripts/voto_k.py (loader, sun elevation, dark floor, usability rule,
EMODnet bed lookup); the data are read in place from that note's data/ folder (VOTO, CC BY 4.0; attribution in
that folder's README.md). MPE-PAR missions only (2024-2026; the OCR-504 PAR channel is usable only to ~65 m).

Per daytime cast (sun > 0) with a bed depth under it:
- composition: where and when the casts over each bed class are (month, lat/lon), so that the class contrast is
  not a season or position artefact;
- month- and mission-matched light at fixed depths (50 / 65 / 80 m) per bed class: the median over casts of the
  dark-corrected PAR, with below-floor bins counted as 0 (as voto_k.py does);
- near-bed light E_b at the bed - 5 m (the bottom 10 m, whiting as prey) for casts that reached within 15 m of the
  bed: the 5 m bin there if usable; if the bin was measured but is below the usability threshold, E_b is set to
  the threshold (an UPPER bound, so g below is biased high = against the protection hypothesis);
- the rule-A light term per cast: f = r2 + (1 - r2) g, g = E/(E + Ke), r2 = (R_nv / R_sat)^2 (fish prey: R_nv 0.3 m,
  R_sat = 2 L_pred; saithe 0.5-0.8 m long -> R_sat 1.0-1.6 m -> r2 0.035-0.09; central 0.06), lx = 50 x umol m-2 s-1
  (45-55 at these depths, research_notes/"Ljus under 65 m" spectral.py);
- monthly daytime mean of f, the day fraction from the sun (share of the month's hours with sun > 0 at 58.1 N), and
  the daily-mean H = D mean_day(f) + (1 - D) r2, all relative to the night/dark value r2 (H/r2 = 1 = no vision).
Ke: 0.02 lx central, bracket 2e-3 and 0.1 lx (research_notes/"Torskfiskars syn i morker"; library 0.02).
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO = sys.argv[1]
VK_DIR = os.path.join(REPO, "research_notes", "Ljus under 65 m", "scripts")
sys.path.insert(0, VK_DIR)  # project-written script folder (not downloaded data)
import voto_k as vk  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "..", "tables")
LX = 50.0
KES = (0.002, 0.02, 0.1)
R2S = (0.035, 0.06, 0.09)
CLASSES = ("<100", "100-150", "150-250", ">=250")


def day_fraction(month, lat=58.1, lon=10.6):
    """Share of the month's hours with sun > 0 deg (15-min steps, 2025)."""
    t0 = np.datetime64(f"2025-{month:02d}-01T00:00:00")
    t1 = np.datetime64(f"2025-{month + 1:02d}-01T00:00:00") if month < 12 else np.datetime64("2026-01-01T00:00:00")
    t = np.arange(t0, t1, np.timedelta64(15, "m")).astype("datetime64[s]")
    return float((vk.sun_elev(t, lat, lon) > 0).mean())


def main():
    meta = json.load(open(os.path.join(vk.DATA, "meta.json")))
    comp = defaultdict(lambda: defaultdict(int))
    pos = defaultdict(list)
    fixed = defaultdict(list)            # (cls, month, z) -> E (umol, 0 below floor)
    fixed_mis = defaultdict(list)        # (cls, mission, z)
    nearbed = defaultdict(list)          # (cls, month) -> (E_b umol, upper_bound flag, bed)
    thr_all = []
    for ds, m in meta.items():
        if "MPE" not in m["attrs"].get("irradiance", ""):
            continue
        casts, _ = vk.load(ds)
        dark, noise, _ = vk.darkfloor(casts, vk.CH_PAR)
        for c in casts:
            if c["sun"] <= 0:
                continue
            bed = vk.bed_of(c)
            if not np.isfinite(bed) or bed <= 0:
                continue
            cls = vk.bed_class(bed)
            mo = int(str(c["t"])[5:7])
            comp[cls][mo] += 1
            pos[cls].append((c["lat"], c["lon"]))
            E = vk.corrected(c, vk.CH_PAR, dark, noise, True)
            for z in (50, 65, 80):
                k = int(z // 5)
                if k in c["bins"] and np.isfinite(c["bins"][k].get(vk.CH_PAR, np.nan)) and k in dark:
                    v = E.get(k, 0.0) * 1e4
                    fixed[(cls, mo, z)].append(v)
                    fixed_mis[(cls, ds, z)].append(v)
            # near-bed bin: bed - 5 m
            zb = bed - 5.0
            kb = int(zb // 5)
            if c["kmax"] < kb or kb not in dark:
                # cast stopped above the bed-5 m bin: try the deepest bin within 15 m of the bed
                kb = c["kmax"]
                if 5 * kb + 2.5 < bed - 15 or kb not in dark:
                    continue
            row = c["bins"].get(kb)
            if row is None or not np.isfinite(row.get(vk.CH_PAR, np.nan)):
                continue
            thr = max(3 * dark[kb], 10 * max(noise[kb], 1e-12)) * 1e4
            thr_all.append(thr)
            if kb in E:
                nearbed[(cls, mo)].append((E[kb] * 1e4, False, bed))
            else:
                nearbed[(cls, mo)].append((thr, True, bed))
    print("VOTO MPE-PAR daytime casts (sun > 0) by EMODnet bed class and month (counts):")
    print("  class    " + " ".join(f"{mo:>5d}" for mo in range(1, 13)) + "   lat range / lon range (p5-p95)")
    for cls in CLASSES:
        p = np.array(pos[cls]) if pos[cls] else np.zeros((0, 2))
        rng = (f"{np.percentile(p[:, 0], 5):.2f}-{np.percentile(p[:, 0], 95):.2f} N / "
               f"{np.percentile(p[:, 1], 5):.2f}-{np.percentile(p[:, 1], 95):.2f} E") if len(p) else "-"
        print(f"  {cls:8s} " + " ".join(f"{comp[cls][mo]:5d}" for mo in range(1, 13)) + "   " + rng)
    print(f"\nUsability threshold near the bed (umol m-2 s-1): median {np.median(thr_all):.2e} "
          f"(= {np.median(thr_all) * LX:.2e} lx at {LX:g} lx/umol)")

    print("\nMonth-matched daytime median PAR (umol m-2 s-1; below-floor = 0; n casts) at fixed depth, by bed class:")
    out = dict(fixed={}, nearbed={}, H={})
    for z in (50, 65, 80):
        print(f"  {z} m")
        for mo in range(1, 13):
            cells = []
            for cls in CLASSES:
                v = fixed.get((cls, mo, z), [])
                if len(v) >= 10:
                    med = float(np.median(v))
                    cells.append(f"{cls} {med:9.2e} n{len(v):4d}")
                    out["fixed"][f"{cls} {mo} {z}"] = (med, len(v))
                else:
                    cells.append(f"{cls}         - n{len(v):4d}")
            if any(len(fixed.get((cls, mo, z), [])) >= 10 for cls in CLASSES[:1]):
                print(f"    month {mo:2d}: " + " | ".join(cells))
    print("\nMission-matched (same glider mission), daytime median PAR at 50 / 65 m, < 100 vs 150-250 m beds:")
    for ds in meta:
        line = []
        for z in (50, 65):
            a = fixed_mis.get(("<100", ds, z), [])
            b = fixed_mis.get(("150-250", ds, z), [])
            if len(a) >= 10 and len(b) >= 10:
                ma, mb = np.median(a), np.median(b)
                line.append(f"{z} m: {ma:.2e} (n{len(a)}) vs {mb:.2e} (n{len(b)}) ratio {ma / mb if mb > 0 else float('nan'):.3f}")
        if line:
            print(f"  {ds}: " + "; ".join(line))

    print("\nNear-bed light (bed - 5 m; casts that reached within 15 m of the bed), daytime, by class and month:")
    print("  E_b median (umol) / share of casts below the usability threshold (E_b set to the threshold) / n / bed median")
    for cls in CLASSES:
        for mo in range(1, 13):
            v = nearbed.get((cls, mo), [])
            if len(v) < 10:
                continue
            e = np.array([x[0] for x in v])
            ub = np.mean([x[1] for x in v])
            bd = np.median([x[2] for x in v])
            print(f"  {cls:8s} month {mo:2d}: E_b {np.median(e):.2e} (p75 {np.percentile(e, 75):.2e}, p90 {np.percentile(e, 90):.2e})"
                  f"  below-thr {ub:.2f}  n{len(v):4d}  bed {bd:.0f} m")
            out["nearbed"][f"{cls} {mo}"] = dict(med=float(np.median(e)), p90=float(np.percentile(e, 90)), ub=float(ub),
                                                 n=len(v), bed=float(bd))

    print("\nRule-A light term at the bed, daily mean H relative to the dark value r2 (H/r2 = 1: no vision gain).")
    print("  H/r2 = 1 + D (1/r2 - 1) mean_day(g); per month where the class has >= 10 near-bed casts; Ke in lx.")
    for ke in KES:
        for r2 in R2S:
            for cls in CLASSES:
                rows = []
                for mo in range(1, 13):
                    v = nearbed.get((cls, mo), [])
                    if len(v) < 10:
                        continue
                    e = np.array([x[0] for x in v]) * LX
                    g = float(np.mean(e / (e + ke)))
                    D = day_fraction(mo)
                    rows.append((mo, D, g, 1 + D * (1 / r2 - 1) * g))
                if not rows:
                    continue
                Hm = float(np.mean([r[3] for r in rows]))
                out["H"][f"Ke{ke} r2{r2} {cls}"] = dict(months=[r[0] for r in rows], H_over_r2=Hm,
                                                         g=[round(r[2], 4) for r in rows])
                if r2 == 0.06:
                    print(f"  Ke {ke:<6g} r2 {r2}: {cls:8s} months {','.join(str(r[0]) for r in rows):24s} "
                          f"mean_day g {' '.join(f'{r[2]:.3f}' for r in rows)}  -> H/r2 {Hm:.2f}")
    json.dump(out, open(os.path.join(TAB, "glider_shallow_beds.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
