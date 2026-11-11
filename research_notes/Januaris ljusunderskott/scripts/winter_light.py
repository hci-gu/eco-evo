"""Winter light in the box from the section 253 data: monthly K(2-20 m) and K(20-50 m) from the SMHI CTD PAR casts,
the surface PAR at 2 m against the clear sky, and the monthly Secchi climatology; compares the derivation chain's
monthly Secchi table (research_notes/"En synregel for fiskparen"/scripts/one_rule_pairs.py SECCHI) with them.

Usage: python3 -I winter_light.py <repo_root> > ../tables/winter_light_stdout.txt

Reads (in place, never modified):
  research_notes/"Ljus under 50 m"/tables/par_casts.json   (par_inventory.py: every SMHI CTD cast with PAR,
                                                          57.4-58.8 N, 9-12 E, 2018-2025)
Secchi numbers are copied from research_notes/"Ljus under 50 m"/tables/secchi_and_deep_water_stdout.txt
(SHARK, A16 / A17 1990-2024, Slaggo 2018-2024).

Method (as par_profiles.py, one change): reliable part of a cast = from 2 m down to the first 1 m bin with
PAR < THR; THR 0.5 umol m-2 s-1 here (the 253 note found slab K unbiased only at PAR >= 0.5; its K(2-20) table
used 0.14). Casts without a recorded water depth (IBTS Q1 2019) take the cast's deepest bin + 5 m as the bed.
Daytime = sun > 5 deg. K by least squares of ln PAR on z (1 m bins), >= 8 bins.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
NOTE = os.path.join(ROOT, "research_notes", "Ljus under 50 m")
casts = json.load(open(os.path.join(NOTE, "tables", "par_casts.json")))
BOX = (57.7, 58.4, 9.9, 11.3)
MN = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def slope(z, lp, z0, z1):
    sel = (z >= z0) & (z <= z1) & np.isfinite(lp)
    if sel.sum() < 8:
        return None
    return float(-np.polyfit(z[sel], lp[sel], 1)[0])


def clear_par(sun):
    s = math.sin(math.radians(sun))
    return 1098.0 * s * math.exp(-0.057 / s) * 0.45 * 4.57          # daylight.clear_sky_par (Haurwitz)


def rows_for(thr):
    out = []
    for k in casts:
        if k["sun_elev"] <= 5:
            continue
        z = np.array(k["z"], float)
        p = np.array([np.nan if v is None else v for v in k["par"]], float)
        bed = float(k["wadep"]) if k["wadep"] not in ("", None) else float(np.nanmax(z)) + 5.0
        if bed < 60:
            continue
        sel2 = (z >= 1.5) & (z <= 2.5)
        if not np.isfinite(p[sel2]).any():
            continue
        p2 = float(np.nanmedian(p[sel2]))
        if not p2 > 1:
            continue
        below = np.where((z >= 2) & ~(p >= thr))[0]
        zr = z[below[0] - 1] if len(below) else z[-1]
        ok = (z >= 2) & (z <= zr)
        lp = np.where(ok, np.log(np.where(p > 0, p, np.nan)), np.nan)
        inbox = BOX[0] <= k["lat"] <= BOX[1] and BOX[2] <= k["lon"] <= BOX[3]
        m = int(k["date"][5:7]) - 1
        cl = k.get("cloud")
        out.append(dict(date=k["date"], st=k["station"] or f"{k['lat']:.2f}N {k['lon']:.2f}E", sun=k["sun_elev"],
                        bed=bed, bed_rec=k["wadep"] not in ("", None), inbox=inbox, m=m, p2=p2, zr=float(zr),
                        k1=slope(z, lp, 2, 20), k2=slope(z, lp, 20, min(50, zr)) if zr >= 30 else None,
                        surf=p2 / clear_par(k["sun_elev"]),
                        cloud=int(cl) if str(cl).strip().isdigit() else None))
    return out


def med(v):
    v = [x for x in v if x is not None]
    return (float(np.median(v)), len(v)) if v else (float("nan"), 0)


for thr in (0.5, 0.14):
    R = rows_for(thr)
    print("=" * 110)
    print(f"Daytime casts (sun > 5 deg, bed >= 60 m), reliability threshold {thr} umol m-2 s-1: n {len(R)}")
    print("=" * 110)
    if thr == 0.5:
        print("winter casts (Nov-Mar), one row per cast:")
        print("  date        station                sun  bed(rec) box  PAR2m  z_rel  K2-20  K20-50  PAR2m/clear-sky  okta")
        for r in sorted(R, key=lambda r: r["date"]):
            if r["m"] in (10, 11, 0, 1, 2):
                f = lambda v: "  -  " if v is None else f"{v:.3f}"
                print(f"  {r['date']}  {r['st'][:20]:20s} {r['sun']:5.1f} {r['bed']:5.0f}{'' if r['bed_rec'] else '*'}"
                      f"  {'Y' if r['inbox'] else '.'}  {r['p2']:6.0f} {r['zr']:5.0f}  {f(r['k1'])}  {f(r['k2'])}"
                      f"   {r['surf']:.3f}          {r['cloud'] if r['cloud'] is not None else '-'}")
        print("  (* bed = deepest bin + 5 m, no water depth recorded)")
    print("\nby month, median (n):  K2-20 all | K2-20 bed >= 100 m | K20-50 all | PAR2m / clear-sky PAR")
    for m in range(12):
        Rm = [r for r in R if r["m"] == m]
        if not Rm:
            continue
        a = med([r["k1"] for r in Rm])
        b = med([r["k1"] for r in Rm if r["bed"] >= 100])
        c = med([r["k2"] for r in Rm])
        d = med([r["surf"] for r in Rm])
        print(f"  {MN[m]}  {a[0]:.3f} ({a[1]:2d}) | {b[0]:.3f} ({b[1]:2d}) | {c[0]:.3f} ({c[1]:2d}) | {d[0]:.3f} ({d[1]:2d})")
    for lab, ms in (("Jan-Feb", (0, 1)), ("Nov-Feb", (10, 11, 0, 1)), ("Apr-Sep", (3, 4, 5, 6, 7, 8))):
        Rm = [r for r in R if r["m"] in ms]
        a = med([r["k1"] for r in Rm])
        b = med([r["k1"] for r in Rm if r["bed"] >= 100])
        c = med([r["k2"] for r in Rm])
        d = med([r["surf"] for r in Rm])
        q = np.percentile([r["k1"] for r in Rm if r["k1"] is not None], [25, 75])
        print(f"  {lab:8s} K2-20 {a[0]:.3f} (n {a[1]}, p25-p75 {q[0]:.3f}-{q[1]:.3f}); bed>=100 {b[0]:.3f} (n {b[1]});"
              f" K20-50 {c[0]:.3f} (n {c[1]}); PAR2m/clear {d[0]:.3f} (n {d[1]})")

# ------------------------------------------------------------------ Secchi climatology vs the derivation's table
print("\n" + "=" * 110)
print("Monthly Secchi (m): the derivation's table vs the SHARK climatology (secchi_and_deep_water_stdout.txt)")
print("=" * 110)
DERIV = [5.0, 7.0, 7.0, 8.5, 8.0, 7.0, 10.0, 9.5, 8.0, 9.0, 9.0, 11.5]           # one_rule_pairs.SECCHI
A17 = [(5.0, 1), (7.0, 1), (8.0, 7), (11.0, 2), (9.5, 7), (7.5, 6), (11.0, 9), (9.0, 5), (8.0, 1), (9.0, 3), (9.0, 1),
       (11.0, 1)]
A16 = [None, (8.0, 2), (6.0, 8), (8.5, 3), (8.0, 5), (5.0, 7), (9.0, 7), (10.0, 3), (8.0, 6), (9.5, 2), None, (12.0, 1)]
SLG = [(5.5, 14), (5.0, 17), (6.0, 17), (6.0, 17), (6.5, 17), (7.0, 17), (8.0, 12), (7.5, 16), (7.0, 19), (6.5, 17),
       (5.8, 18), (5.0, 13)]
BOX_MEDIAN = 8.2                       # A16 + A17 all visits, n 88
slg = np.array([s for s, _ in SLG])
# Slaggo shape scaled to the box: the box all-visit median over the Slaggo visit-weighted median
slg_all = np.repeat(slg, [n for _, n in SLG])
scale = BOX_MEDIAN / float(np.median(slg_all))
shaped = slg * scale
print(f"  Slaggo all-visit median {np.median(slg_all):.2f} m (n {len(slg_all)}); scale to the box median 8.2 m: x{scale:.3f}")
print("  month  deriv  A17 (n)     A16 (n)     Slaggo (n)   Slaggo-shaped box   K2-20 deriv  K2-20 shaped (1.56/S)")
for m in range(12):
    a = f"{A17[m][0]:5.1f} ({A17[m][1]})" if A17[m] else "   -     "
    b = f"{A16[m][0]:5.1f} ({A16[m][1]})" if A16[m] else "   -     "
    print(f"  {MN[m]}   {DERIV[m]:5.1f}  {a:10s}  {b:10s}  {SLG[m][0]:5.1f} ({SLG[m][1]:2d})   {shaped[m]:6.2f}"
          f"              {1.56 / DERIV[m]:.3f}        {1.56 / shaped[m]:.3f}")
print(f"  Dec/Jan K ratio: derivation {DERIV[0] / DERIV[11]:.2f}; Slaggo-shaped {shaped[0] / shaped[11]:.2f}")
json.dump(dict(shaped_secchi=shaped.tolist(), deriv_secchi=DERIV, scale=scale),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", "winter_light.json"), "w"),
          indent=1)
