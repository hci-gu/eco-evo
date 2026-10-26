"""Light at depth for visual planktivory in the Mareld box (58.1 N): depth of the herring
biting thresholds by month at noon and at given sun elevations, and hours per day in each
light regime. No downloads are read.

Usage: python3 -I light_at_depth.py

Inputs (all stated, see evidence_vertical_overlap.md):
- surface illuminance vs sun elevation: twilight values of the AMS Glossary (0 deg ~500 lx,
  -6 deg ~3 lx, -12 deg ~0.008 lx, -18 deg ~6e-4 lx; log-linear in between); daytime clear sky
  E = 1.3e5 sin(h) lx for h >= 5 deg (judgement, standard order of magnitude), log-linear 0-5 deg;
  overcast factor 0.1-0.3 (judgement); night floor 6e-4 lx (starlight, moonless).
- attenuation: three-layer profile, upper (0-20 m) / middle (20-50 m) / deep (>50 m):
  low-K 0.20/0.08/0.04, central 0.30/0.12/0.06, high-K 0.45/0.15/0.08 m-1. Upper layer from
  Aas et al. 2014 (Zq(1%) = 1.8 Secchi, Skagerrak Secchi 8.3 +- 2.8 m -> Kq 0.23-0.46) and
  Arst et al. 1993 (Skagerrak z(1%) 23 m, mean K ~0.20); deeper layers from Aksnes et al. 2009
  Eq. 2-3 (Kd(500) from salinity, oxygen, chl: S 34-35, O2 5-6 ml/l, chl 0.05-1) - judgement.
- herring biting thresholds: 1e-3 lx (Batty et al. 1990, [S] via sibling note) and
  7e-3 - 3.6e-2 lx (as cited by Onsrud et al. 2004).
"""
import math
import numpy as np

LAT = 58.1
PROFILES = {"lowK": (0.20, 0.08, 0.04), "central": (0.30, 0.12, 0.06), "highK": (0.45, 0.15, 0.08)}
THRESH = {"1e-3 lx": 1e-3, "7e-3 lx": 7e-3, "3.6e-2 lx": 3.6e-2, "1 lx": 1.0}


def surface_lux(h, cloud=1.0):
    pts = [(-18, 6e-4), (-12, 8e-3), (-6, 3.0), (0, 500.0), (5, 1.3e5 * math.sin(math.radians(5)))]
    if h >= 5:
        e = 1.3e5 * math.sin(math.radians(h))
    elif h <= -18:
        e = 6e-4
    else:
        for (h0, e0), (h1, e1) in zip(pts, pts[1:]):
            if h0 <= h <= h1:
                f = (h - h0) / (h1 - h0)
                e = 10 ** (math.log10(e0) + f * (math.log10(e1) - math.log10(e0)))
                break
    return e * (cloud if h > -6 else 1.0)


def depth_of(e0, et, k):
    """Depth where e0 exp(-int K dz) falls to et, K piecewise (0-20, 20-50, >50 m)."""
    if e0 <= et:
        return 0.0
    need = math.log(e0 / et)
    ku, km, kd = k
    if need <= 20 * ku:
        return need / ku
    need -= 20 * ku
    if need <= 30 * km:
        return 20 + need / km
    need -= 30 * km
    return 50 + need / kd


def elevation(day, hour):
    dec = math.radians(23.44) * math.sin(2 * math.pi * (284 + day) / 365.0)
    ha = math.radians(15 * (hour - 12))
    phi = math.radians(LAT)
    s = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(ha)
    return math.degrees(math.asin(max(-1, min(1, s))))


months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
mid = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]

print("1. Noon sun elevation and depth (m) of each threshold at noon, overcast (x0.2) / clear sky")
for name, k in PROFILES.items():
    print(f"\n  K profile {name} {k}")
    print("  month noon_elev  " + "  ".join(f"{t:>16s}" for t in THRESH))
    for m, d in zip(months, mid):
        h = elevation(d, 12)
        cells = []
        for t in THRESH.values():
            z_o = depth_of(surface_lux(h, 0.2), t, k)
            z_c = depth_of(surface_lux(h, 1.0), t, k)
            cells.append(f"{z_o:7.0f}/{z_c:<7.0f}")
        print(f"  {m:5s} {h:9.1f}  " + "  ".join(f"{c:>16s}" for c in cells))

print("\n2. Depth (m) of each threshold at given sun elevations (twilight; cloud ignored below -6 deg)")
for name, k in PROFILES.items():
    print(f"  K profile {name}")
    for h in (0, -4, -6, -8, -10, -12, -18):
        e0 = surface_lux(h, 1.0)
        print(f"    elev {h:4d} deg  E0 {e0:9.3g} lx  " + "  ".join(
            f"{tn}: {depth_of(e0, t, k):5.0f}" for tn, t in THRESH.items()))

print("\n3. Hours per day by regime (mid-month): day (h > 0), civil (0..-6), nautical (-6..-12), night (< -12),"
      " and hours with h > -8 deg (engine threshold)")
tot = np.zeros(5)
for m, d in zip(months, mid):
    hs = [elevation(d, x / 60.0) for x in range(0, 1440)]
    day = sum(h > 0 for h in hs) / 60
    civ = sum(-6 < h <= 0 for h in hs) / 60
    nau = sum(-12 < h <= -6 for h in hs) / 60
    nig = sum(h <= -12 for h in hs) / 60
    g8 = sum(h > -8 for h in hs) / 60
    tot += (day, civ, nau, nig, g8)
    print(f"  {m}: day {day:5.1f}  civil {civ:4.1f}  nautical {nau:4.1f}  night {nig:5.1f}  h>-8 {g8:5.1f}")
tot /= 12
print(f"  annual mean: day {tot[0]:.1f} civil {tot[1]:.1f} nautical {tot[2]:.1f} night {tot[3]:.1f} h>-8 {tot[4]:.1f}"
      f" (fractions {tot[0]/24:.2f}/{tot[1]/24:.2f}/{tot[2]/24:.2f}/{tot[3]/24:.2f}; h>-8 {tot[4]/24:.2f})")
