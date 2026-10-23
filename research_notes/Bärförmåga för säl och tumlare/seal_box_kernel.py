"""Share of harbour seal at-sea usage that falls in the Mareld 60x60 km box
(own calculation behind seals.md section 3; research note, not model code).

Usage around each haul-out anchor ~ exp(-d/L) * w(depth) per km2 of sea,
normalised over all sea cells of a wide EMODnet tile (lat 57.4-59.2,
lon 9.3-12.0). Box = SWEREF 99 TM E 211000-271000, N 6414000-6474000.

Input: the tile as a numpy array (row 0 = 59.2 N, col 0 = 9.3 E, metres,
negative = sea). Fetch and convert it with:
  curl -o wide.tif "https://ows.emodnet-bathymetry.eu/wcs?service=WCS&version=2.0.1&request=GetCoverage&coverageId=emodnet__mean&subset=Lat(57.4,59.2)&subset=Long(9.3,12.0)&format=image/tiff"
  python3 -c "from PIL import Image; import numpy as np; np.save('wide.npy', np.array(Image.open('wide.tif')).astype(float))"
Run: python3 seal_box_kernel.py wide.npy
Output = counted-seal-equivalents (2023 moult counts x share of each colony's
at-sea usage inside the box); multiply by count->total (1.54-1.75), at-sea
fraction (0.75-0.9) and mean mass (50-60 kg) to get tonnes.
"""
import math
import sys
import numpy as np

wide = np.load(sys.argv[1])  # (1728, 2592), lat 59.2 -> 57.4, lon 9.3 -> 12.0
H, W = wide.shape
LAT0, LAT1, LON0, LON1 = 57.4, 59.2, 9.3, 12.0

# aggregate to ~1 km cells: 9 rows (~0.0094 deg lat ~1.04 km) x 18 cols (~0.0188 deg lon ~0.99 km)
br, bc = 9, 18
Hc, Wc = H // br, W // bc
blk = wide[:Hc * br, :Wc * bc].reshape(Hc, br, Wc, bc)
sea_frac = (blk < 0).mean(axis=(1, 3))
depth = np.where(blk < 0, -blk, np.nan)
with np.errstate(all="ignore"):
    zmean = np.nanmean(depth, axis=(1, 3))
lat = LAT1 - (np.arange(Hc) + 0.5) * br * (LAT1 - LAT0) / (H - 1)
lon = LON0 + (np.arange(Wc) + 0.5) * bc * (LON1 - LON0) / (W - 1)
LON, LAT = np.meshgrid(lon, lat)
kx = 111.32 * math.cos(math.radians(58.1))
cell_area = (br * (LAT1 - LAT0) / (H - 1) * 111.2) * (bc * (LON1 - LON0) / (W - 1) * kx)
X = (LON - 11.0) * kx
Y = (LAT - 58.0) * 111.2
area = sea_frac * cell_area  # km2 of sea per cell


def wgs84_to_sweref(lat_deg, lon_deg):
    a = 6378137.0
    f = 1.0 / 298.257222101
    k0, lon0, FE = 0.9996, math.radians(15.0), 500000.0
    e2 = f * (2 - f)
    n = f / (2 - f)
    a_hat = a / (1 + n) * (1 + n * n / 4 + n ** 4 / 64)
    A = e2
    B = (5 * e2 ** 2 - e2 ** 3) / 6
    C = (104 * e2 ** 3 - 45 * e2 ** 4) / 120
    D = (1237 * e2 ** 4) / 1260
    b1 = n / 2 - 2 * n * n / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180
    b2 = 13 * n * n / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440
    b3 = 61 * n ** 3 / 240 - 103 * n ** 4 / 140
    b4 = 49561 * n ** 4 / 161280
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    s = np.sin(phi)
    phi_s = phi - s * np.cos(phi) * (A + B * s ** 2 + C * s ** 4 + D * s ** 6)
    dl = lam - lon0
    xi = np.arctan(np.tan(phi_s) / np.cos(dl))
    eta = np.arctanh(np.cos(phi_s) * np.sin(dl))
    N = k0 * a_hat * (xi + b1 * np.sin(2 * xi) * np.cosh(2 * eta) + b2 * np.sin(4 * xi) * np.cosh(4 * eta)
                      + b3 * np.sin(6 * xi) * np.cosh(6 * eta) + b4 * np.sin(8 * xi) * np.cosh(8 * eta))
    E = FE + k0 * a_hat * (eta + b1 * np.cos(2 * xi) * np.sinh(2 * eta) + b2 * np.cos(4 * xi) * np.sinh(4 * eta)
                           + b3 * np.cos(6 * xi) * np.sinh(6 * eta) + b4 * np.cos(8 * xi) * np.sinh(8 * eta))
    return E, N


E, N = wgs84_to_sweref(LAT, LON)
inbox = (E >= 211000) & (E < 271000) & (N >= 6414000) & (N < 6474000)
print(f"check: box cells {inbox.sum()}, box sea area {area[inbox].sum():.0f} km2, "
      f"mean depth {np.nansum(zmean[inbox]*area[inbox])/area[inbox].sum():.1f} m, "
      f"<50 m {area[inbox & (zmean < 50)].sum():.0f} km2, <100 m {area[inbox & (zmean < 100)].sum():.0f} km2")

# Skagerrak part of the tile: north of the Skagen (57.73 N, 10.6 E) - Goteborg (57.7 N, 11.8 E) line
skag = (LAT > 57.72) & ~((LAT < 57.75) & (LON > 10.6))
for zc in (50, 100):
    print(f"tile Skagerrak sea area < {zc} m: {area[skag & (zmean < zc)].sum():.0f} km2 "
          f"(box {area[inbox & (zmean < zc)].sum():.0f})")
print(f"tile Skagerrak sea area total: {area[skag].sum():.0f} km2")

# colonies: name, (lat, lon) anchors, counted 2023 (Carroll Fig 3 GAM read-off / S1 Table)
colonies = [
    ("Lysekil (Smogen-Gullmarn-Maseskar)", [(58.350, 11.215), (58.255, 11.380), (58.090, 11.330)], 1050),
    ("Marstrand (Pater Noster-Marstrand)", [(57.900, 11.460), (57.875, 11.560)], 1750),
    ("Vaderoarna", [(58.575, 11.060)], 1150),
    ("Koster", [(58.880, 11.020)], 1250),
    ("Hvaler", [(59.050, 10.980)], 400),
    ("Faerder (NO, 2022)", [(59.030, 10.450)], 360),
]


def weight_depth(mode):
    z = np.nan_to_num(zmean, nan=0.0)
    if mode == "none":
        return np.ones_like(z)
    if mode == "mild":
        return np.exp(-z / 100.0)
    if mode == "strong":
        return np.clip((100.0 - z) / 50.0, 0.0, 1.0)  # 1 at <=50 m, 0 at >=100 m
    raise ValueError(mode)


results = {}
for mode in ("none", "mild", "strong"):
    wd = weight_depth(mode) * area
    for L in (10, 15, 25, 35):
        tot_in = 0.0
        shares = []
        for name, anchors, cnt in colonies:
            u = np.zeros_like(area)
            for (la, lo) in anchors:
                ax, ay = (lo - 11.0) * kx, (la - 58.0) * 111.2
                d = np.hypot(X - ax, Y - ay)
                uu = np.exp(-d / L) * wd
                u += uu / uu.sum() / len(anchors)
            sh = u[inbox].sum()
            shares.append(sh)
            tot_in += sh * cnt
        results[(mode, L)] = (tot_in, shares)
        print(f"depth={mode:6s} L={L:2d} km  counted-seal-equivalents in box {tot_in:6.0f}  "
              + "  ".join(f"{c[0].split()[0]}:{s*100:.1f}%" for c, s in zip(colonies, shares)))

