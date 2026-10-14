"""Fishing-effort maps for the Mareld grid from EMODnet (section 161).

    python3 tools/make_fishing_maps.py [--cache DIR] [--out DIR]

Writes one .npz per gear, a 60x60 field of fishing effort density in
MW-h per km2 per year (vessel power x fishing hours, VMS vessels > 12 m),
row 0 = north, column 0 = west, on the grid of vindparker.png
(SWEREF 99 TM, E 211000-271000, N 6414000-6474000, 1 km cells):

    maps/fishing_bottom_otter_trawls.npz  key bottom_trawling   (gadoids)
    maps/fishing_pelagic.npz              key pelagic_trawling  (pelagic fish)

Source: EMODnet Human Activities WFS layers 'fishingbottomottertrawls' and
'fishingpelagic' (attribute mw_fshn per 0.05 x 0.05 deg c-square and
4-year window avgyears). mw_fshn is per c-square, so it is divided by the
c-square's area (~16.4 km2 at 58 N) to give a density. The c-squares are
rasterised by sampling each one on a fine lat/lon lattice, projecting the
samples to SWEREF 99 TM and adding density x sample area to the 1 km cell
they land in: the area-weighted mean density per cell.

Today's map: the spatial pattern is the mean over all windows of each
window divided by its grid mean (the windows overlap and the Skagerrak-
wide total drops by ~60 % after 2018-2021, so raw averaging would weight
the old effort), scaled to the grid mean of the latest window. The level
does not set the fishing mortality - the impact table's catchability does
(F / mean effort, fgconfig/fg_library.yaml) - but it keeps the unit real,
so a scenario map in the same unit (more, less or closed effort) is read
by the same table. Cells under the Bratten closure come out ~0 by
themselves.
"""
import argparse
import json
import math
import os
import urllib.request

import numpy as np

LAYERS = {
    "fishingbottomottertrawls": ("bottom_trawling", "fishing_bottom_otter_trawls.npz"),
    "fishingpelagic": ("pelagic_trawling", "fishing_pelagic.npz"),
}
BBOX = (57.6, 58.5, 9.8, 11.4)          # lat min/max, lon min/max (> the grid)
GRID = 60
E0, N0, CELL = 211000.0, 6414000.0, 1000.0   # SW corner of the grid, metres
CSQ = 0.05                               # c-square side, degrees
SUB = 80                                 # samples per c-square side


def wgs84_to_sweref99tm(lon, lat):
    """Gauss-Krueger forward projection (GRS80, SWEREF 99 TM); arrays ok."""
    a, f, k0 = 6378137.0, 1.0 / 298.257222101, 0.9996
    lon0, fn, fe = math.radians(15.0), 0.0, 500000.0
    e2 = f * (2 - f)
    n = f / (2 - f)
    a_hat = a / (1 + n) * (1 + n * n / 4 + n ** 4 / 64)
    A = e2
    B = (5 * e2 * e2 - e2 ** 3) / 6
    C = (104 * e2 ** 3 - 45 * e2 ** 4) / 120
    D = 1237 * e2 ** 4 / 1260
    b1 = n / 2 - 2 * n * n / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180
    b2 = 13 * n * n / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440
    b3 = 61 * n ** 3 / 240 - 103 * n ** 4 / 140
    b4 = 49561 * n ** 4 / 161280
    phi, lam = np.radians(lat), np.radians(lon)
    s = np.sin(phi)
    phi_star = phi - s * np.cos(phi) * (A + B * s ** 2 + C * s ** 4 + D * s ** 6)
    dlam = lam - lon0
    xi = np.arctan(np.tan(phi_star) / np.cos(dlam))
    eta = np.arctanh(np.cos(phi_star) * np.sin(dlam))
    north = k0 * a_hat * (xi + b1 * np.sin(2 * xi) * np.cosh(2 * eta)
                          + b2 * np.sin(4 * xi) * np.cosh(4 * eta)
                          + b3 * np.sin(6 * xi) * np.cosh(6 * eta)
                          + b4 * np.sin(8 * xi) * np.cosh(8 * eta)) + fn
    east = k0 * a_hat * (eta + b1 * np.cos(2 * xi) * np.sinh(2 * eta)
                         + b2 * np.cos(4 * xi) * np.sinh(4 * eta)
                         + b3 * np.cos(6 * xi) * np.sinh(6 * eta)
                         + b4 * np.cos(8 * xi) * np.sinh(8 * eta)) + fe
    return east, north


def fetch(layer, cache):
    path = os.path.join(cache, layer + ".json") if cache else None
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)["features"]
    lat0, lat1, lon0, lon1 = BBOX
    url = ("https://ows.emodnet-humanactivities.eu/wfs?SERVICE=WFS&VERSION=1.1.0"
           f"&REQUEST=GetFeature&typeName={layer}&outputFormat=application/json"
           f"&CQL_FILTER=lat%20BETWEEN%20{lat0}%20AND%20{lat1}%20AND%20lon%20BETWEEN"
           f"%20{lon0}%20AND%20{lon1}")
    with urllib.request.urlopen(url, timeout=300) as r:
        data = json.load(r)
    if path:
        os.makedirs(cache, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    return data["features"]


def rasterise(cells):
    """``cells``: [(lat_min, lon_min, density)] -> (GRID, GRID) density."""
    out = np.zeros((GRID, GRID))
    step = CSQ / SUB
    offsets = (np.arange(SUB) + 0.5) * step
    for lat_min, lon_min, density in cells:
        lat = lat_min + offsets[:, None] + 0 * offsets[None, :]
        lon = lon_min + offsets[None, :] + 0 * offsets[:, None]
        area = (step * 111.32) * (step * 111.32 * np.cos(np.radians(lat)))   # km2
        east, north = wgs84_to_sweref99tm(lon, lat)
        col = np.floor((east - E0) / CELL).astype(int)
        row = GRID - 1 - np.floor((north - N0) / CELL).astype(int)
        inside = (col >= 0) & (col < GRID) & (row >= 0) & (row < GRID)
        np.add.at(out, (row[inside], col[inside]), density * area[inside])
    return out / (CELL * CELL / 1e6)     # per km2 of cell


def build(features):
    windows = sorted({f["properties"]["avgyears"] for f in features})
    maps = {}
    for window in windows:
        cells = []
        for f in features:
            p = f["properties"]
            if p["avgyears"] != window:
                continue
            ring = f["geometry"]["coordinates"][0][0]
            lat_min = min(c[1] for c in ring)
            lon_min = min(c[0] for c in ring)
            csq_km2 = (CSQ * 111.32) ** 2 * math.cos(math.radians(lat_min + CSQ / 2))
            cells.append((lat_min, lon_min, float(p["mw_fshn"]) / csq_km2))
        maps[window] = rasterise(cells)
    pattern = np.mean([m / m.mean() for m in maps.values() if m.mean() > 0], axis=0)
    today = pattern * maps[windows[-1]].mean()
    return today.astype(np.float32), maps, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help="directory of cached WFS GeoJSON")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for layer, (impact_id, filename) in LAYERS.items():
        today, maps, windows = build(fetch(layer, args.cache))
        path = os.path.join(args.out, filename)
        np.savez(path, **{impact_id: today},
                 windows=np.array(windows), unit=np.array("MW-h/km2/year"))
        print(f"{layer} -> {path}")
        for w in windows:
            m = maps[w]
            print(f"  {w}: grid mean {m.mean():7.3f}  max {m.max():7.2f}  "
                  f"cells > 0 {int((m > 0).sum()):4d}")
        print(f"  today: grid mean {today.mean():.3f}  max {today.max():.2f}  "
              f"max/mean {today.max() / today.mean():.1f}  "
              f"zero cells {int((today <= 0).sum())}  MW-h/km2/year")


if __name__ == "__main__":
    main()
