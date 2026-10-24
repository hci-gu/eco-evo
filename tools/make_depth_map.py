"""Depth map for the Mareld grid from EMODnet Bathymetry (section 207).

    python3 tools/make_depth_map.py [--cache DIR] [--out DIR]

Writes maps/depth.npz, key ``depth``: a 60x60 field of the mean water
depth per 1 km cell in metres (positive down), row 0 = north, column 0 =
west, on the grid of vindparker.png and the fishing maps (SWEREF 99 TM,
E 211000-271000, N 6414000-6474000).

Source: EMODnet Bathymetry WCS, coverage ``emodnet__mean`` (the mean
DTM, ~1/16 arc-minute), fetched as a GeoTIFF for lat 57.7-58.4, lon
9.9-11.3 (the bbox and pixel convention of make_djup.py). Every DTM
pixel centre is projected to SWEREF 99 TM and averaged into the 1 km
cell it falls in, so a cell is the mean over its ~150 DTM pixels rather
than one sample at its centre (the trench slope drops tens of metres
inside a cell). Land pixels (depth >= 0) are left out; the box is all
sea.

The field drives the benthos capacity table
(``capacity_depth_table`` in fgconfig/fg_library.yaml) and is exposed
to the spawn strategies as the reference field ``depth``.
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_fishing_maps import CELL, E0, GRID, N0, wgs84_to_sweref99tm  # noqa: E402

LAT_MIN, LAT_MAX = 57.7, 58.4
LON_MIN, LON_MAX = 9.9, 11.3
URL = ("https://ows.emodnet-bathymetry.eu/wcs?service=WCS&version=2.0.1"
       "&request=GetCoverage&coverageId=emodnet__mean"
       f"&subset=Lat({LAT_MIN},{LAT_MAX})&subset=Long({LON_MIN},{LON_MAX})"
       "&format=image/tiff")


def fetch(cache):
    path = os.path.join(cache, "depth.tif")
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        os.makedirs(cache, exist_ok=True)
        print("Fetching EMODnet Bathymetry WCS ...")
        urllib.request.urlretrieve(URL, path)
    return np.array(Image.open(path), dtype=np.float64)


def cell_means(dtm):
    """Mean depth (m, positive) per grid cell from the DTM pixels."""
    h, w = dtm.shape
    lon = LON_MIN + np.arange(w) / (w - 1) * (LON_MAX - LON_MIN)
    lat = LAT_MAX - np.arange(h) / (h - 1) * (LAT_MAX - LAT_MIN)
    lon, lat = np.meshgrid(lon, lat)
    east, north = wgs84_to_sweref99tm(lon, lat)
    col = np.floor((east - E0) / CELL).astype(np.int64)
    row = GRID - 1 - np.floor((north - N0) / CELL).astype(np.int64)
    keep = (col >= 0) & (col < GRID) & (row >= 0) & (row < GRID) & (dtm < 0)
    flat = row[keep] * GRID + col[keep]
    total = np.bincount(flat, weights=-dtm[keep], minlength=GRID * GRID)
    count = np.bincount(flat, minlength=GRID * GRID)
    if (count == 0).any():
        raise SystemExit(f"{int((count == 0).sum())} cells without DTM pixels")
    return (total / count).reshape(GRID, GRID), count.reshape(GRID, GRID)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default="maps/.cache")
    parser.add_argument("--out", default="maps")
    args = parser.parse_args()
    depth, count = cell_means(fetch(args.cache))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "depth.npz")
    np.savez_compressed(path, depth=depth.astype(np.float32))
    print(f"{path}: {depth.min():.1f}-{depth.max():.1f} m, mean {depth.mean():.1f} m, "
          f"{int(count.min())}-{int(count.max())} DTM pixels per cell")
    for lo, hi in ((0, 100), (100, 150), (150, 200), (200, 300), (300, 1000)):
        print(f"  {lo:4d}-{hi:4d} m: {int(((depth >= lo) & (depth < hi)).sum()):5d} cells")


if __name__ == "__main__":
    main()
