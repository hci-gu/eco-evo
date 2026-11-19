#!/usr/bin/env python3
"""A PLACEHOLDER wind farm footprint and noise field for the paper's
accumulated-effects scenario, until Poseidon's real outline is in.

Writes ``maps/poseidon_placeholder.npz`` with

* ``mask`` (H, W) in [0, 1]: the share of each cell inside the polygon
  (supersampled), used as the trawling ban;
* ``windfarm_noise`` (H, W) dB: ``inside_db`` inside the footprint,
  ``inside_db - 20 log10(1 + d) - alpha d`` outside, d the distance in km
  (cells are 1 km) to the footprint.

The polygon is an arbitrary pentagon (row, col in cells) over the
trawled 100-250 m beds in the middle of the box. Replace ``--polygon``
(or the whole file) with Poseidon's real outline; the scenario files only
read ``mask`` and ``windfarm_noise``.

    uv run tools/paper/make_windfarm_placeholder.py
"""

import argparse
import json
import os

import numpy as np

DEFAULT_POLYGON = [[22, 27], [21, 36], [28, 41], [36, 37], [34, 26]]


def polygon_share(polygon, H, W, samples=8):
    """Share of each cell inside ``polygon`` [(row, col)], by supersampling."""
    from matplotlib.path import Path
    path = Path(np.asarray(polygon, dtype=float))
    offsets = (np.arange(samples) + 0.5) / samples
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    share = np.zeros((H, W))
    for dr in offsets:
        for dc in offsets:
            pts = np.stack([(rr + dr).ravel(), (cc + dc).ravel()], axis=1)
            share += path.contains_points(pts).reshape(H, W)
    return share / samples ** 2


def distance_km(mask):
    """Distance from each cell centre to the nearest cell at least half inside."""
    inside = np.argwhere(mask >= 0.5)
    H, W = mask.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = np.full((H, W), np.inf)
    for r, c in inside:
        d = np.minimum(d, np.hypot(rr - r, cc - c))
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="maps/poseidon_placeholder.npz")
    parser.add_argument("--grid", type=int, nargs=2, default=[60, 60])
    parser.add_argument("--polygon", type=json.loads, default=DEFAULT_POLYGON,
                        help="JSON list of [row, col] vertices in cells")
    parser.add_argument("--inside-db", type=float, default=110.0)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="extra loss in dB per km outside the footprint")
    args = parser.parse_args()
    H, W = args.grid
    mask = polygon_share(args.polygon, H, W).astype(np.float32)
    d = distance_km(mask)
    noise = np.where(mask >= 0.5, args.inside_db,
                     args.inside_db - 20.0 * np.log10(1.0 + d) - args.alpha * d)
    noise = np.clip(noise, 0.0, None).astype(np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, mask=mask, windfarm_noise=noise,
             unit=np.array("dB re 1 uPa (placeholder)"),
             polygon=np.asarray(args.polygon, dtype=np.float32),
             note=np.array("PLACEHOLDER footprint, not Poseidon's outline"))
    print(f"{args.out}: {mask.sum():.1f} km2 inside, noise >= 80 dB on "
          f"{int((noise >= 80).sum())} cells, >= 100 dB on {int((noise >= 100).sum())}")


if __name__ == "__main__":
    main()
