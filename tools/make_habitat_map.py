"""Shallow-bed map for the water column's habitat term (section 300).

    python3 tools/make_habitat_map.py [--depth maps/depth.npz] [--out maps/shallow_beds.npz]

Writes key ``shallow_beds``: 1.0 on every cell whose mean bed depth is
under 150 m (the budget_gate bed classes < 100 and 100-150 m,
``field_diet.bed_class`` 0 and 1), 0.0 elsewhere, on the 60x60 grid of
maps/depth.npz. The impact loader resamples it nearest-neighbour like the
depth map, so both select the same cells on any grid.

The map is a physical field (where the beds are shallow); what it costs a
group is that group's ``<fg>_impacted_by_shallow_beds`` table in the
library. Section 300: only the water column has one.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lib.diagnostics.field_diet import bed_class  # noqa: E402


def shallow_beds(depth):
    return (bed_class(depth) <= 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--depth', default=os.path.join(ROOT, 'maps', 'depth.npz'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'maps', 'shallow_beds.npz'))
    args = parser.parse_args()
    depth = np.load(args.depth)['depth']
    field = shallow_beds(depth)
    np.savez_compressed(args.out, shallow_beds=field)
    print(f'{args.out}: {field.shape}, {int(field.sum())} of {field.size} cells under 150 m '
          f'({field.mean():.3f})')


if __name__ == '__main__':
    main()
