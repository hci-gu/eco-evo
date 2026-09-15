import numpy as np

# Fix 2 (Section 73): maximum fraction of the *visible* prey biomass that
# predation may remove from a single cell in a single tick. Strictly < 1
# so that `B_old - intake` can never round to exactly 0.0 in float32.
#
# The old cap, `B_vis / (total_demand + 1e-9)`, left behind only
# `B_vis * 1e-9 / (D + 1e-9)`. Relative precision in float32 is ~1.2e-7,
# i.e. LARGER than that residue, so an overharvested cell was zeroed
# exactly. Zero is an absorbing state: logistic growth is multiplicative
# in B, phytoplankton has movement_speed 0 (no diffusion from neighbours)
# and Holling type III is multiplicative in prey density, so the type III
# refuge can never restore a cell from 0. Keeping a 0.1 % survivor margin
# makes that refuge reachable. The margin is far above float32 epsilon and
# far below any ecologically meaningful biomass.
MAX_HARVEST_FRAC = np.float32(0.999)

MOVE_SLICE = slice(0, 4)
REST_INDEX = 4
EAT_START = 5

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

DIRECTIONS = ("N", "E", "S", "W")
