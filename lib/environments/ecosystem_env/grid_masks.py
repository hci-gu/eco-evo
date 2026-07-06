import numpy as np

from lib.environments.ecosystem_env.constants import EAST, NORTH, SOUTH, WEST


def build_movement_mask(grid, migration, dtype):
    H, W = grid.height, grid.width
    move_mask = np.ones((4, H, W), dtype=dtype)

    if not migration:
        move_mask[NORTH, 0, :] = 0.0
        move_mask[SOUTH, -1, :] = 0.0
        move_mask[EAST, :, -1] = 0.0
        move_mask[WEST, :, 0] = 0.0

    access_map = grid.get_map("accessibility")
    if access_map is None:
        return move_mask

    access = access_map.astype(dtype)
    accessible_north = np.ones((H, W), dtype=dtype)
    accessible_south = np.ones((H, W), dtype=dtype)
    accessible_east = np.ones((H, W), dtype=dtype)
    accessible_west = np.ones((H, W), dtype=dtype)
    accessible_north[1:, :] = access[:-1, :]
    accessible_south[:-1, :] = access[1:, :]
    accessible_east[:, :-1] = access[:, 1:]
    accessible_west[:, 1:] = access[:, :-1]

    move_mask[NORTH] *= (accessible_north > 0).astype(dtype)
    move_mask[SOUTH] *= (accessible_south > 0).astype(dtype)
    move_mask[EAST] *= (accessible_east > 0).astype(dtype)
    move_mask[WEST] *= (accessible_west > 0).astype(dtype)
    return move_mask


def build_edge_immigration_weights(grid, dtype):
    H, W = grid.height, grid.width
    edge_mask = np.zeros((H, W), dtype=dtype)
    edge_mask[0, :] = 1.0
    edge_mask[-1, :] = 1.0
    edge_mask[:, 0] = 1.0
    edge_mask[:, -1] = 1.0

    access_map = grid.get_map("accessibility")
    if access_map is None:
        edge_weights = edge_mask
    else:
        edge_weights = edge_mask * np.clip(access_map.astype(dtype), 0.0, None)

    edge_sum = float(edge_weights.sum())
    if edge_sum > 0.0:
        return (edge_weights / edge_sum).astype(dtype)

    n_edge = float(edge_mask.sum())
    if n_edge > 0.0:
        return (edge_mask / n_edge).astype(dtype)
    return np.zeros((H, W), dtype=dtype)
