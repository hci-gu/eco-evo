"""Geodesy helpers: WGS84/GRS80 lat/lon -> SWEREF 99 TM (Lantmateriet's Gauss
conformal formulas), and the local direction of grid north."""
import numpy as np

A_AX = 6378137.0
FLAT = 1.0 / 298.257222101
LAM0 = np.radians(15.0)
K0 = 0.9996
FE = 500000.0
FN = 0.0


def to_sweref(lat, lon):
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    e2 = FLAT * (2 - FLAT)
    n = FLAT / (2 - FLAT)
    a_roof = A_AX / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    A = e2
    B = (5 * e2 ** 2 - e2 ** 3) / 6.0
    C = (104 * e2 ** 3 - 45 * e2 ** 4) / 120.0
    D = (1237 * e2 ** 4) / 1260.0
    b1 = n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180
    b2 = 13 * n ** 2 / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440
    b3 = 61 * n ** 3 / 240 - 103 * n ** 4 / 140
    b4 = 49561 * n ** 4 / 161280
    s = np.sin(lat)
    phis = lat - s * np.cos(lat) * (A + B * s ** 2 + C * s ** 4 + D * s ** 6)
    dl = lon - LAM0
    xi = np.arctan2(np.tan(phis), np.cos(dl))
    eta = np.arctanh(np.cos(phis) * np.sin(dl))
    N = K0 * a_roof * (xi + b1 * np.sin(2 * xi) * np.cosh(2 * eta) + b2 * np.sin(4 * xi) * np.cosh(4 * eta)
                       + b3 * np.sin(6 * xi) * np.cosh(6 * eta) + b4 * np.sin(8 * xi) * np.cosh(8 * eta)) + FN
    E = K0 * a_roof * (eta + b1 * np.cos(2 * xi) * np.sinh(2 * eta) + b2 * np.cos(4 * xi) * np.sinh(4 * eta)
                       + b3 * np.cos(6 * xi) * np.sinh(6 * eta) + b4 * np.cos(8 * xi) * np.sinh(8 * eta)) + FE
    return E, N


def grid_convergence(lat, lon, d=1e-4):
    """Angle gamma (rad) of TRUE north measured clockwise from SWEREF grid north."""
    E0, N0 = to_sweref(lat, lon)
    E1, N1 = to_sweref(np.asarray(lat) + d, lon)
    return np.arctan2(E1 - E0, N1 - N0)


def to_grid_components(u_east, v_north, gamma):
    """Rotate geographic (east, north) velocity to SWEREF (grid east, grid north)."""
    uE = u_east * np.cos(gamma) - v_north * np.sin(gamma)
    vN = u_east * np.sin(gamma) + v_north * np.cos(gamma)
    return uE, vN


BOX = dict(E0=211000.0, E1=271000.0, N0=6414000.0, N1=6474000.0)


def in_box(E, N):
    return (E >= BOX["E0"]) & (E <= BOX["E1"]) & (N >= BOX["N0"]) & (N <= BOX["N1"])
