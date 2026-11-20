"""Arithmetic aids for converting the sourced patch statistics to a 1 km x 1 km cell and a 6 h tick.
No data files are read. All inputs are printed with their origin; outputs are [D] on [F/A] ranges and [J] choices.
(1) Spherical variogram (no nugget) correlation at lag h for range a: rho = 1 - 1.5 h/a + 0.5 (h/a)^3 (h < a).
    Ranges: Simard & Lavoie 1999 abstract 'autocorrelated up to distances of 10-15 km' and the small-scale
    anisotropic structure 'radii of ~2 x 5 km'. A nugget (sub-cell swarms, unknown share) lowers every rho by
    the factor (1 - nugget share).
(2) Frozen-field advection decorrelation at a fixed cell: a patch of length L carried past at speed U leaves the
    cell after T = L/U; with an exponential ACF rho(lag) = exp(-lag U / L). U 5-15 cm/s (caller: box deep currents
    along the slope), 2 cm/s (Oslofjord, Christiansen et al. 2024), 20 cm/s (Cochrane et al. 1994 mixed layer).
(3) Averaging a swarm field into a 1 km cell: with n independent swarms per km of track (Klevjer et al. 2010:
    median 1-2 per km by day; Kaartvedt et al. 2005: ~1.35 per km) and a 1 km x 1 km cell crossed by ~1/w
    parallel tracks of swarm width w, the number of swarms per cell is ~ n / w (per km2), and the
    between-cell CV of a Poisson swarm count is 1/sqrt(N). Swarm biomass is itself skewed (Klevjer: >50 % of
    biomass in <1 % of swarms), which raises the CV; printed as a lower bound only."""
import math
print("(1) spherical correlation at lag h (km), no nugget")
for a in (2, 5, 10, 15):
    row = []
    for h in (1, 2, 3, 5):
        r = 1 - 1.5 * h / a + 0.5 * (h / a) ** 3 if h < a else 0.0
        row.append(f"h={h}: {r:.2f}")
    print(f"  range {a:>2} km: " + ", ".join(row))
print("(2) advection decorrelation rho(lag) = exp(-lag*U/L)")
for U in (0.02, 0.05, 0.10, 0.15, 0.20):
    for L in (1, 2, 5, 10, 15):
        km_per_h = U * 3.6
        r6 = math.exp(-6 * km_per_h / L); r24 = math.exp(-24 * km_per_h / L); r72 = math.exp(-72 * km_per_h / L)
        print(f"  U {U*100:4.0f} cm/s ({km_per_h:.2f} km/h) L {L:>2} km: T={L/km_per_h:6.1f} h  rho6h {r6:.2f} rho1d {r24:.2f} rho3d {r72:.2f}")
print("(3) Poisson lower bound on between-cell CV from swarm counts")
for n_per_km in (1.0, 1.35, 2.0):
    for w_m in (40, 50, 120):
        N = n_per_km / (w_m / 1000.0)  # swarms per km2 if swarms are discs of diameter w and n counted per km of track
        print(f"  {n_per_km} swarms/km of track, width {w_m} m -> ~{N:.0f} swarms per km2 cell, CV >= {1/math.sqrt(N):.2f}")
