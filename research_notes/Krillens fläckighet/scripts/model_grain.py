"""Krill patchiness at the model grain (1 km x 1 km cell, 6 h tick): arithmetic only [D].

Every input is a field number from evidence.md / tables/*.csv, or a stated judgement [J]; nothing comes from model
output. Run: python3 -I model_grain.py  (no arguments; stdout -> tables/model_grain_stdout.txt)

Conventions [D]:
- Two-state equivalent. A field with mean m and CV c behaves, for a predator that is saturated inside patches, like
  a two-state field with occupancy p_eff = 1/(1 + c^2) and in-patch density m / p_eff. 1 + c^2 is Lloyd's patchiness
  (mean crowding / mean), so p_eff = 1 / Lloyd. For an exact two-state field with occupancy p, c^2 = (1 - p)/p and
  p_eff = p.
- Change of support [D/J]. For an exponential process with e-folding length L, the variance of a 1-D average over x
  lengths is the sill times g(x) = 2 (x - 1 + e^-x) / x^2. A CV measured over a transect of length s is moved to a
  1 km line by the factor g(1/L) / g(s/L); the square cell averages more across the line, so this overstates the cell
  CV slightly (understates p_eff). Sub-cell swarm variance (a nugget) is not in the transect CV and averages out.
- Persistence at a fixed cell: T_E = 1 / (1/T_bio + U/L), U the current (advection of patches past the cell), T_bio
  the patch lifetime in the water's frame. Correlation over one tick rho_tick = exp(-6 h / T_E).
- Stomach-side occupancy [D]. A fish whose stomach shows krill for a window W (h) samples n = W / 6 h ticks of its
  own track. With cell-tick occupancy p and correlation rho_tick between successive samples, the effective number of
  independent samples is taken as n_eff = 1 + (n - 1)(1 - rho_tick), and P(krill in stomach) = 1 - (1 - p)^n_eff, so
  p = 1 - (1 - P)^(1 / n_eff). rho_tick = rho_krill(6 h) x exp(-d / L) with rho_krill(6 h) 0.8 [J, from the
  Gullmarsfjord r 0.62-0.85 at 6-12 h] and L 5 km [J, from 2-5 km St Lawrence radii and the 10 km residual Moran's I];
  d the fish's net move per tick. The stomach P is occupancy x P(eats | in krill), so p is a LOWER bound on occupancy.
"""
import math


def g(x):
    if x < 1e-9:
        return 1.0
    return 2.0 * (x - 1.0 + math.exp(-x)) / (x * x)


def peff(cv2):
    return 1.0 / (1.0 + cv2)


def stomach_p(P, W_h, rho_tick):
    n = W_h / 6.0
    n_eff = max(1.0, 1.0 + (n - 1.0) * (1.0 - rho_tick))
    return 1.0 - (1.0 - P) ** (1.0 / n_eff), n, n_eff


def two_state_intake(X_k, X_o, p):
    """Mean saturating potential X/(1+X) when the krill term X_k is concentrated in a share p of cell-ticks at X_k/p
    (zero elsewhere), the other prey X_o smooth, the predator NOT tracking krill; relative to the smooth field."""
    smooth = (X_k + X_o) / (1 + X_k + X_o)
    patchy = p * (X_o + X_k / p) / (1 + X_o + X_k / p) + (1 - p) * X_o / (1 + X_o)
    return patchy / smooth


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    print("g(x) check: g(0.1) %.3f g(1) %.3f g(5) %.3f g(20) %.3f\n" % (g(0.1), g(1), g(5), g(20)))

    print("== A. Acoustic / net field statistics -> two-state occupancy p_eff = 1/(1+CV^2) at the stated support")
    rows = [
        # label, CV, support, tag
        ("Gullmarsfjord adult M. norvegica, day, between transects 2-8 (Everson 2007 T5)", 0.54, "transect ~1-2 km, fjord", "D on F"),
        ("  same, day, range of survey CVs (low end)", 0.34, "", "D on F"),
        ("  same, day, range of survey CVs (high end)", 0.73, "", "D on F"),
        ("Gullmarsfjord, night, transects 2-8", 0.35, "transect ~1-2 km, fjord", "D on F"),
        ("Bohle 1991 IKMT fixed stations, time-at-point, implied lognormal (median station)", 1.29, "2.3 km tow, weeks lag, night 5-65 m", "D on F"),
        ("Bohle 1991 in-box station-winter means, between stations", 1.62, "season mean per station, 5-40 km apart", "D on F"),
        ("  same without st. 51 1989-90 (one 30 l haul)", 0.81, "", "D on F"),
        ("IMR NSEC trawl, east/Skagerrak stations", 2.02, "station, tens of km apart, Apr-May", "D on F"),
        ("IMR NSEC trawl, west/north Trench stations", 0.46, "station, tens of km apart, Apr-May", "D on F"),
        ("IMR NSEC trawl, all 18 stations without st. 418 (440 outlier)", 0.66, "station, tens of km apart, Apr-May", "D on F"),
    ]
    for lab, cv, sup, tag in rows:
        print(f"  {lab:86s} CV {cv:4.2f} Lloyd {1 + cv * cv:4.2f} p_eff {peff(cv * cv):4.2f}  [{tag}] {sup}")
    print("  Two-state occupancy read directly: Bohle mean/max quartiles 0.30 / 0.44 / 0.50 (night, 2.3 km, weeks);"
          " Bamstedt & Karlson 1998 share of Skagerrak stations with krill 0.18-0.54 [F, q1 R-table];"
          " Antarctic (far proxy) 1-km resets with a swarm by day 0.49-0.69 (Klevjer 2010) [F].")

    print("\n== B. Change of support from a ~1.5 km transect to the 1 km cell (exponential structure, e-folding L)")
    for L in (2.0, 5.0, 10.0):
        f1 = g(1.0 / L) / g(1.5 / L)
        print(f"  L {L:4.1f} km: day CV at 1 km ~ {0.54 * math.sqrt(f1):4.2f} -> p_eff {peff(0.54 ** 2 * f1):4.2f};"
              f" night CV {0.35 * math.sqrt(f1):4.2f} -> p_eff {peff(0.35 ** 2 * f1):4.2f}")
    print("  (the transect is a line; the cell is a square, which averages more across the line: these are upper bounds on"
          " the cell CV, i.e. lower bounds on p_eff, within ~5 %)")

    print("\n== C. Decorrelation time at a fixed cell, T_E = 1/(1/T_bio + U/L)")
    for L in (2.0, 5.0, 10.0):
        for U_cms in (2.0, 5.0, 15.0):
            U = U_cms * 0.036  # km/h
            for Tb in (24.0, 72.0):
                TE = 1.0 / (1.0 / Tb + U / L)
                print(f"  L {L:4.1f} km U {U_cms:4.1f} cm/s T_bio {Tb:4.0f} h: T_E {TE:5.1f} h = {TE / 6:4.1f} ticks; rho(6 h) {math.exp(-6 / TE):4.2f}")
    print("  Field anchors: Gullmarsfjord pattern r 0.62-0.85 at 6-12 h, one reorganisation inside 3 d, then r 0.78-0.99"
          " over 4-49 h (Everson T5, D on F); Scotian basin 25-h means swing x2-4 over 1-3 d (Cochrane 1994, D from figure);"
          " whiting stomach krill vanished after 24-42 h at fixed stations (Rindorf 2003, F).")

    print("\n== D. Stomach-side occupancy per cell-tick from per-fish FO over a detection window")
    print("  GER (WGSAM whiting set, 28 cm, 8.4 C, krill 4.3-6 kJ/g): one 0.2 g krill 90 % gone in 7-9 h, a 1 %BW meal in"
          " 22-29 h, a 3 %BW meal in 37-49 h [D] -> window W 12 / 24 / 36 h (2 / 4 / 6 ticks)")
    print("  P = per-fish FO of krill: saithe single fish 0.34 (n 122), leave-one-out FO beside a krill eater 0.55 (n 20), whiting krill-present periods 0.20 (Rindorf 2003),"
          " Oslofjord in the layer 0.35-0.80 (Onsrud 2004), haul share with krill ~0.5 (upper, any fish in the haul)")
    print("  rho_tick = rho_krill(6 h) 0.8 [J] x exp(-d/L), d the fish's net move per tick (whiting 0.6 km, saithe 3 km), L 5 km [J]")
    for P in (0.2, 0.35, 0.5):
        for W in (12.0, 24.0, 36.0):
            for lab, rho in (("whiting rho 0.70", 0.8 * math.exp(-0.6 / 5)), ("saithe rho 0.44", 0.8 * math.exp(-3 / 5)), ("independent", 0.0)):
                p, n, ne = stomach_p(P, W, rho)
                print(f"  P {P:4.2f} W {W:4.0f} h (n {n:3.1f}) {lab:17s}: n_eff {ne:4.2f} -> p per cell-tick {p:4.2f}")
    print("  Ration ratios (central band, band cancels): all / krill-positive hauls 0.57-0.71 (whiting krill strata),"
          " 0.45 whiting all strata, 0.42 saithe; all / krill-dominated 0.28-0.43 whiting krill strata, 0.22 whiting all"
          " strata, 0.23 saithe (stomach_occupancy_stdout)")

    print("\n== E. Effect size on a saturating predator that does not track krill (yardstick only, nothing chosen from it)")
    print("  X = sum a h B_vis from 'Intag under fodosok' section 0 (resume 289 decomposition): whiting X 5.45, krill share 0.82;"
          " water column X 3.55, krill share 0.46")
    for name, X, ks in (("whiting", 5.45, 0.82), ("water column", 3.55, 0.46)):
        Xk, Xo = X * ks, X * (1 - ks)
        out = ", ".join(f"p {p:4.2f}: x{two_state_intake(Xk, Xo, p):4.2f}" for p in (0.9, 0.77, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05))
        print(f"  {name:12s} potential relative to the smooth field: {out}")
        for target in (0.66, 0.52):
            lo, hi = 1e-4, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if two_state_intake(Xk, Xo, mid) > target:
                    hi = mid
                else:
                    lo = mid
            print(f"    p giving x{target:.2f}: {0.5 * (lo + hi):.3f}" if two_state_intake(Xk, Xo, 1e-4) < target else f"    x{target:.2f} not reachable at any p")
    print("  (the foraging-time factor of 'Intag' section 0: whiting x0.52-0.66 takes the potential 0.74 to 0.39-0.49; water column x0.53-0.66. Floor check: whiting at p -> 0 is x0.586 x 0.74 = 0.43, against 0.45 with krill at 0 in 'Intag' (the krill share keeps its in-patch saturation in p, not 0))")
