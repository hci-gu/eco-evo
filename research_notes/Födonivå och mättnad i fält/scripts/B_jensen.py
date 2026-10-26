"""B_jensen.py - scale-transition (Jensen) correction of a Holling type II
response for sub-cell prey heterogeneity.

f(N) = a N / (1 + a h N). With the cell mean N_bar and x = a h N_bar the
ratio R = E[f(N)] / f(N_bar) depends only on x and the shape of the
within-cell density distribution (predator search uniform over the cell,
no aggregative response, no predator-prey covariance).

Second-order (Chesson 1998 scale-transition) approximation:
    R ~ 1 + 0.5 f''(N_bar) Var(N) / f(N_bar) = 1 - CV^2 x / (1 + x)^2
Exact: numerical integration over gamma and lognormal densities.
Occupancy model: prey only in a share p of the cell at density N_bar/p:
    R = p (1 + x) / (p + x)  ->  p as x -> infinity.

Also: grab-scale CV of macrobenthos counts from Downing (1979) Taylor's
power law s^2 = a M^b (M per m2), and the 4-station CV of Nephrops burrow
density from the 2025 Swedish UWTV interim report (km scale, not sub-km).

Run: python3 -I B_jensen.py > ../tables/B_jensen_stdout.txt
"""
import math

import numpy as np
from scipy import integrate, stats


def ratio_dist(x, dist):
    """E[f(N)]/f(1) for N with mean 1, f(N) = N/(1 + x N) (a = 1, h = x)."""
    f = lambda n: n / (1.0 + x * n)
    val, _ = integrate.quad(lambda n: f(n) * dist.pdf(n), 0, np.inf, limit=400)
    return val / f(1.0)


def gamma_mean1(cv):
    k = 1.0 / cv**2
    return stats.gamma(a=k, scale=1.0 / k)


def lognorm_mean1(cv):
    s2 = math.log(1.0 + cv**2)
    s = math.sqrt(s2)
    return stats.lognorm(s=s, scale=math.exp(-0.5 * s2))


def main():
    xs = [0.3, 1.0, 3.0, 10.0, 100.0, 500.0]
    cvs = [0.3, 0.6, 1.0, 2.0, 3.0]
    print("# R = E[f(N)]/f(N_bar), Holling II, x = a h N_bar")
    print("# rows: CV of within-cell prey density; columns: x")
    for name, mk in (("second-order", None), ("gamma", gamma_mean1),
                     ("lognormal", lognorm_mean1)):
        print(f"\n## {name}")
        print("CV    " + " ".join(f"x={x:<7g}" for x in xs))
        for cv in cvs:
            row = []
            for x in xs:
                if mk is None:
                    r = 1.0 - cv**2 * x / (1.0 + x) ** 2
                else:
                    r = ratio_dist(x, mk(cv))
                row.append(f"{r:9.4f}")
            print(f"{cv:<5g} " + " ".join(row))

    print("\n## occupancy model: prey only in share p of the cell, random search")
    ps = [0.9, 0.7, 0.5, 0.3, 0.1]
    print("p     " + " ".join(f"x={x:<7g}" for x in xs))
    for p in ps:
        print(f"{p:<5g} " + " ".join(f"{p * (1 + x) / (p + x):9.4f}" for x in xs))

    print("\n## Downing 1979 Taylor's power law, CV = sqrt(a M^(b-2)), M ind/m2")
    taylor = {"overall (1462 sets)": (5.24, 1.462),
              "Polychaeta (13)": (3.24, 1.561),
              "Pelecypoda (138)": (7.87, 1.496),
              "Amphipoda (92)": (10.45, 1.339),
              "Mud sediment (466)": (5.16, 1.495)}
    Ms = [10, 100, 1000, 5000]
    print("set                    " + " ".join(f"M={m:<6d}" for m in Ms))
    for k, (a, b) in taylor.items():
        print(f"{k:<22s} " + " ".join(f"{math.sqrt(a * m ** (b - 2)):8.3f}" for m in Ms))
    print("# replicate grabs/cores within a site (m to ~100 m); mostly freshwater")
    print("# data sets in Downing; counts, not biomass")

    print("\n## Nephrops burrows, Swedish UWTV 2025, 4 Norwegian Skagerrak stations")
    d = np.array([0.27, 0.32, 0.17, 0.13])
    print(f"mean {d.mean():.4f} /m2, sd {d.std(ddof=1):.4f}, CV {d.std(ddof=1) / d.mean():.3f}"
          " (stations ~5-15 km apart: between-station, not sub-km)")

    print("\n## engine regimes (from the task / summary.md):")
    for label, x in (("benthos pairs a h N", 100), ("benthos pairs a h N", 500),
                     ("fish->fish pairs near the knee", 1.0)):
        for cv in (0.3, 0.6, 1.0):
            r = ratio_dist(x, lognorm_mean1(cv))
            print(f"{label:<32s} x={x:<5g} CV={cv:<4g} lognormal R={r:.4f}"
                  f"  (1-R = {100 * (1 - r):.2f} %)")


if __name__ == "__main__":
    main()
