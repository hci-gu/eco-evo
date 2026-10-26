#!/usr/bin/env python3
"""Sub-cell prey patchiness: how much does mean-field Holling overestimate intake?

Scale-transition arithmetic (Chesson 1998, 2012; Englund & Leonardsson 2008):
if the prey density N inside a 1 km cell varies with mean m and coefficient
of variation c, and a foraging fish samples the cell at random (no
aggregative response), its mean intake is E[f(N)], not f(m), with
f(N) = a N / (1 + a h N). Because f is concave, E[f(N)] < f(m); the
correction is second order in c for small x = a h m (~ 1 - c^2 x/(1+x)^2)
and vanishes for a saturated pair (x >> 1) unless a share of the cell is
effectively bare.

Distributions:
  * gamma with CV c (negative-binomial-like aggregation, Taylor's power law
    at a fixed mean);
  * zero-inflated: a bare share z (no accessible prey) and the rest gamma
    with the same cell mean.
A fish that aggregates on the rich patches (aggregative response) gets
MORE than f(m) - the opposite sign - so random search is the conservative
(largest-penalty) case.

Prints E[f(N)] / f(m) for a grid of x and c, and the x values of the
engine pairs (engine_feeding_level.py output).
"""
import numpy as np
from math import gamma as G

rng = np.random.default_rng(20261026)
n = 400_000


def ratio_gamma(x, c):
    if c == 0:
        return 1.0
    k = 1.0 / (c * c)
    N = rng.gamma(k, 1.0 / k, n)  # mean 1, CV c
    f = x * N / (1.0 + x * N)
    return f.mean() / (x / (1.0 + x))


def ratio_zero_inflated(x, z, c):
    k = 1.0 / (c * c)
    N = rng.gamma(k, 1.0 / k, n) / (1.0 - z)  # cell mean stays 1
    bare = rng.random(n) < z
    N[bare] = 0.0
    f = x * N / (1.0 + x * N)
    return f.mean() / (x / (1.0 + x))


xs = [0.3, 1.0, 3.0, 10.0, 44.0, 87.0, 266.0]
cs = [0.5, 1.0, 1.5, 2.0, 3.0]
print("E[f(N)] / f(mean), gamma-distributed prey inside the cell (random search)")
print("x \\ CV " + "".join(f"{c:>8.1f}" for c in cs))
for x in xs:
    print(f"{x:7.1f} " + "".join(f"{ratio_gamma(x, c):8.3f}" for c in cs))
print()
print("Zero-inflated: bare share z, the rest gamma CV 1, same cell mean")
zs = [0.1, 0.3, 0.5, 0.7, 0.9]
print("x \\ z  " + "".join(f"{z:>8.1f}" for z in zs))
for x in xs:
    print(f"{x:7.1f} " + "".join(f"{ratio_zero_inflated(x, z, 1.0):8.3f}" for z in zs))
print()
print("Engine x at today's density: benthivores -> infauna 266, -> epifauna 87,")
print("predators -> epifauna 44, fish -> fish pairs 3.6-10 (engine_feeding_level_stdout.txt).")
print("Reading: for a saturated pair the mean-field error is ~ the share of the cell that is")
print("effectively bare (density below the half-saturation). CV <= 1 costs <= 5 % at x >= 44;")
print("CV 2-3 costs 20-50 % only because a gamma with CV > 1 puts much of the area near zero.")
print("Caveat (J): this Jensen penalty assumes saturation acts on the patch scale. The engine's")
print("1/h is a gastric-evacuation (digestion) ceiling with a gut time of ~1-2 days; a fish that")
print("crosses many sub-km patches per gut time integrates the encounters linearly, so the")
print("penalty applies only if fish stay within one patch for longer than the gut time.")
