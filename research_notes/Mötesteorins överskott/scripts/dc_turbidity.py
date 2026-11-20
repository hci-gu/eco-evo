"""[D] Aksnes & Utne (1997) type visual range: R^2 exp(c R) = R0^2 (saturated light, R0 = clear-water range).
Prints R(c) for several clear-water R0 and beam attenuation c, and the ratio to the chain's c 0.25/m, and the
encounter-rate factor (R ratio)^2 (encounter ~ R^2)."""
import math
def R(R0, c):
    lo, hi = 0.0, R0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if m * m * math.exp(c * m) > R0 * R0: hi = m
        else: lo = m
    return lo
print("R0(clear)  c=0.1  c=0.25(chain)  c=0.5  c=1.0   ratio c0.5/c0.25  ratio^2   ratio c1.0/c0.25  ratio^2")
for R0 in (0.3, 0.5, 0.8, 1.2, 2.0):
    r = {c: R(R0, c) for c in (0.1, 0.25, 0.5, 1.0)}
    q1 = r[0.5] / r[0.25]; q2 = r[1.0] / r[0.25]
    print(f"{R0:5.2f}  {r[0.1]:6.3f} {r[0.25]:6.3f} {r[0.5]:6.3f} {r[1.0]:6.3f}   {q1:6.3f} {q1*q1:6.3f}   {q2:6.3f} {q2*q2:6.3f}")
