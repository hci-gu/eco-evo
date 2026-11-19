"""Ke for cod from Anthony's (1981) contrast thresholds (digitised from
Aksnes & Utne 1997 Fig. 5; tables/anthony1981_fig5_digitised_levels.csv).

Aksnes & Utne (1997) eq. 16: Ct = T3 (Ke + Fb) / Fb, Fb background radiance.
In their visual-range model (c = 0) R^2 is proportional to 1/Ct at fixed prey,
so R^2 / R^2_sat = Fb / (Ke + Fb) = g, the library's light factor.
Fits are least squares in log10(Ct) on the per-level weighted means.

Conversions (stated assumptions, not data):
  E = pi * L  (isotropic field; lower bound of E_d / L for a fish viewing
               horizontally in a downwelling field, see summary)
  lx per W m-2 at 490-500 nm: 1 W m-2 = 4.10-4.18 umol m-2 s-1;
  35-54 lx per umol (deep blue-green, the notes' convention) -> 145-226 lx/W.
Usage: python3 -I fit_cod_ke.py <levels.csv>
"""
import sys
import numpy as np
from scipy.optimize import least_squares

d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
logL = d["log10_radiance_W_m2_sr"]; ct = d["ct_weighted_mean"]
L = 10 ** logL


def fit(mask):
    def res(p):
        T, lk = p
        return np.log10(10 ** T * (10 ** lk + L[mask]) / L[mask]) - np.log10(ct[mask])
    r = least_squares(res, x0=[-1.5, -6])
    return 10 ** r.x[0], 10 ** r.x[1], np.sqrt(np.mean(r.fun ** 2))


PI = np.pi
LXW = (145.0, 226.0)


def lx(Lrad):
    return PI * Lrad * LXW[0], PI * Lrad * LXW[1]


print("Per level: logL  Ct  R2_rel(=Ct_min/Ct, rel. to brightest)  E_lx(isotropic, 145-226 lx/W)")
ctmin = ct[-1]
for a, b in zip(logL, ct):
    lo, hi = lx(10 ** a)
    print("  %5.2f  %.4f  %.3f   %.2e - %.2e lx" % (a, b, ctmin / b, lo, hi))

for name, m in [("all 7 levels", np.ones_like(L, bool)),
                ("scotopic, <= -5.5 (3 levels)", logL <= -5.4),
                ("photopic, >= -5.5 (5 levels)", logL >= -5.6),
                ("photopic, >= -4.4 (4 levels)", logL >= -4.5)]:
    T, Ke, rms = fit(m)
    lo, hi = lx(Ke)
    print("%-30s T3 %.4f  Ke %.2e W m-2 sr-1 (log %.2f)  rms(log Ct) %.3f  -> Ke %.1e - %.1e lx"
          % (name, T, Ke, np.log10(Ke), rms, lo, hi))

# Aksnes & Utne's own published single fit, for reference
for nm, Ke in [("A&U 1997 single fit", 2.9e-6), ("A&U photopic/scotopic break (Anthony)", 8e-6)]:
    lo, hi = lx(Ke)
    print("%-38s %.1e W m-2 sr-1 -> %.1e - %.1e lx (isotropic)" % (nm, Ke, lo, hi))

# Empirical R^2 ratios between light levels (no hyperbola), interpolated in log L
print("\nEmpirical R2 ratio between two lights (interp. of 1/Ct in log L; isotropic, 54 lx/umol):")
lxw = LXW[1]
for e_hi, e_lo in [(0.3, 0.01), (0.1, 0.003), (0.03, 0.001), (0.01, 3e-4), (0.01, 1e-4), (0.1, 1e-4)]:
    def r2(E):
        Lr = E / lxw / PI
        return np.interp(np.log10(Lr), logL, ctmin / ct)
    print("  %.0e lx vs %.0e lx: R2 rel %.3f vs %.3f, ratio %.2f" % (e_hi, e_lo, r2(e_hi), r2(e_lo), r2(e_hi) / r2(e_lo)))

# Geometry: a fish viewing horizontally sees a background radiance L_h below
# E_d / pi. F = E_d / (pi L_h) is not measured here [J: 1 (isotropic) to ~30].
print("\nGeometry factor F = E_d/(pi L_bg): empirical R2 ratios and Ke in downwelling lx (54 lx/umol):")
pairs = [(0.1, 0.006), (0.1, 0.003), (0.03, 0.001), (0.01, 3e-4)]
for F in (1, 3, 10, 30):
    def r2(E):
        Lr = E / lxw / PI / F
        return np.interp(np.log10(Lr), logL, ctmin / ct)
    s = "; ".join("%g/%g lx %.2f" % (a, b, r2(a) / r2(b)) for a, b in pairs)
    print("  F %2d: %s" % (F, s))
    print("        Ke all-levels %.1e lx, A&U single fit %.1e lx, photopic(>=-4.4) %.1e lx, scotopic %.1e lx"
          % (1.33e-6 * PI * lxw * F, 2.9e-6 * PI * lxw * F, 3.16e-5 * PI * lxw * F, 3.32e-7 * PI * lxw * F))
print("Note: the scotopic branch has T3 0.100 against 0.035-0.041 for the others, so its plateau R2 is ~1/3 of"
      " the bright one; a 3-point, 2-parameter fit.")
