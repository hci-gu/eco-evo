"""Spectral composition of the deep box light, from the glider OCR-504 K(490) and K(532): lx per umol and the
gadoid rod band.

Usage: python3 -I spectral.py > ../tables/spectral_stdout.txt   (needs ../tables/voto_k.json from voto_k.py)

Model [D]: in each slab, K(l) = Kw(l) + X exp(-S (l - 490)), with Kw = a_w (Pope & Fry 1997, 10 nm values
as recalled) + 0.5 b_w (b_w = 0.0026 (l/490)^-4.32); X and S solved from the measured pooled K490 and K532 of that slab.
X lumps CDOM, particles and phytoplankton (a smooth, blue-rising term); this is the minimum-assumption shape
and is checked against the measured K_PAR of the same OCR-504 casts.
Surface photon spectrum: flat in quanta over 400-700 nm (the shape at 20 m barely matters below 80 m; checked
with a blue-poor alternative, quanta x (l/550)^2 below 550 nm).
Response curves: CIE photopic V(l) (683 lm/W) and a Govardovskii et al. 2000 A1 rhodopsin template at 496 nm
(gadoid rod, alpha band) [S for the 496 nm value].
Below the deepest slab with both bands measured, K(l) of that slab is held.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = np.arange(400, 701, 10.0)
AW = np.array([0.00663, 0.00473, 0.00454, 0.00495, 0.00635, 0.00922, 0.00979, 0.0106, 0.0127, 0.0150, 0.0204,
               0.0325, 0.0409, 0.0434, 0.0474, 0.0565, 0.0619, 0.0695, 0.0896, 0.1351, 0.2224, 0.2644, 0.2755,
               0.2916, 0.3108, 0.3400, 0.4100, 0.4390, 0.4650, 0.5160, 0.6240])
KW = AW + 0.5 * 0.0026 * (L / 490.0) ** -4.32
V = np.array([0.0004, 0.0012, 0.0040, 0.0116, 0.023, 0.038, 0.060, 0.091, 0.139, 0.208, 0.323, 0.503, 0.710, 0.862,
              0.954, 0.995, 0.995, 0.952, 0.870, 0.757, 0.631, 0.503, 0.381, 0.265, 0.175, 0.107, 0.061, 0.032,
              0.017, 0.0082, 0.0041])
H, C, NA = 6.626e-34, 2.998e8, 6.022e23


def rod(lmax=496.0):
    x = lmax / L
    a = 0.8795 + 0.0459 * math.exp(-((lmax - 300) ** 2) / 11940.0)
    return 1.0 / (np.exp(69.7 * (a - x)) + np.exp(28 * (0.922 - x)) + np.exp(-14.9 * (1.104 - x)) + 0.674)


R = rod()


def kw_at(l):
    return float(np.interp(l, L, KW))


def solve(k490, k532):
    x490 = k490 - kw_at(490)
    x532 = k532 - kw_at(532)
    s = math.log(x490 / x532) / 42.0
    return x490, s


def lux_per_umol(q):
    """q: photon spectrum (umol m-2 s-1 nm-1) on L -> lx per umol."""
    w = q * 1e-6 * NA * H * C / (L * 1e-9)
    return 683.0 * np.trapz(V * w, L) / np.trapz(q, L)


def main():
    res = json.load(open(os.path.join(HERE, "..", "tables", "voto_k.json")))["pooled"]
    slabs = [(20, 35), (35, 50), (50, 65), (65, 80), (80, 95)]
    fits = []
    print("Slab fits from pooled OCR-504 K (median over casts):")
    for lo, hi in slabs:
        k490 = res[f"Ed490 {lo}-{hi}"][2]
        k532 = res[f"Ed532 {lo}-{hi}"][2]
        kpar = res.get(f"OCR504 {lo}-{hi}", [None, None, float("nan")])[2]
        x, s = solve(k490, k532)
        kl = KW + x * np.exp(-s * (L - 490))
        fits.append((lo, hi, kl))
        print(f"  {lo}-{hi} m: K490 {k490:.3f}, K532 {k532:.3f} -> X490 {x:.4f}, S {s:.4f} /nm; min K {kl.min():.3f} at "
              f"{L[np.argmin(kl)]:.0f} nm; K(470) {np.interp(470, L, kl):.3f}, K(500) {np.interp(500, L, kl):.3f}, "
              f"K(550) {np.interp(550, L, kl):.3f}, K(570) {np.interp(570, L, kl):.3f}; measured K_PAR {kpar:.3f}")
    for label, q0 in (("flat quanta at 20 m", np.ones_like(L)),
                      ("blue-poor at 20 m", np.where(L < 550, (L / 550.0) ** 2, 1.0))):
        print(f"\nPropagation from 20 m ({label}); K(l) of 80-95 m held below 95 m:")
        print("   z    K_PAR(model)  K_rod496(model)  K490  peak nm  lx/umol  rod/PAR (rel. to 20 m)")
        zs = np.arange(20, 301, 1.0)
        q = q0.copy()
        out = []
        prev = None
        r20 = None
        for z in zs:
            if prev is not None:
                kl = next((f[2] for f in fits if f[0] <= z - 0.5 < f[1]), fits[-1][2])
                q = q * np.exp(-kl * 1.0)
            par = np.trapz(q, L)
            rr = np.trapz(q * R, L)
            if r20 is None:
                r20 = rr / par
            out.append((z, par, rr, np.interp(490, L, q)))
            prev = z
        out = np.array(out)
        for zt in (50, 65, 80, 100, 125, 150, 200, 250):
            i = int(zt - 20)
            j = max(i - 10, 0)
            kpar = -math.log(out[i, 1] / out[j, 1]) / (out[i, 0] - out[j, 0])
            krod = -math.log(out[i, 2] / out[j, 2]) / (out[i, 0] - out[j, 0])
            k490 = -math.log(out[i, 3] / out[j, 3]) / (out[i, 0] - out[j, 0])
            # spectrum at zt
            qz = q0.copy()
            for zz in range(20, zt):
                kl = next((f[2] for f in fits if f[0] <= zz + 0.5 < f[1]), fits[-1][2])
                qz = qz * np.exp(-kl)
            print(f"  {zt:4d}   {kpar:.4f}        {krod:.4f}          {k490:.4f}  {L[np.argmax(qz)]:.0f}     "
                  f"{lux_per_umol(qz):6.1f}   {out[i, 2] / out[i, 1] / r20:.2f}")
    print("\nMonochromatic lx per umol m-2 s-1: " + ", ".join(
        f"{l:.0f} nm {lux_per_umol(np.where(L == l, 1.0, 0.0) + 1e-12):.0f}" for l in (470, 490, 500, 530, 550)))


if __name__ == "__main__":
    main()
