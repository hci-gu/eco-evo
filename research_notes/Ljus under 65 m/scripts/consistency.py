"""Consistency checks between the two glider radiometers, and a within-cast test of the near-bed layer.

Usage: python3 -I consistency.py > ../tables/consistency_stdout.txt

A. OCR-equivalent floor on the MPE-PAR casts. The OCR-504 Ed490 slabs pass only where Ed490 - dark >= 10 noise.
   The PAR level at that point is found from the OCR casts' own (Ed490 - dark) : PAR ratio at 50-80 m. The MPE-PAR
   K at 65-125 m is then recomputed using only bins with PAR >= that level, by bed class, to show how much of the
   OCR / MPE difference is the clear-cast selection.
B. OCR PAR against MPE PAR at 20-65 m, where both are well above their floors, matched by month and bed class.
C. Near-bed layer within casts. For MPE casts over 150-250 m beds whose lowest usable slab lies < 50 m above the
   bed, the difference between that slab's K and the K of the slab 45-60 m higher in the same cast. The backscatter
   ratio is taken over the same pair.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this scripts/ folder only
import voto_k as vk  # noqa: E402

SL = vk.SLABS


def main():
    meta = json.load(open(os.path.join(vk.DATA, "meta.json")))
    ocr = [d for d, m in meta.items() if "OCR" in m["attrs"].get("irradiance", "")]
    mpe = [d for d, m in meta.items() if "MPE" in m["attrs"].get("irradiance", "")]
    # --- A: OCR PAR level where Ed490 reaches 10 noise
    ratios, floors490 = [], []
    ocr_par_k = defaultdict(list)
    for ds in ocr:
        casts, _ = vk.load(ds)
        d4, n4, _ = vk.darkfloor(casts, "down_irradiance_490")
        dp, npar, _ = vk.darkfloor(casts, vk.CH_PAR)
        if 15 in n4:
            floors490.append(10 * n4[15])
        for c in casts:
            if c["sun"] <= 5:
                continue
            for k in range(10, 16):
                if k in c["bins"] and k in d4 and k in dp:
                    e = c["bins"][k].get("down_irradiance_490", np.nan) - d4[k]
                    p = c["bins"][k].get(vk.CH_PAR, np.nan) - dp[k]
                    if np.isfinite(e) and np.isfinite(p) and p > 20 * max(npar[k], 1e-6) and e > 20 * n4[k]:
                        ratios.append(e / p)
            E = vk.corrected(c, vk.CH_PAR, dp, npar, False)
            m = int(str(c["t"])[5:7])
            bc = vk.bed_class(vk.bed_of(c))
            for lo, hi in SL[:3]:
                kk = vk.slab_k(E, lo, hi)
                if kk is not None:
                    ocr_par_k[(m, bc, lo)].append(kk)
    r = float(np.median(ratios))
    f490 = float(np.median(floors490))
    par_floor = f490 / r
    print(f"A. OCR casts: (Ed490 - dark)/PAR at 50-80 m = {r:.3f} (Ed490 unit per umol m-2 s-1, n {len(ratios)});"
          f" Ed490 10-noise level {f490:.4f} -> PAR {par_floor:.3f} umol m-2 s-1")
    mk = defaultdict(list)
    mpe_par_k = defaultdict(list)
    near = defaultdict(list)
    for ds in mpe:
        casts, _ = vk.load(ds)
        dark, noise, _ = vk.darkfloor(casts, vk.CH_PAR)
        for c in casts:
            if c["sun"] <= 5:
                continue
            E = vk.corrected(c, vk.CH_PAR, dark, noise, True)
            Eo = {k: v for k, v in E.items() if v * 1e4 >= par_floor}
            bed = vk.bed_of(c)
            bc = vk.bed_class(bed)
            m = int(str(c["t"])[5:7])
            for lo, hi in SL[3:7]:
                a = vk.slab_k(E, lo, hi)
                b = vk.slab_k(Eo, lo, hi)
                if a is not None:
                    mk[("all", bc, lo)].append(a)
                if b is not None:
                    mk[("ocrfloor", bc, lo)].append(b)
            for lo, hi in SL[:3]:
                kk = vk.slab_k(E, lo, hi)
                if kk is not None:
                    mpe_par_k[(m, bc, lo)].append(kk)
            # C: within-cast near-bed contrast
            if np.isfinite(bed) and 150 <= bed < 250:
                ks = {lo: vk.slab_k(E, lo, hi) for lo, hi in SL[2:]}
                ks = {lo: k for lo, k in ks.items() if k is not None and bed - (lo + 7.5) >= 0}
                if not ks:
                    continue
                low = max(ks)
                hab = bed - (low + 7.5)
                if hab < 50:
                    up = [lo for lo in ks if 45 <= low - lo <= 60]
                    if up:
                        u = max(up)
                        bbv = lambda lo: np.nanmean([c["bins"][k].get("particulate_backscatter", c["bins"][k].get("backscatter", np.nan))  # noqa: E731
                                                     for k in c["bins"] if lo <= 5 * k + 2.5 < lo + 15])
                        hc = "0-25" if hab < 25 else "25-50"
                        near[hc].append((ks[low] - ks[u], ks[low], ks[u], bbv(low) / bbv(u)))
    print("   MPE-PAR K by bed class: all usable bins vs only bins above the OCR-equivalent floor (median, n):")
    for bc in ("100-150", "150-250", ">=250"):
        line = []
        for lo, hi in SL[3:7]:
            a, b = mk[("all", bc, lo)], mk[("ocrfloor", bc, lo)]
            if len(b) >= 10:
                line.append(f"{lo}-{hi}: {np.median(a):.3f} n{len(a)} -> {np.median(b):.3f} n{len(b)}")
        print(f"     bed {bc:>7s}: " + "; ".join(line))
    print("\nB. K_PAR at 20-65 m, OCR-504 (2020-22) vs MPE-PAR (2024-26), same month and bed class (median, n >= 20 each):")
    diffs = []
    for key in sorted(ocr_par_k):
        a, b = ocr_par_k[key], mpe_par_k.get(key, [])
        if len(a) >= 20 and len(b) >= 20:
            d = np.median(b) - np.median(a)
            diffs.append(d)
            print(f"   month {key[0]:2d} bed {key[1]:>7s} {key[2]}-{key[2] + 15} m: OCR {np.median(a):.3f} n{len(a)}, "
                  f"MPE {np.median(b):.3f} n{len(b)}, MPE - OCR {d:+.3f}")
    if diffs:
        print(f"   median MPE - OCR over matched cells: {np.median(diffs):+.3f} (range {min(diffs):+.3f} to {max(diffs):+.3f}, n {len(diffs)})")
    print("\nC. Within-cast near-bed contrast, MPE casts over 150-250 m beds (lowest usable slab < 50 m above the bed"
          " vs the slab 45-60 m higher):")
    for hc, v in near.items():
        a = np.array(v)
        print(f"   lowest slab {hc} m above bed: n {len(a)}, K low {np.median(a[:, 1]):.3f}, K higher {np.median(a[:, 2]):.3f},"
              f" difference median {np.median(a[:, 0]):+.3f} [p25 {np.percentile(a[:, 0], 25):+.3f}, p75 {np.percentile(a[:, 0], 75):+.3f}],"
              f" backscatter ratio low/high {np.nanmedian(a[:, 3]):.2f}")


if __name__ == "__main__":
    main()
