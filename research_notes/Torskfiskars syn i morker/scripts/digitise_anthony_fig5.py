"""Digitise the cod contrast thresholds (Anthony 1981, Fig. 5) as redrawn in
Aksnes & Utne 1997 (Sarsia 82:137-147), Fig. 5, from a 400-dpi render of page 7.

Usage: python3 -I digitise_anthony_fig5.py <fig5_crop.png> <out.csv>
The crop is page 7 of the PDF rendered at 400 dpi, cropped to
(0.51 w, 0.06 h, 0.95 w, 0.255 h). Axis calibration (pixels in the crop) was
read from the tick marks: x ticks -8..-1 at 126.5, 265, 402, 538.5, 676.5,
812.5, 952, 1087.5; y ticks 0.0/0.2/0.4/0.6 at 584.5/408/234/56.5 on the
y axis; the scan is tilted, the zero line runs 584.5 -> 597.5 px from x 140
to 1080 px. Dots are found as blobs that survive a 5x5 binary erosion (the
model curves are 2-3 px wide and vanish).
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

img, out = sys.argv[1], sys.argv[2]
a = np.array(Image.open(img).convert("L"))
dark = a < 140
core = ndimage.binary_erosion(dark, structure=np.ones((5, 5)))
lab, n = ndimage.label(core)
xt = np.array([126.5, 265, 402, 538.5, 676.5, 812.5, 952, 1087.5])
lx = np.arange(-8, 0)
px_per_unit_y = (584.5 - 56.5) / 0.6
rows = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(xs) < 4:
        continue
    x, y = xs.mean(), ys.mean()
    y0 = 584.5 + 0.01383 * (x - 140)
    if y > y0 + 2 or x < 140:       # tick labels below the axis / y axis
        continue
    ct = (y0 - y) / px_per_unit_y
    logL = np.interp(x, xt, lx)
    rows.append((logL, ct, len(xs)))
rows.sort()
with open(out, "w") as f:
    f.write("log10_radiance_W_m2_sr,contrast_threshold,blob_px\n")
    for r in rows:
        f.write("%.3f,%.4f,%d\n" % r)
print(len(rows), "blobs")
# pixel-weighted mean per radiance level (merged blobs = several stacked dots)
levels = sorted(set(round(r[0], 1) for r in rows))
with open(out.replace(".csv", "_levels.csv"), "w") as f:
    f.write("log10_radiance_W_m2_sr,ct_weighted_mean,ct_min,ct_max,n_blobs\n")
    for L in levels:
        rr = [r for r in rows if round(r[0], 1) == L]
        w = np.array([r[2] for r in rr], float); c = np.array([r[1] for r in rr])
        f.write("%.2f,%.4f,%.4f,%.4f,%d\n" % (L, (w * c).sum() / w.sum(), c.min(), c.max(), len(rr)))
