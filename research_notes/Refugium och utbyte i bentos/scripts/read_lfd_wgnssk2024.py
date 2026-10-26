"""Read the stacked bar heights (total catch, all fractions and sexes) from
WGNSSK 2024 Figure 10.1.4 (Nephrops 3.a, catch LFD 2023, DK+SE, thousands of individuals per 1 mm CL).
Input: PNG of PDF page 41 rendered at 300 dpi with pdftoppm.
Axis calibration from the probe (scripts/probe_axes.py): y grid lines 8000 at row 606.5,
2000 at row 1707.5 -> 0 at row 2075; x grid lines 15 mm at col 606.5, 5 mm = 74 px.
Also reads the height of the two "consumption" (landings) colours vs the discard colours.
Usage: python3 -I read_lfd_wgnssk2024.py <png> <out_csv>
"""
import sys
import numpy as np
from PIL import Image

img = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
Y0, Y8000 = 2075.0, 606.5
PX_PER_UNIT = (Y0 - Y8000) / 8000.0
X15, PX_PER_MM = 606.5, 74.0 / 5.0
TOP, BOTTOM = 466, 2074
sat = img.max(axis=2) - img.min(axis=2)

rows = []
for L in range(14, 77):
    xc = int(round(X15 + (L - 15) * PX_PER_MM))
    tops = []
    for x in range(xc - 3, xc + 4):
        col = sat[TOP:BOTTOM, x]
        idx = np.where(col > 25)[0]
        tops.append(TOP + idx[0] if idx.size else BOTTOM + 1)
    top = float(np.median(tops))
    total = max(0.0, (Y0 - top) / PX_PER_UNIT)
    # landings part: strongly saturated red (230,50,70) or blue (90,80,230) pixels
    col_rgb = img[TOP:BOTTOM, xc]
    r, g, b = col_rgb[:, 0], col_rgb[:, 1], col_rgb[:, 2]
    land = ((r > 180) & (g < 110) & (b < 120)) | ((b > 180) & (r < 140) & (g < 140))
    li = np.where(land)[0]
    landings = max(0.0, (Y0 - (TOP + li[0])) / PX_PER_UNIT) if li.size else 0.0
    rows.append((L, round(total, 1), round(landings, 1)))

with open(sys.argv[2], "w") as f:
    f.write("CL_mm,catch_thousands,landings_thousands\n")
    for L, t, l in rows:
        f.write(f"{L},{t},{l}\n")
for L, t, l in rows:
    print(L, t, l)
