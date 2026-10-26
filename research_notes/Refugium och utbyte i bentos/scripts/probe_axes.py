"""Probe a rendered figure PNG: find the plot frame and grid lines (dark/grey pixel runs).
Usage: python3 -I probe_axes.py <png>"""
import sys
import numpy as np
from PIL import Image

img = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
h, w, _ = img.shape
gray = img.mean(axis=2)
sat = img.max(axis=2) - img.min(axis=2)
dark = (gray < 120) & (sat < 30)
# columns / rows with long dark runs = frame lines
colsum = dark.sum(axis=0)
rowsum = dark.sum(axis=1)
print("image", w, h)
print("dark columns (>300 px):", [x for x in range(w) if colsum[x] > 300])
print("dark rows (>300 px):", [y for y in range(h) if rowsum[y] > 300])
# light grey grid lines inside the frame
lg = (gray > 200) & (gray < 245) & (sat < 10)
print("grid columns (>500 px):", [x for x in range(w) if lg[:, x].sum() > 500])
print("grid rows (>500 px):", [y for y in range(h) if lg[y, :].sum() > 500])
