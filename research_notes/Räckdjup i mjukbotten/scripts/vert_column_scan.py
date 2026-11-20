"""Print grey-level transitions down one pixel column of a figure crop.

Usage: python3 -I vert_column_scan.py PNG X [X ...]
Used to digitize Flach & Heip 1996 Fig. 5e/f by hand: transitions are
printed as (y, mean grey of the next run, run length); the 0 % and 100 %
axis lines are read from a column through the y axis.
"""
import sys
from PIL import Image


def main():
    im = Image.open(sys.argv[1]).convert("L")
    w, h = im.size
    px = im.load()
    print("size", w, h)
    for xs in sys.argv[2:]:
        x = int(xs)
        # average 5 columns to smooth hatching
        col = [sum(px[min(w - 1, max(0, x + d)), y] for d in range(-2, 3)) / 5
               for y in range(h)]
        cls = ["W" if v > 200 else ("K" if v < 60 else "G") for v in col]
        out, start = [], 0
        for y in range(1, h + 1):
            if y == h or cls[y] != cls[start]:
                if y - start >= 3:
                    out.append(f"{cls[start]}[{start}-{y - 1}]")
                start = y
        print(x, " ".join(out))


if __name__ == "__main__":
    main()
