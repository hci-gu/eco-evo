"""Grep an already-extracted text file. Usage: python3 -I dc_grep.py <txt> <maxhits> <width> regex..."""
import re, sys
txt = re.sub(r"\s+", " ", open(sys.argv[1], encoding="utf-8", errors="replace").read())
n, w = int(sys.argv[2]), int(sys.argv[3])
for pat in sys.argv[4:]:
    for i, m in enumerate(re.finditer(pat, txt, flags=re.I)):
        if i >= n: break
        print("----", pat, ":", txt[max(0, m.start() - w):m.end() + w])
