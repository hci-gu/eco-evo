"""pdftotext a PDF (passed as path) to a text file in the scratch out path and grep context.
Usage: python3 -I st_pdfgrep.py <pdf> <out_txt> <regex> [context_lines]"""
import re, subprocess, sys
pdf, out, rx = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 2
subprocess.run(["pdftotext", "-layout", pdf, out], check=True)
lines = open(out, errors="replace").read().splitlines()
pat = re.compile(rx, re.I)
for i, l in enumerate(lines):
    if pat.search(l):
        print("---", i)
        print("\n".join(lines[max(0, i - ctx): i + ctx + 1]))
