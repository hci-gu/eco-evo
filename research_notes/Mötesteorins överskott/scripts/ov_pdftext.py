"""Extract text from a downloaded PDF (untrusted) to <pdf>.txt using pdftotext; print grep hits.
Usage: python3 -I scripts/ov_pdftext.py <pdf> [regex ...]"""
import re, subprocess, sys
pdf = sys.argv[1]
out = pdf[:-4] + ".txt"
subprocess.run(["pdftotext", "-layout", pdf, out], check=True)
txt = open(out, errors="replace").read()
print(out, len(txt), "chars")
for pat in sys.argv[2:]:
    for m in re.finditer(pat, txt, re.I):
        s = max(0, m.start() - 250); e = min(len(txt), m.end() + 250)
        print("---", pat, ":", re.sub(r"\s+", " ", txt[s:e]))
