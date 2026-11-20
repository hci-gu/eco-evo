"""Convert a PDF to text and print up to 6 context windows per regex.
Usage: python3 -I dc_pdftext.py <pdf> <out_txt> [regex ...]"""
import re, subprocess, sys
pdf, out = sys.argv[1], sys.argv[2]
subprocess.run(["pdftotext", pdf, out], check=True)
txt = re.sub(r"\s+", " ", open(out, encoding="utf-8", errors="replace").read())
print("chars", len(txt))
for pat in sys.argv[3:]:
    for i, m in enumerate(re.finditer(pat, txt, flags=re.I)):
        if i >= 6: break
        print("----", pat, ":", txt[max(0, m.start() - 220):m.end() + 220])
