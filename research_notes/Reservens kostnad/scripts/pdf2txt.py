"""Extract text from a downloaded PDF (untrusted) with pdftotext; run with python3 -I.
Usage: python3 -I pdf2txt.py <in.pdf> <out.txt>"""
import subprocess, sys
src, dst = sys.argv[1], sys.argv[2]
subprocess.run(["/usr/bin/pdftotext", "-layout", src, dst], check=True)
print(open(dst, encoding="utf-8", errors="replace").read()[:300])
