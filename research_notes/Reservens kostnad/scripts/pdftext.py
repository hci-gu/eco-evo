"""Extract text from a PDF (path argv[1]) to argv[2] using pdftotext; print page count."""
import subprocess, sys
subprocess.run(["pdftotext", "-layout", sys.argv[1], sys.argv[2]], check=True)
t = open(sys.argv[2], encoding="utf-8", errors="replace").read()
print(len(t), "chars,", t.count("\f"), "pages")
