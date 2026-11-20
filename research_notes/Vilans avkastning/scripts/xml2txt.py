"""Strip tags from an XML/HTML file and print text. Usage: python3 -I xml2txt.py <file> [out]"""
import re, sys, html
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
t = re.sub(r"(?i)</(p|title|sec|div|tr|li|h\d|caption|label)>", "\n", t)
t = re.sub(r"<[^>]+>", " ", t)
t = html.unescape(t)
t = re.sub(r"[ \t]+", " ", t)
t = re.sub(r"\n\s*\n+", "\n", t)
out = sys.argv[2] if len(sys.argv) > 2 else None
if out: open(out, "w").write(t)
else: print(t)
