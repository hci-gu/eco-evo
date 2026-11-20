"""Strip tags from a (JATS) XML/HTML file and print plain text. Usage: python3 -I xmltext.py <file>"""
import re, sys, html
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", t, flags=re.S)
t = re.sub(r"</(p|title|sec|caption|tr|li|h\d)>", "\n", t)
t = re.sub(r"<[^>]+>", " ", t)
t = html.unescape(t)
t = re.sub(r"[ \t]+", " ", t)
print(re.sub(r"\n\s*\n+", "\n", t))
