"""Page number of a phrase in pdftotext output (pages split by form feeds).
Usage: q1_pages.py <txt> <first_page> <phrase> [<phrase> ...]"""
import sys, re
txt = open(sys.argv[1], encoding='utf8', errors='replace').read()
pages = txt.split('\f')
first = int(sys.argv[2])
norm = lambda s: re.sub(r'\s+', ' ', s)
for ph in sys.argv[3:]:
    hits = [i + first for i, p in enumerate(pages) if norm(ph) in norm(p)]
    print(repr(ph[:50]), '->', hits)
