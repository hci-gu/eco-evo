"""Print sentences of a (text/xml/html) file matching a regex (tags stripped).
Usage: python3 -I find_sent.py <file> <regex> [max]"""
import re, sys, html
t = open(sys.argv[1], encoding='utf8', errors='replace').read()
t = html.unescape(re.sub(r'<[^>]+>', ' ', t))
t = re.sub(r'\s+', ' ', t)
pat = re.compile(sys.argv[2], re.I)
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10
n = 0
for s in re.split(r'(?<=[.!?])\s+', t):
    if pat.search(s):
        print('-', s[:600]); n += 1
        if n >= mx: break
