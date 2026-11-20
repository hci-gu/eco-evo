"""Print lines of a text file (argv[1]) matching any regex in argv[2:], with page number and context."""
import re, sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
pats = [re.compile(p, re.I) for p in sys.argv[2:]]
for pi, page in enumerate(t.split("\f"), 1):
    lines = page.split("\n")
    for i, l in enumerate(lines):
        if any(p.search(l) for p in pats):
            ctx = " / ".join(x.strip() for x in lines[max(0, i-1):i+2])
            print(f"p{pi}: {ctx[:400]}")
