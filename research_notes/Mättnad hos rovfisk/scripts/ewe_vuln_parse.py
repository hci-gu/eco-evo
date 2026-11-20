"""Parse Mackinson & Daskalov 2007 Table 4.5 (fitted 1991 vulnerability matrix; rows = prey, columns =
predators 1..34) from `pdftotext -layout` output. The header compresses to 3-char spacing after column 9,
but data rows keep 4-char spacing, so a data value at character position p belongs to predator column
k = round((p - x1)/4) + 1, x1 = position of column 1 on that page (derived from the header's '5' minus 16).
Usage: python3 -I ewe_vuln_parse.py <tech142.txt>"""
import sys, re
L = open(sys.argv[1], encoding="utf-8", errors="replace").read().split("\n")
names = {13: "JuvCod", 14: "Cod", 15: "JuvWhiting", 16: "Whiting", 17: "JuvHaddock", 18: "Haddock",
         19: "JuvSaithe", 20: "Saithe", 21: "Hake"}
starts = [i for i, l in enumerate(L) if l.lstrip("\f").startswith("Table 4.5")][:1]  # page 1 only: alignment verified by hand; later pages ambiguous
for s in starts:
    hj = next((j for j in range(s, s + 6) if L[j].startswith("ee")), None)
    if hj is None: continue
    toks = [(m.start(), m.group()) for m in re.finditer(r"\S+", L[hj])]
    x5 = next(p for p, t in toks if t == "5")
    x1 = x5 - 16
    print(f"== page at line {s+1}: column 1 at char {x1}")
    for l in L[hj + 1: hj + 60]:
        if l.startswith("Table"): break
        m = re.match(r"^(\S.*?)\s{2,}(\d+)(\s.*)?$", l)
        if not m: continue
        name, rid = m.group(1).strip(), int(m.group(2))
        if rid > 70: continue
        vals = {}
        for mm in re.finditer(r"\d+(?:\.\d+)?", l[m.end(2):]):
            p = m.end(2) + mm.start()
            k = round((p - x1) / 4) + 1
            vals[k] = mm.group()
        sel = {names[k]: vals[k] for k in names if k in vals}
        if sel: print(f"{rid:3d} {name[:30]:30s} {sel}")
