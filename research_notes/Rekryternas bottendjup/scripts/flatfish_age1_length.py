"""Mean length at age 1 in Q3 (the model's stage cut: J = below the species' length at age 1 in Q3) for the
benthivore members without a cut in the library (dab, long rough dab, lemon sole), plus the four with one
(haddock, cod, plaice, witch) as the control against research_notes/"Arsungarnas andel" (A-ring) section 0.

Usage: python3 -I flatfish_age1_length.py <datras_dir>
  <datras_dir> = research_notes/"Torskfiskarnas artsammansattning" (a-umlaut)/data/datras (CA_2021-25_Q1/Q3.xml.gz,
  NS-IBTS, ICES DATRAS, CC BY 4.0). Parsed only.

Two areas: the Skagerrak rectangles 43-47 F7-G1 (as the library's cuts) and all NS-IBTS rectangles (fallback where
the Skagerrak has < 20 aged fish). CA rows are length-stratified samples, so a plain mean over CA rows is the
convention of the library's cuts (graduation_length in juvenile_share_by_quarter.py), not a population mean.
"""
import gzip
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

NS = "{ices.dk.local/DATRAS}"
SP = {126437: "haddock", 126436: "cod", 127143: "plaice", 127136: "witch", 127139: "dab",
      127137: "long rough dab", 127140: "lemon sole"}
SKAG = {f"{r}{c}" for r in range(43, 48) for c in ("F7", "F8", "F9", "G0", "G1")}
acc = defaultdict(lambda: [0.0, 0.0, 0.0])  # (area, species, quarter, age) -> sum n, sum n L, sum n L^2
for f in sorted(Path(sys.argv[1]).glob("CA_*.xml.gz")):
    with gzip.open(f) as g:
        for _, el in ET.iterparse(g):
            if el.tag != NS + "Cls_DatrasExchange_CA":
                continue
            k = {c.tag.replace(NS, ""): (c.text or "").strip() for c in el}
            el.clear()
            a = k.get("Valid_Aphia", "")
            if not a.isdigit() or int(a) not in SP:
                continue
            try:
                age = int(float(k.get("Age", "-9")))
                L = float(k["LngtClass"])
                n = float(k.get("CANoAtLngt", "1") or 1)
            except ValueError:
                continue
            if age < 0 or age > 3 or n <= 0:
                continue
            code = k.get("LngtCode", "").strip()
            L = L / 10.0 if code in (".", "0") else L
            q = int(k["Quarter"])
            for area in ("skag", "all"):
                if area == "skag" and k.get("AreaCode", "").strip() not in SKAG:
                    continue
                s = acc[(area, SP[int(a)], q, age)]
                s[0] += n
                s[1] += n * L
                s[2] += n * L * L
print("mean length (cm) at age, CA 2021-25 (n aged fish); the cut is age 1 in Q3")
print(f"{'species':>16} {'area':>5} {'Q3 age0':>14} {'Q3 age1':>14} {'Q1 age1':>14} {'Q3 age2':>14}")
for sp in SP.values():
    for area in ("skag", "all"):
        cells = []
        for q, age in ((3, 0), (3, 1), (1, 1), (3, 2)):
            n, s1, s2 = acc[(area, sp, q, age)]
            cells.append(f"{s1 / n:6.1f} ({int(n):5d})" if n > 0 else f"{'-':>14}")
        print(f"{sp:>16} {area:>5} " + " ".join(f"{c:>14}" for c in cells))
