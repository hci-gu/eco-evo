"""Empty-stomach percentages from ICES CRR 219 (1991 stomach sampling project), Table 7A-E.

Run: python3 -I scripts/A_crr219_empty.py
Input: data/A_ices_crr219/crr219.txt (pdftotext -layout of the figshare PDF).
The column alignment of the size classes is lost in the text layer, so only the row values
(per quarter: 'Total' over areas and 'Area 8' = Skagerrak roundfish area) are summarised:
min, median, max over the size classes present, plus the stomach-weighted value cannot be
formed (n per cell not in this table). Values are percentages.
"""
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "data" / "A_ices_crr219" / "crr219.txt").read_text(errors="ignore").splitlines()
TABLES = {"7-A": "cod", "7-B": "haddock", "7-0": "whiting", "7-C": "whiting", "7-D": "mackerel", "7-E": "saithe"}
num = re.compile(r"(?<![\d.])(\d{1,3}\.\d{2})(?![\d.])")

start = {}
for i, line in enumerate(TXT):
    m = re.search(r"TABLE\s*(7-[A-E0])\s*:\s*Percentage of empty", line)
    if m and TABLES[m.group(1)] not in start:
        start[TABLES[m.group(1)]] = i
order = sorted(start.items(), key=lambda kv: kv[1])
for k, (sp, i0) in enumerate(order):
    i1 = order[k + 1][1] if k + 1 < len(order) else i0 + 200
    q = None
    print(f"== {sp} (lines {i0}-{i1}); header: {TXT[i0].strip()[:110]}")
    for line in TXT[i0:i1]:
        mq = re.search(r"Quarter (\d)", line)
        if mq:
            q = mq.group(1)
        if "Total Year" in line:
            q = "year"
        s = line.strip()
        lab = None
        if re.match(r"^\W{0,3}\s*Total\b", s) and "Year" not in s:
            lab = "Total"
        elif re.match(r"^\W{0,3}\s*Area 8\b", s):
            lab = "Area 8"
        if lab:
            vals = [float(v) for v in num.findall(line.split(lab, 1)[1])]
            vals = [v for v in vals if v <= 100]
            if vals:
                print(f"  Q{q:4s} {lab:7s} n_classes {len(vals):2d} min {min(vals):6.2f} median {st.median(vals):6.2f} "
                      f"max {max(vals):6.2f}  values {vals}")
