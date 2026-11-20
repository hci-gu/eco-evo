"""Krill counted per stomach (single and pooled records) for whiting, saithe, hake, Norway pout, haddock, cod
in the processed ICES stomach records (research_notes/"Krill over grunda bottnar"/stomachs/tables/records_linked.csv.gz,
o-umlaut in the real folder name). Converted to %BW with a krill wet weight of 0.15 / 0.23 / 0.30 g
(the bracket of research_notes/"Krillparen mot vintermagar"/scripts/onsrud_rations.py) and the length-weight
relations of ices_krill_content.py. Pooled records give the mean per stomach of the pool.

Usage: python3 -I ices_krill_counts.py <records_linked.csv.gz>
"""
import csv, gzip, sys
from collections import defaultdict
LW = {"whiting": (0.0058, 3.10), "saithe": (0.0089, 3.05), "norway pout": (0.0065, 3.17),
      "haddock": (0.0083, 3.028), "cod": (0.0076, 3.08), "hake": (0.0060, 3.10)}  # hake [J]
KW = (0.15, 0.23, 0.30)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


rows = defaultdict(list)
with gzip.open(sys.argv[1], "rt") as fh:
    for r in csv.DictReader(fh):
        sp = r["sp"].strip().lower()
        if sp not in LW:
            continue
        kc, N, L = f(r["krill_count"]), f(r["Number"]) or 1.0, f(r["L_cm"])
        if not kc or not L:
            continue
        fed = f(r["fed"])
        per = kc / N
        a, b = LW[sp]
        W = a * L ** b
        rows[sp].append((per, L, W, N, r["month"], r["year"], r["bed_true"] or r["bed"], r["area"]))
print("krill counted per stomach (pool mean); top records per species; %BW at krill 0.15/0.23/0.30 g")
for sp, v in rows.items():
    v.sort(key=lambda x: -x[0] * 0.23 / x[2])
    print(f"== {sp}: {len(v)} records with krill counts")
    for per, L, W, N, m, y, bed, area in v[:8]:
        pct = "/".join(f"{100 * per * k / W:.1f}" for k in KW)
        print(f"   {per:6.1f} krill/stomach  L {L:5.1f} cm  W {W:6.0f} g  N {N:4.0f}  {y}-{m}  bed {bed}  {area}:  {pct} %BW")
