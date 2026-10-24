"""R/SSB by period and catch/total biomass from SAG StockDownload JSON files.

Usage: python3 -I sag_metrics.py <stockdl_*.json> ...
R is shifted by the recruitment age (R(y+ra) / SSB(y)). Medians per period. Catch = Catches,
else Landings + Discards, else Landings + Custom1 ('dead discards') where named so.
"""
import json, sys, statistics as st

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

for fn in sys.argv[1:]:
    rows = json.load(open(fn))
    lab = rows[0]["StockKeyLabel"]
    ra = int(num(rows[0].get("RecruitmentAge")) or 0)
    by = {int(r["Year"]): r for r in rows}
    print(f"== {lab} (assessment {rows[0]['AssessmentYear']}, key {rows[0]['AssessmentKey']}), recruitment age {ra}, SSB units {rows[0]['StockSizeUnits']}")
    yrs = sorted(by)
    rs = {}
    for y in yrs:
        s = num(by[y].get("StockSize"))
        r = num(by.get(y + ra, {}).get("Recruitment"))
        if s and r:
            rs[y] = r / s
    if rs:
        y0 = min(rs)
        cuts = [(y0, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2015, 2024), (2020, 2025)]
        for a, b in cuts:
            v = [rs[y] for y in rs if a <= y <= b]
            R = [num(by[y + ra]["Recruitment"]) for y in rs if a <= y <= b]
            S = [num(by[y]["StockSize"]) for y in rs if a <= y <= b]
            if v:
                print(f"  SSB years {a}-{b}: n {len(v):2d}  median R/SSB {st.median(v):8.3f}  median R {st.median(R):12.0f}  median SSB {st.median(S):10.0f}")
    out = []
    for y in yrs:
        r = by[y]
        tb = num(r.get("TBiomass")); ssb = num(r.get("StockSize"))
        c = num(r.get("Catches"))
        L, D = num(r.get("Landings")), num(r.get("Discards"))
        dd = num(r.get("Custom1")) if (r.get("CustomName1") or "").startswith("dead") else None
        if c is None and L is not None:
            c = L + (D or 0) + (dd or 0)
        F = num(r.get("FishingPressure"))
        if y >= 2015 and c:
            out.append((y, c, L, D, dd, tb, ssb, F))
    for y, c, L, D, dd, tb, ssb, F in out:
        s = f"  {y}: catch {c:9.0f} (L {L}, D {D}, dead D {dd})"
        if tb:
            s += f"  TB {tb:10.0f}  C/TB {c/tb:.3f}"
        if ssb and ssb > 100:
            s += f"  SSB {ssb:10.0f}  C/SSB {c/ssb:.3f}"
        s += f"  F/HR {F}"
        print(s)
