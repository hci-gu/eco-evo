"""[D] Relative predation risk of poor- vs good-condition smolts from Tucker et al. 2016 Table 3
(proportions of trawl-caught = available and auklet-predated fish). RR = (pred/trawl)_poor / (pred/trawl)_good.
Only cells whose trawl share is >= 0.03 are used (0.00-0.01 cells are rounding noise)."""
T = {  # (species,size): {cond: (trawl, predated)} ; 2012 and 2013
 2012: {("SE","L"):{"+":(0.06,0.00),"-":(0.05,0.02)}, ("SE","S"):{"+":(0.07,0.11),"-":(0.07,0.42)},
        ("PK","L"):{"+":(0.09,0.00),"-":(0.09,0.01)}, ("PK","S"):{"+":(0.13,0.06),"-":(0.16,0.22)},
        ("CM","L"):{"+":(0.09,0.00),"-":(0.08,0.05)}, ("CM","S"):{"+":(0.05,0.04),"-":(0.06,0.08)}},
 2013: {("PK","S"):{"+":(0.04,0.11),"-":(0.03,0.14)},
        ("CM","L"):{"+":(0.34,0.02),"-":(0.39,0.13)}, ("CM","S"):{"+":(0.05,0.09),"-":(0.09,0.35)}},
}
for yr, cells in T.items():
    print("Year", yr)
    for k, c in cells.items():
        (tg, pg), (tp, pp) = c["+"], c["-"]
        if min(tg, tp) < 0.03: continue
        sel_g, sel_p = pg / tg, pp / tp
        rr = sel_p / sel_g if sel_g > 0 else float("inf")
        print("  %s-%s: selection good %.2f poor %.2f RR poor/good %s" % (k[0], k[1], sel_g, sel_p, "inf (0 good taken)" if sel_g == 0 else "%.1f" % rr))
    tg = sum(c["+"][0] for c in cells.values()); pg = sum(c["+"][1] for c in cells.values())
    tp = sum(c["-"][0] for c in cells.values()); pp = sum(c["-"][1] for c in cells.values())
    print("  pooled usable cells: RR poor/good = %.2f (size not controlled)" % ((pp/tp)/(pg/tg)))
    for s in ("S",):
        tg = sum(c["+"][0] for k,c in cells.items() if k[1]==s); pg = sum(c["+"][1] for k,c in cells.items() if k[1]==s)
        tp = sum(c["-"][0] for k,c in cells.items() if k[1]==s); pp = sum(c["-"][1] for k,c in cells.items() if k[1]==s)
        print("  small fish only: RR poor/good = %.2f" % ((pp/tp)/(pg/tg)))
