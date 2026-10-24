"""ICES official nominal catches (2006-2024) of given species in 27.3.a, 27.3.a.20, 27.3.a.21, 27.3.a_NK by country.
Usage: python3 -I ices_nominal_3a.py OfficialNominalCatches.zip NOP WHB SPR"""
import csv, io, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
f = io.TextIOWrapper(z.open("OfficialNominalCatches/ICESCatchDataset2006-2024.csv"), encoding="utf-8", errors="replace")
r = csv.reader(f); h = next(r)
yrs = [str(y) for y in range(2006, 2025)]
for sp in sys.argv[2:]:
    print("==", sp, "tonnes live weight; areas 27.3.a.20 (Skagerrak), 27.3.a.21, 27.3.a_NK, 27.3.a total of those; 27.4")
    tot = {}
    for x in r if False else []:
        pass
    f.seek(0); r = csv.reader(f); next(r)
    for x in r:
        if x[0] != sp or x[3] != "TLW":
            continue
        a = x[1]
        if a in ("27.3.a.20", "27.3.a.21", "27.3.a_NK", "27.3.a", "27.4"):
            for y in yrs:
                v = x[h.index(y)]
                try:
                    v = float(v)
                except ValueError:
                    v = 0.0
                tot[(a, y)] = tot.get((a, y), 0.0) + v
                tot[(a + ":" + x[2], y)] = tot.get((a + ":" + x[2], y), 0.0) + v
    keys = sorted({k for k, _ in tot})
    print("area/country".ljust(22) + " ".join(y[2:].rjust(7) for y in yrs))
    for k in keys:
        vals = [tot.get((k, y), 0.0) for y in yrs]
        if sum(vals) > 0:
            print(k.ljust(22) + " ".join(f"{v:7.0f}" for v in vals))
