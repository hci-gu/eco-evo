"""Biomass-weight the ICES cod size-class diets (2000+) by the box cod length mix.
Usage: python3 -I cod_weighted.py <ices_benth_diet_pools.csv> <length_class_pct.csv> <out_csv>
Length mix: variant A_floor, quarter pooled (Torskfiskarnas bytesandelar/tables/length_class_pct.csv).
Classes mapped: <20 = [0,10)+[10,15)+[15,20); 20-30; 30-40; 40-50; 50-70; >=70. A class with fewer than
5 fed stomachs borrows the neighbouring larger class (stated rule).
"""
import csv, sys
pools, lcp, out = sys.argv[1:4]
mix = None
for r in csv.DictReader(open(lcp)):
    if r["variant"] == "A_floor" and r["quarter"] == "pooled" and r["species"] == "cod":
        g = lambda k: float(r[k])
        mix = {"00-20": g("[0, 10)") + g("[10, 15)") + g("[15, 20)"), "20-30": g("[20, 25)") + g("[25, 30)"),
               "30-40": g("[30, 40)"), "40-50": g("[40, 50)"), "50-70": g("[50, 60)") + g("[60, 70)"), "70-": g("[70, 200)")}
P = ["B", "Z", "Pas", "P", "W", "Bf", "Sand", "Ofi", "Gu", "Fu", "Ceph", "O"]
D = {}
for r in csv.DictReader(open(pools)):
    if r["species"] == "cod" and r["period"] == "2000+":
        D[(r["area"], r["size_cm"])] = r
order = ["00-20", "20-30", "30-40", "40-50", "50-70", "70-"]
rows = []
for area in ("Skagerrak", "NNS", "Kattegat"):
    acc = {p: 0.0 for p in P}; used = []
    for i, s in enumerate(order):
        j = i; r = D.get((area, s))
        while (r is None or int(r["fed"]) < 5) and j < len(order) - 1:
            j += 1; r = D.get((area, order[j]))
        jj = i
        while (r is None or int(r["fed"]) < 5) and jj > 0:
            jj -= 1; r = D.get((area, order[jj]))
        used.append(f"{s}<-{r['size_cm']}({r['fed']})")
        tot = sum(float(r[f"pct_{p}"]) for p in P)
        for p in P:
            acc[p] += mix[s] * float(r[f"pct_{p}"]) / tot
    w = sum(mix.values())
    rows.append([area] + [round(100 * acc[p] / w, 1) for p in P] + [" ".join(used)])
with open(out, "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["area"] + P + ["classes_used(fed)"]); wr.writerows(rows)
print("mix", {k: round(v, 1) for k, v in mix.items()})
for r in rows: print(r)
