"""Cod (and other benthivores) fish prey by prey species and predator size, plus prey/predator length ratios.
Usage: python3 -I cod_fish_prey.py <ices_fish_prey_items.csv> <out_csv> <ratio_csv>
Reads the table written by benth_diet_pools.py (derived data, not a download).
"""
import csv, sys, collections, statistics
src, out, rout = sys.argv[1:4]
rows = list(csv.DictReader(open(src)))
def cls(L):
    L = float(L)
    for a, b in [(0, 30), (30, 40), (40, 50), (50, 70), (70, 999)]:
        if a <= L < b:
            return f"{a}-{b if b < 999 else ''}"
mass = collections.defaultdict(lambda: collections.defaultdict(float))
n = collections.defaultdict(lambda: collections.defaultdict(int))
for r in rows:
    if r["predator"] not in ("cod", "haddock", "long_rough_dab", "starry_ray") or not r["pred_len_cm"]:
        continue
    per = "pre2000" if int(r["year"]) < 2000 else "2000+"
    areas = [r["area"]] + (["Box"] if r["box"] else [])
    for a in areas:
        for s in (cls(r["pred_len_cm"]), "all"):
            k = (r["predator"], a, per, s)
            mass[k][r["sub"]] += float(r["g"]); n[k][r["sub"]] += 1
            mass[k]["_tot"] += float(r["g"])
subs = ["herring", "sprat", "clupeid_unid", "norway_pout", "blue_whiting", "other_P", "sandeel", "whiting",
        "other_W", "haddock", "cod", "flatfish", "other_Bf", "goby", "dragonet", "rockling_poorcod",
        "other_fish", "gadoid_unid", "fish_unid"]
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["predator", "area", "period", "pred_cm", "fish_g"] + [f"pct_{s}" for s in subs] + [f"n_{s}" for s in subs])
    for k in sorted(mass):
        t = mass[k]["_tot"]
        w.writerow(list(k) + [round(t, 1)] + [round(100 * mass[k][s] / t, 1) for s in subs] + [n[k][s] for s in subs])
# prey/predator length ratios (all areas, all years: a morphological measure)
rat = collections.defaultdict(list)
for r in rows:
    if r["prey_len_cm"] and r["pred_len_cm"]:
        pl, L = float(r["prey_len_cm"]), float(r["pred_len_cm"])
        if pl > 0 and L > 0 and pl / L < 1.5:
            rat[(r["predator"], r["sub"])].append((pl / L, pl, L))
            rat[(r["predator"], "ALL_FISH")].append((pl / L, pl, L))
with open(rout, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["predator", "prey", "n", "ratio_median", "ratio_p90", "ratio_p99", "ratio_max", "prey_cm_median", "prey_cm_max", "pred_cm_median"])
    for k in sorted(rat):
        v = sorted(x[0] for x in rat[k]); pl = sorted(x[1] for x in rat[k]); L = [x[2] for x in rat[k]]
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        w.writerow(list(k) + [len(v), round(statistics.median(v), 3), round(q(0.9), 3), round(q(0.99), 3), round(v[-1], 3),
                              round(statistics.median(pl), 1), round(pl[-1], 1), round(statistics.median(L), 1)])
print("ok")
