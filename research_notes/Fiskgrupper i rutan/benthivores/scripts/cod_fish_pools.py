"""Cod fish prey mapped to model pools, with unidentified gadoids (Gu) allocated by a stated rule.
Usage: python3 -I cod_fish_pools.py <ices_fish_prey_items.csv> <ices_benth_diet_pools.csv> <out_csv>
Rule (judgement): Gu is split over Norway pout (P), whiting (W) and haddock+cod (Bf) in proportion to the
identified gadoid prey mass of the same area, period and cod size class; if none is identified there,
in proportion to the area's all-size identified gadoids. Fu (fish not identified) is kept apart.
Shares are % of the FISH part, plus the fish part as % of the whole identified diet (from the pools table).
"""
import csv, sys, collections
items, pools, out = sys.argv[1:4]
def cls(L):
    L = float(L)
    return "<40" if L < 40 else "40-50" if L < 50 else "50-70" if L < 70 else ">=70"
M = collections.defaultdict(lambda: collections.defaultdict(float))
N = collections.defaultdict(int)
for r in csv.DictReader(open(items)):
    if r["predator"] != "cod" or not r["pred_len_cm"] or int(r["year"]) < 2000:
        continue
    for a in [r["area"]] + (["Box"] if r["box"] else []):
        if a not in ("Skagerrak", "Kattegat", "NNS", "Box"):
            continue
        for s in (cls(r["pred_len_cm"]), "all"):
            M[(a, s)][r["sub"]] += float(r["g"]); N[(a, s)] += 1
fish_share = {}
for r in csv.DictReader(open(pools)):
    if r["species"] == "cod" and r["period"] == "2000+" and r["size_cm"] == "all":
        fish_share[r["area"]] = round(sum(float(r[f"pct_{p}"]) for p in ("P", "W", "Bf", "Sand", "Ofi", "Gu", "Fu")), 1)
P = {"herring": "P", "sprat": "P", "clupeid_unid": "P", "norway_pout": "P", "blue_whiting": "P", "other_P": "P",
     "whiting": "W", "other_W": "W", "haddock": "Bf", "cod": "Bf", "flatfish": "Bf", "other_Bf": "Bf",
     "sandeel": "Sand", "goby": "Ofi", "dragonet": "Ofi", "rockling_poorcod": "Ofi", "other_fish": "Ofi",
     "gadoid_unid": "Gu", "fish_unid": "Fu"}
rows = []
for (a, s), d in sorted(M.items()):
    t = sum(d.values())
    g_np, g_w, g_bf = d["norway_pout"], d["whiting"] + d["other_W"] * 0, d["haddock"] + d["cod"]
    gs = g_np + g_w + g_bf
    if gs <= 0:
        dd = M[(a, "all")]; g_np, g_w, g_bf = dd["norway_pout"], dd["whiting"], dd["haddock"] + dd["cod"]; gs = g_np + g_w + g_bf
    pool = collections.defaultdict(float)
    for k, v in d.items():
        pool[P[k]] += v
    gu = pool["Gu"]
    alloc = {"P": gu * g_np / gs, "W": gu * g_w / gs, "Bf": gu * g_bf / gs} if gs > 0 else {}
    rows.append([a, s, N[(a, s)], round(t, 1), fish_share.get(a, "") if s == "all" else ""] +
                [round(100 * pool[p] / t, 1) for p in ("P", "W", "Bf", "Sand", "Ofi", "Gu", "Fu")] +
                [round(100 * (pool[p] + alloc.get(p, 0)) / t, 1) for p in ("P", "W", "Bf")] +
                [round(100 * d[k] / t, 1) for k in ("norway_pout", "herring", "sprat", "blue_whiting", "whiting", "haddock", "cod", "flatfish", "sandeel", "goby")])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["area", "cod_cm", "fish_items", "fish_g", "fish_pct_of_diet", "P", "W", "Bf", "Sand", "Ofi", "Gu", "Fu",
                "P_withGu", "W_withGu", "Bf_withGu", "norway_pout", "herring", "sprat", "blue_whiting", "whiting",
                "haddock", "cod", "flatfish", "sandeel", "goby"])
    w.writerows(rows)
print(open(out).read())
