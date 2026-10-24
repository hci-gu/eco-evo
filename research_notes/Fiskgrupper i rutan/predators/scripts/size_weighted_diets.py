"""Size-weighted diet by pool for whiting and hake (box length mix), with unidentified fish
re-allocated: Gu (gadoids not to species) pro-rata over identified gadoids of the same
species/area/period (P: Norway pout, blue whiting, silvery pout; Bf: cod, haddock, poor cod, rocklings;
W: whiting, saithe, pollack, hake, ling); Fu pro-rata over all identified fish (P + Bf + W).

Usage: python3 -I size_weighted_diets.py <tables_dir> <length_class_pct.csv>
Reads <tables_dir>/ices_diet_pools.csv and ices_diet_pools_taxa.csv.
"""
import csv, os, sys, collections
tdir, lcp = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(tdir, "ices_diet_pools.csv"))))
taxa = list(csv.DictReader(open(os.path.join(tdir, "ices_diet_pools_taxa.csv"))))
GAD_P = ("Norway pout", "Micromesistius poutassou", "Gadiculus argenteus")
GAD_BF = ("Gadus morhua", "Melanogrammus aeglefinus", "Trisopterus minutus", "Trisopterus luscus", "Enchelyopus", "Ciliata", "Gaidropsarus")
GAD_W = ("Merlangius merlangus", "Pollachius virens", "Pollachius pollachius", "Merluccius merluccius", "Molva molva")
def gad_split(sp, ar, per):
    g = collections.Counter()
    for t in taxa:
        if (t["species"], t["area"], t["period"]) != (sp, ar, per):
            continue
        x = t["taxon"]; w = float(t["g"])
        if x in GAD_P or x.startswith("Micromesistius"): g["P"] += w
        elif x in GAD_BF or x.startswith("Gadus") or x.startswith("Melanogrammus"): g["Bf"] += w
        elif x in GAD_W or x.startswith("Merlangius") or x.startswith("Pollachius"): g["W"] += w
    s = sum(g.values())
    return {k: g[k] / s for k in ("P", "Bf", "W")} if s else {"P": 1 / 3, "Bf": 1 / 3, "W": 1 / 3}
LC = {}
for r in csv.DictReader(open(lcp)):
    if r["variant"] == "A_floor" and r["quarter"] == "pooled":
        LC[r["species"]] = {k: float(v) for k, v in r.items() if k.startswith("[")}
def wclass(sp):
    c = LC[sp]
    if sp == "whiting":
        return {"10": c["[0, 10)"] + c["[10, 15)"] + c["[15, 20)"], "20": c["[20, 25)"] + c["[25, 30)"],
                "30": c["[30, 40)"], "40": c["[40, 50)"] + c["[50, 60)"] + c["[60, 70)"] + c["[70, 200)"]}
    return {"10": c["[0, 10)"] + c["[10, 15)"] + c["[15, 20)"], "20": c["[20, 25)"] + c["[25, 30)"],
            "30": c["[30, 40)"], "40": c["[40, 50)"], "50": c["[50, 60)"], "60": c["[60, 70)"], "70": c["[70, 200)"]}
OUT = ["P", "Bf", "W", "Z", "Zpas", "Zeuph", "B", "O"]
def realloc(r, gs):
    v = {q: float(r["pct_" + q]) for q in ("P", "Bf", "W", "Gu", "Fu", "Z", "B", "O", "Zpas", "Zeuph")}
    for q in ("P", "Bf", "W"):
        v[q] += v["Gu"] * gs[q]
    fish = v["P"] + v["Bf"] + v["W"]
    if fish > 0:
        for q in ("P", "Bf", "W"):
            v[q] += v["Fu"] * v[q] / fish
    else:
        v["P"] += v["Fu"]  # no identified fish: unidentified fish put to P (judgement)
    return v
for sp, ar, per in [("whiting", "Skagerrak", "pre2000"), ("whiting", "Box", "pre2000"), ("whiting", "Skagerrak", "2022"),
                    ("whiting", "Kattegat", "pre2000"), ("whiting", "CSNS", "2022"),
                    ("hake", "Skagerrak", "2024"), ("hake", "Box", "2024")]:
    gs = gad_split(sp, ar, per)
    cls = {r["size_cm"]: r for r in rows if (r["species"], r["area"], r["period"]) == (sp, ar, per)}
    W = wclass(sp)
    # mass per class (identified), merge >= top class
    def mass(r):
        tot = float(r["g_total"]) / (1 + (float(r["pct_U"]) + float(r["pct_Uc"])) / 100)
        return tot
    res = collections.Counter(); wsum = 0; used = []
    for c, w in W.items():
        keys = [k for k in cls if k != "all" and k != "na" and (k == c or (c == max(W) and int(k) > int(c)) or (c == "10" and k == "00"))]
        if not keys:
            # take nearest available class
            avail = sorted((abs(int(k) - int(c)), k) for k in cls if k not in ("all", "na"))
            keys = [avail[0][1]]
        tm = sum(mass(cls[k]) for k in keys)
        agg = collections.Counter()
        for k in keys:
            v = realloc(cls[k], gs)
            for q in OUT:
                agg[q] += v[q] * mass(cls[k]) / tm
        for q in OUT:
            res[q] += w * agg[q]
        wsum += w; used.append(f"{c}:{'+'.join(keys)}(w{w:.1f})")
    pooled = realloc(cls["all"], gs)
    print(f"{sp} {ar} {per}  gadoid split P/Bf/W {gs['P']:.2f}/{gs['Bf']:.2f}/{gs['W']:.2f}  fed {cls['all']['fed']}")
    print("   size-weighted: " + " ".join(f"{q} {res[q]/wsum:.1f}" for q in OUT))
    print("   pooled       : " + " ".join(f"{q} {pooled[q]:.1f}" for q in OUT))
    print("   classes: " + " ".join(used))
