"""By-NUMBER prey composition of saithe (and whiting) stomachs without prey weights in the ICES
public download (1981, 1991 records carry Count but no Weight). Uses the pool classifier of
predator_diet_pools.py (imported by path, same folder).

Usage: python3 -I saithe_1981_counts.py <ices_stomach_dir> <worms_dir> [<worms_dir2>]
"""
import csv, os, sys, collections, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1]
sys.argv = [sys.argv[0], root, os.devnull] + sys.argv[2:]
spec = importlib.util.spec_from_file_location("pdp_classify", os.path.join(here, "predator_diet_pools.py"))
# only the classifier is needed; load the source up to the main loop
src = open(os.path.join(here, "predator_diet_pools.py")).read().split("agg = collections.defaultdict")[0]
ns = {"__name__": "pdp"}
exec(compile(src, "predator_diet_pools.py", "exec"), ns)
classify, area_of, fnum = ns["classify"], ns["area_of"], ns["fnum"]
PRED = {"126441": "saithe", "126438": "whiting"}
for yr in ("x1981", "x1991"):
    p = os.path.join(root, yr)
    hauls = {r["tblHaulID"]: r for r in csv.DictReader(open(os.path.join(p, "HaulInformation.csv"), encoding="utf-8", errors="replace"))}
    preds = {}
    for r in csv.DictReader(open(os.path.join(p, "PredatorInformation.csv"), encoding="utf-8", errors="replace")):
        sp = PRED.get(r["AphiaIDPredator"].strip('"'))
        if sp:
            preds[r["tblPredatorInformationID"]] = (sp, r)
    cnt = collections.defaultdict(collections.Counter); taxa = collections.defaultdict(collections.Counter)
    nw = collections.Counter(); stom = collections.defaultdict(set); months = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(os.path.join(p, "PreyInformation.csv"), encoding="utf-8", errors="replace")):
        pid = r["tblPredatorInformationID"]
        if pid not in preds:
            continue
        sp, pr = preds[pid]
        h = hauls.get(pr["tblHaulID"], {})
        ar = area_of(fnum(h.get("ShootLat")), fnum(h.get("ShootLong")))
        n = fnum(r["Count"])
        if n != n or n <= 0 or n >= 9999:
            continue
        pool, leaf = classify(r["AphiaIDPrey"].strip('"'))
        if pool == "drop":
            continue
        L = fnum(pr["Length"]); L = L / 10 if L == L and L > 200 else L
        k = (sp, ar, "lt40" if L == L and L < 40 else "ge40")
        cnt[k][pool] += n; taxa[k][(pool, leaf)] += n; stom[k].add(pid); months[k][pr["Month"]] += 1
        if fnum(r["Weight"]) == fnum(r["Weight"]):
            nw[k] += 1
    for k in sorted(cnt):
        if k[1] not in ("NNS", "Skagerrak"):
            continue
        tot = sum(cnt[k].values())
        fish = sum(cnt[k][q] for q in ("P", "Bf", "W", "Gu", "Fu"))
        print(yr[1:], *k, "records", len(stom[k]), "items", int(tot), "rows_with_weight", nw[k], "months", dict(months[k]))
        print("   by number, all items: " + " ".join(f"{q} {100*cnt[k][q]/tot:.1f}" for q in ("P", "Bf", "W", "Gu", "Fu", "Z", "B", "O", "U", "Uc")))
        if fish:
            print("   fish items only (%): " + " ".join(f"{q} {100*cnt[k][q]/fish:.1f}" for q in ("P", "Bf", "W", "Gu", "Fu")))
            top = sorted(((n, t) for t, n in taxa[k].items() if t[0] in ("P", "Bf", "W", "Gu")), reverse=True)[:8]
            print("   top fish taxa (n): " + "; ".join(f"{t[1]} {int(n)}" for n, t in top))
