"""Norway pout (SMS species 24) in the North Sea SMS key run 2023: biomass-weighted M1, M2 by predator, F, TSB, SP/TSB.
Usage: python3 -I sms_pout.py <sms2023 dir> [species_no=24] [species_name=Nor.pout]
Rates are consumption (or loss) per year / annual mean biomass (mean of the quarterly mean biomasses N.bar*west)."""
import csv, os, sys
from collections import defaultdict
d = sys.argv[1]
sp = int(sys.argv[2]) if len(sys.argv) > 2 else 24
spname = sys.argv[3] if len(sys.argv) > 3 else "Nor.pout"
# summary.out: Year Quarter Species.n Age M1 M2 M F Z N N.bar C.hat C.obs west weca Yield ... BIO SSB Lsea ration
hdr = None
bq = defaultdict(float)   # (year, q) -> mean biomass in quarter (N.bar * west)
m1loss = defaultdict(float); m2loss = defaultdict(float); floss = defaultdict(float); yld = defaultdict(float)
byage = defaultdict(lambda: defaultdict(float))
for line in open(os.path.join(d, "summary.out")):
    p = line.split()
    if hdr is None:
        hdr = p; continue
    r = dict(zip(hdr, p))
    if int(r["Species.n"]) != sp:
        continue
    y, q, a = int(r["Year"]), int(r["Quarter"]), int(r["Age"])
    nbar, w = float(r["N.bar"]), float(r["west"])
    b = nbar * w
    bq[(y, q)] += b
    m1loss[y] += float(r["M1"]) * b
    m2loss[y] += float(r["M2"]) * b
    floss[y] += float(r["F"]) * b
    yld[y] += float(r["Yield"])
    byage[(y, a)]["M1"] += float(r["M1"]); byage[(y, a)]["M2"] += float(r["M2"]); byage[(y, a)]["F"] += float(r["F"])
years = sorted({y for y, _ in bq})
Bann = {y: sum(bq[(y, q)] for q in range(1, 5)) / 4 for y in years}
eaten = defaultdict(lambda: defaultdict(float))
for r in csv.DictReader(open(os.path.join(d, "who_eats_whom_level2.csv"))):
    if r["Prey"].strip() == spname.replace("_", "").strip() or r["Prey"].strip().startswith(spname.split("_")[0]):
        eaten[int(r["Year"])][r["Predator"].strip()] += float(r["eatenW"])
tsb = {}
for i, line in enumerate(open(os.path.join(d, "summary_table_raw.out"))):
    p = line.split()
    if i == 0:
        th = p; continue
    r = dict(zip(th, p))
    if int(r["Species.n"]) == sp:
        f = lambda k: float(r[k]) if r[k] != "NA" else float("nan")
        tsb[int(r["Year"])] = (f("TSB"), f("SSB"), f("Yield"), f("Eaten"), f("mean.F"))
print("year meanB TSB(Q1) yield eaten(table) M1w M2w Fw  catch/meanB  SP/TSB")
for y in years:
    B = Bann[y]
    t = tsb.get(y)
    sp_ = (tsb[y + 1][0] - t[0] + t[2]) / t[0] if t and (y + 1) in tsb else float("nan")
    print(y, f"{B:9.0f} {t[0]:9.0f} {t[2]:8.0f} {t[3]:9.0f}  {m1loss[y]/B:.3f} {m2loss[y]/B:.3f} {floss[y]/B:.3f}  {yld[y]/B:.3f}  {sp_:.3f}")
def per(p0, p1):
    ys = [y for y in years if p0 <= y <= p1]
    n = len(ys)
    out = {"M1w": sum(m1loss[y] / Bann[y] for y in ys) / n, "M2w": sum(m2loss[y] / Bann[y] for y in ys) / n,
           "Fw": sum(floss[y] / Bann[y] for y in ys) / n, "C/B": sum(yld[y] / Bann[y] for y in ys) / n,
           "meanB": sum(Bann[y] for y in ys) / n}
    preds = defaultdict(float)
    for y in ys:
        for k, v in eaten[y].items():
            preds[k] += v / Bann[y] / n
    return out, preds, ys
for p in [(1984, 2000), (2001, 2010), (2011, 2022), (2016, 2022)]:
    out, preds, ys = per(*p)
    print(f"\nperiod {p}: n={len(ys)} " + " ".join(f"{k} {v:.3f}" for k, v in out.items()))
    tot = sum(preds.values())
    print(f"  M2 from eaten/meanB total {tot:.3f}")
    for k, v in sorted(preds.items(), key=lambda x: -x[1]):
        if v > 0.002:
            print(f"   {k:14s} {v:.3f}  ({v/tot*100:.1f} %)")
print("\nannual M1, M2, F by age (sum of quarters), mean 2011-2022:")
for a in range(0, 4):
    ys = [y for y in range(2011, 2023) if (y, a) in byage]
    if ys:
        print(" age", a, " ".join(f"{k} {sum(byage[(y,a)][k] for y in ys)/len(ys):.3f}" for k in ("M1", "M2", "F")))
