"""SMS key run 2023: saithe's modelled consumption of haddock (and whiting, Norway pout,
herring, sandeel) per year from who_eats_whom_level2.csv, divided by saithe's total
consumption (sum over quarters and ages of ration * N.bar from summary.out, species 19).
Also haddock 0-group (Q3) and 1-group (Q1) abundance and biomass of ages 0-1.
Note: SMS density dependence is built in (size/vulnerability suitability x available
biomass / (sum + other food)); this is the model's functional response, fitted to the
1981-1991 stomachs, not an independent observation.
Usage: python3 -I sms_saithe_haddock_share.py <who_eats_whom_level2.csv> <summary.out>
"""
import sys, csv, collections
eat = collections.defaultdict(lambda: collections.defaultdict(float))
for r in csv.DictReader(open(sys.argv[1])):
    if r["Predator"] == "Saithe":
        eat[int(r["Year"])][r["Prey"]] += float(r["eatenW"])
lines = open(sys.argv[2]).read().split("\n")
h = lines[0].split(); ix = {k: i for i, k in enumerate(h)}
cons = collections.defaultdict(float); had0 = {}; had1 = {}; hadB01 = collections.defaultdict(float)
saiB = collections.defaultdict(float)
for l in lines[1:]:
    f = l.split()
    if len(f) < len(h):
        continue
    y, q, sp, a = int(f[0]), int(f[1]), int(f[2]), int(f[3])
    if sp == 19:
        cons[y] += float(f[ix["ration"]]) * float(f[ix["N.bar"]])
        if q == 1:
            saiB[y] += float(f[ix["BIO"]])
    if sp == 18:
        if a == 0 and q == 3: had0[y] = float(f[ix["N"]])
        if a == 1 and q == 1: had1[y] = float(f[ix["N"]])
        if a <= 1 and q == 3: hadB01[y] += float(f[ix["BIO"]])
print("year saithe_Q1_biomass(t) saithe_cons(t) eaten_haddock(t) haddock_share% whiting% Nor.pout% herring% sandeel% other_food%(by difference) haddock_N0_Q3(1e6) haddock_N1_Q1(1e6) haddock_B01_Q3(t)")
xs, ys = [], []
for y in sorted(cons):
    e = eat.get(y, {})
    c = cons[y]
    sh = {k: 100 * v / c for k, v in e.items()}
    sand = sh.get("N.sandeel", 0) + sh.get("S.sandeel", 0)
    other = 100 - sum(sh.values())
    print(y, "%.0f %.0f %.0f %.2f %.2f %.1f %.1f %.1f %.1f %.0f %.0f %.0f" % (saiB[y], c, e.get("Haddock", 0), sh.get("Haddock", 0), sh.get("Whiting", 0),
          sh.get("Nor.pout", 0), sh.get("Herring", 0), sand, other, had0.get(y, 0) / 1e3, had1.get(y, 0) / 1e3, hadB01[y]))
    if y in had0:
        xs.append(hadB01[y]); ys.append(sh.get("Haddock", 0))
import math
def corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
print("r(haddock 0-1 biomass Q3, saithe haddock share) =", round(corr(xs, ys), 3), "n =", len(xs))
la = [math.log(x) for x in xs]; lb = [math.log(max(y, 1e-3)) for y in ys]
ma, mb = sum(la) / len(la), sum(lb) / len(lb)
slope = sum((x - ma) * (y - mb) for x, y in zip(la, lb)) / sum((x - ma) ** 2 for x in la)
print("log-log slope (share vs haddock 0-1 biomass) = %.2f (1 = share proportional to prey biomass, 0 = fixed share)" % slope)
