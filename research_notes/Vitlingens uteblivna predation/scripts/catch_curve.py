"""Catch curves for whiting from NS-IBTS CPUE-at-age per haul (DATRAS getCPUEAge, 1991-2025).

Usage: python3 -I catch_curve.py <cpue_age_all.csv.gz> > tables/catch_curve_stdout.txt
Input: ../../Beståndsreglering/data/cpue_age_all.csv.gz (cached; parsed as text). Nothing downloaded.

Units (mean numbers per hour of GOV, zero hauls included):
  BOX  hauls with 57.7-58.4 N, 9.9-11.3 E (the box of tools/make_depth_map.py), Area code as given
  RFA8 roundfish area 8 (Skagerrak), all depths; RFA8_50 the same at bed >= 50 m
  NNS  roundfish areas 1-3 (northern North Sea + Fladen / Shetland, the northern stock side)
  STOCK areas 1-9: mean of rectangle means (each rectangle one weight), as ../../Beståndsreglering/scripts/mig_cohort_rates.py

Z estimators (Q1 -> Q1 and Q3 -> Q3 same cohort, dt 1 year):
  cohort Z(a -> a+1) = -ln(sum_c D(a+1,y+1) / sum_c D(a,y)), both sums over the cohorts sampled in both years
  (weights = cohort size). 90 % interval: bootstrap over cohorts (year classes).
  catch curve: slope of ln(D) on age within each year class over ages a0..a1, median over year classes, and the
  slope of ln(mean over years of D) on age (a cross-section, assumes steady recruitment; the mean smooths year classes).
Caveats the numbers carry: CPUE-at-age uses DATRAS's area-level age-length keys (the box's age composition is
partly borrowed from the roundfish area); q-at-age is assumed flat from the first age used; GOV sees only the bed
strip and < ~300 m; with 4-5 box hauls a year the box mean is noisy. Z here is everything that lowers the survey index
of a cohort: M + F + net emigration + any drop in q at age.
"""
import csv, gzip, math, random, sys, collections
AP = "Merlangius merlangus"
A = list(range(0, 11))
rows = []
with gzip.open(sys.argv[1], "rt") as f:
    for r in csv.DictReader(f):
        if r["Species"] != AP or r["Quarter"] not in ("1", "3"):
            continue
        rows.append(dict(y=int(r["Year"]), q=int(r["Quarter"]), lat=float(r["ShootLat"]), lon=float(r["ShootLon"]),
                         depth=float(r["Depth"]) if r["Depth"] not in ("", "-9") else -9, area=r["Area"].strip(), rect=r["SubArea"].strip(),
                         a=[float(r[f"Age_{i}"]) if r[f"Age_{i}"] not in ("", "NA", "-9") else 0.0 for i in A]))
print("whiting haul rows Q1+Q3:", len(rows), "years", min(r["y"] for r in rows), max(r["y"] for r in rows))
def in_box(r): return 57.7 <= r["lat"] <= 58.4 and 9.9 <= r["lon"] <= 11.3
UNITS = {"BOX": in_box, "RFA8": lambda r: r["area"] == "8", "RFA8_50": lambda r: r["area"] == "8" and r["depth"] >= 50,
         "NNS": lambda r: r["area"] in ("1", "2", "3"), "STOCK": lambda r: r["area"] in tuple("123456789")}
def index(unit, q, rect_mean=False):
    """D[(y,a)] = mean numbers/h, hauls per year n[y]."""
    sel = [r for r in rows if r["q"] == q and UNITS[unit](r)]
    D, n = {}, collections.Counter()
    for y in sorted({r["y"] for r in sel}):
        s = [r for r in sel if r["y"] == y]
        n[y] = len(s)
        if rect_mean:
            g = collections.defaultdict(list)
            for r in s: g[r["rect"]].append(r)
            for a in A:
                D[(y, a)] = sum(sum(r["a"][a] for r in v) / len(v) for v in g.values()) / len(g)
        else:
            for a in A:
                D[(y, a)] = sum(r["a"][a] for r in s) / len(s)
    return D, n
def z_pool(D, a, ys):
    num = sum(D.get((y + 1, a + 1), 0.0) for y in ys)
    den = sum(D.get((y, a), 0.0) for y in ys)
    return -math.log(num / den) if num > 0 and den > 0 else float("nan")
def boot(D, a, ys, nb=2000, seed=3):
    rnd = random.Random(seed); v = []
    for _ in range(nb):
        s = rnd.choices(ys, k=len(ys)); z = z_pool(D, a, s)
        if not math.isnan(z): v.append(z)
    v.sort(); return (v[int(.05 * len(v))], v[int(.95 * len(v)) - 1]) if v else (float("nan"),) * 2
PERIODS = {"1991-2025": (1991, 2025), "2000-2025": (2000, 2025), "2010-2025": (2010, 2025)}
for unit in UNITS:
    for q in (1, 3):
        D, n = index(unit, q, rect_mean=unit == "STOCK")
        ys_all = sorted(n)
        print(f"\n=== {unit} Q{q}: hauls/yr median {sorted(n.values())[len(n)//2]}, total {sum(n.values())}, years {ys_all[0]}-{ys_all[-1]}")
        print("mean numbers/h by age, 2000-2025:", " ".join(f"{a}:{sum(D[(y,a)] for y in ys_all if y>=2000)/max(1,sum(1 for y in ys_all if y>=2000)):.1f}" for a in range(0, 8)))
        for name, (y0, y1) in PERIODS.items():
            line = []
            for a in range(0, 7):
                ys = [y for y in ys_all if y0 <= y and y + 1 <= y1 and (y + 1) in n]
                z = z_pool(D, a, ys); lo, hi = boot(D, a, ys)
                line.append(f"{a}->{a+1} {z:5.2f} [{lo:5.2f},{hi:5.2f}]")
            print(f"  cohort Z {name}: " + " | ".join(line))
        # catch curve over age ranges, within year class
        for a0, a1 in ((1, 4), (2, 5), (2, 6), (3, 6)):
            slopes = []
            for yc in range(1985, 2023):
                pts = [(a, D.get((yc + a, a))) for a in range(a0, a1 + 1) if (yc + a, a) in D]
                pts = [(a, math.log(d)) for a, d in pts if d and d > 0]
                if len(pts) == a1 - a0 + 1:
                    mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
                    slopes.append(-sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts))
            if slopes:
                slopes.sort(); k = len(slopes)
                print(f"  year-class catch curve ages {a0}-{a1}: median Z {slopes[k//2]:.2f} (p10 {slopes[int(.1*k)]:.2f}, p90 {slopes[int(.9*k)]:.2f}), n classes with all ages > 0: {k}")
            xs = list(range(a0, a1 + 1))
            for y0, y1 in ((2000, 2025),):
                m = [sum(D[(y, a)] for y in ys_all if y0 <= y <= y1) / max(1, sum(1 for y in ys_all if y0 <= y <= y1)) for a in xs]
                if all(v > 0 for v in m):
                    l = [math.log(v) for v in m]; mx = sum(xs) / len(xs); my = sum(l) / len(l)
                    print(f"  cross-section ages {a0}-{a1} {y0}-{y1}: Z {-sum((x - mx) * (v - my) for x, v in zip(xs, l)) / sum((x - mx) ** 2 for x in xs):.2f}")
