"""Haul-to-haul spread of krill catches in and near the box, from tables already transcribed in
research_notes/"Krill i Skagerrak"/tables (read-only):
- bohle1991_table1_box.csv: Bohle & Moksness 1991 (ICES CM 1991/L:4) Table 1, IKMT night hauls 5-65 m (oblique,
  ~30 min at 2.5 kn = ~2.3 km of tow), litres per haul, min / max / mean per station and winter, Dec-Apr 1987-91.
  Repeated hauls at a FIXED station across cruises of one winter: a temporal (weeks) spread at one place.
- nsec_mnorvegica.csv: IMR North Sea ecosystem cruise macroplankton trawl, M. norvegica g/m3 x 1e4 per station,
  Norwegian Trench and Skagerrak, April-May 2023 and 2025: a spatial spread between stations tens of km apart.

Derived [D]:
- share of station-winters with at least one zero haul (min 0) and max/mean;
- the two-state occupancy estimate p2 = mean/max: if a station holds krill at density D_in in a share p of hauls and
  ~0 otherwise, mean = p D_in and max = D_in, so p = mean/max (biased low when the in-patch density itself varies,
  high when n is small and no haul hit a patch; n is 2-7 here);
- lognormal sigma implied by E[max]/mean for n draws (simulated; saturates at sigma 4 when one haul holds nearly
  all the catch, i.e. max/mean close to n: then read p2 only);
- CV and log-sd of the NSEC station values; Lloyd's patchiness 1 + CV^2 and the two-state occupancy
  p_eff = 1 / (1 + CV^2) at that support.

Usage: python3 -I haul_spread.py <bohle1991_table1_box.csv> <nsec_mnorvegica.csv>
"""
import csv, math, random, statistics, sys

random.seed(1)


def emax_over_mean(sigma, n, reps=4000):
    acc = 0.0
    for _ in range(reps):
        xs = [math.exp(random.gauss(0, sigma)) for _ in range(n)]
        acc += max(xs) / (sum(xs) / n)
    return acc / reps


def sigma_from_ratio(ratio, n):
    lo, hi = 0.01, 4.0
    if ratio <= 1.0:
        return 0.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if emax_over_mean(mid, n, 1500) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print(__doc__.split("Usage")[0])
rows = list(csv.DictReader(open(sys.argv[1])))
print("== Bohle & Moksness 1991, fixed stations, repeated night IKMT hauls within one winter")
print("station winter bed_m n min max mean max/mean implied_lognormal_sigma CV_lognormal in_box")
sig = []
p2s = []
zero = 0
for r in rows:
    n = int(r["n_hauls"])
    mn, mx, me = float(r["min_l"]), float(r["max_l"]), float(r["mean_l"])
    if mx < me:      # printed inconsistency (mean above max), flagged in "Krill i Skagerrak" 1.1
        print(f"{r['station']:>4s} {r['season']} {r['echo_depth_m']:>4s} n {n} min {mn} max {mx} mean {me}: mean > max, skipped")
        continue
    ratio = mx / me if me > 0 else float("nan")
    s = sigma_from_ratio(ratio, n) if n >= 3 else float("nan")
    if n >= 3:
        sig.append(s)
    zero += mn == 0.0
    cv = math.sqrt(math.exp(s * s) - 1) if s == s else float("nan")
    p2s.append(me / mx)
    print(f"{r['station']:>4s} {r['season']} {r['echo_depth_m']:>4s} n {n} min {mn:5.2f} max {mx:5.2f} mean {me:5.2f}"
          f" max/mean {ratio:4.2f} p2 {me / mx:4.2f} sigma {s:4.2f} CV {cv:4.2f} {'box' if r['in_box'] == '1' else 'out'}")
print(f"station-winters: {len(rows)}; with at least one zero haul: {zero}")
sig.sort()
if sig:
    med = sig[len(sig) // 2]
    print(f"implied lognormal sigma (n >= 3 stations, excluding zeros): min {sig[0]:.2f} median {med:.2f} max {sig[-1]:.2f};"
          f" CV at the median {math.sqrt(math.exp(med * med) - 1):.2f}; p_eff = 1/(1+CV^2) = {1 / math.exp(med * med):.2f}")
p2s.sort()
print(f"two-state occupancy p2 = mean/max over station-winters (n {len(p2s)}): min {p2s[0]:.2f} quartiles "
      f"{p2s[len(p2s) // 4]:.2f} / {p2s[len(p2s) // 2]:.2f} / {p2s[3 * len(p2s) // 4]:.2f} max {p2s[-1]:.2f}")
p5 = sorted(float(r["mean_l"]) / float(r["max_l"]) for r in rows if int(r["n_hauls"]) >= 5 and float(r["max_l"]) >= float(r["mean_l"]))
print(f"  p2 cannot fall below 1/n (one positive haul). Stations with n >= 5 only (floor <= 0.2): n {len(p5)}, min {p5[0]:.2f} median "
      f"{p5[len(p5) // 2]:.2f} max {p5[-1]:.2f}")
deep = [float(r["mean_l"]) / float(r["max_l"]) for r in rows if float(r["echo_depth_m"]) >= 150 and float(r["max_l"]) >= float(r["mean_l"])]
shal = [float(r["mean_l"]) / float(r["max_l"]) for r in rows if float(r["echo_depth_m"]) < 150 and float(r["max_l"]) >= float(r["mean_l"])]
print(f"  beds >= 150 m: median p2 {sorted(deep)[len(deep) // 2]:.2f} (n {len(deep)}); beds < 150 m: median p2 {sorted(shal)[len(shal) // 2]:.2f} (n {len(shal)})")
means = [float(r["mean_l"]) for r in rows if r["in_box"] == "1"]
cvm = statistics.pstdev(means) / statistics.mean(means)
print(f"in-box station-winter means (n {len(means)}): CV between stations {cvm:.2f}; p_eff {1 / (1 + cvm * cvm):.2f}")
m2 = [float(r["mean_l"]) for r in rows if r["in_box"] == "1" and not (r["station"] == "51" and r["season"] == "89-90")]
cv2 = statistics.pstdev(m2) / statistics.mean(m2)
print(f"  without st. 51 1989-90 (one 30 l haul in 3, flagged by the authors): n {len(m2)} CV {cv2:.2f}; p_eff {1 / (1 + cv2 * cv2):.2f}")

print()
print("== IMR NSEC macroplankton trawl, M. norvegica, stations tens of km apart, April-May")
ns = list(csv.DictReader(open(sys.argv[2])))
for label, sel in (("all", lambda r: True), ("2023", lambda r: r["year"] == "2023"), ("2025", lambda r: r["year"] == "2025"),
                   ("west/north trench", lambda r: r["region"].startswith("west")),
                   ("east / Skagerrak", lambda r: r["region"].startswith("east"))):
    v = [float(r["mn_1e4_g_m3"]) for r in ns if sel(r)]
    if len(v) < 3:
        continue
    m = statistics.mean(v)
    cv = statistics.pstdev(v) / m
    ls = statistics.pstdev([math.log(x) for x in v])
    print(f"{label:18s} n {len(v):2d} mean {m:6.1f} median {statistics.median(v):6.1f} CV {cv:4.2f} log-sd {ls:4.2f}"
          f" Lloyd 1+CV^2 {1 + cv * cv:4.2f} p_eff {1 / (1 + cv * cv):4.2f}; max/mean {max(v) / m:4.1f}")
v = [float(r["mn_1e4_g_m3"]) for r in ns if r["station"] != "418"]
m = statistics.mean(v)
cv = statistics.pstdev(v) / m
print(f"all without the 440 outlier (st. 418) n {len(v)} CV {cv:4.2f} p_eff {1 / (1 + cv * cv):4.2f}")
