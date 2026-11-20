"""Addition or substitution: is total stomach content higher in krill-bearing records than in krill-free ones
taken in the same stratum (species, year, quarter, area)?

Reading [J]:
- acquisition-limited fish: krill met adds to the other food, so krill-bearing records hold MORE in total and
  about the same non-krill food;
- regulated (appetite-limited) fish: krill replaces other food, so the totals are about EQUAL and the non-krill
  part is lower in krill-bearing records.
A pool (Number > 1) is krill-bearing when any stomach in it held krill, which dilutes both contrasts.

Input: research_notes/"Krill over grunda bottnar"/stomachs/tables/records_linked.csv.gz (o-umlaut in the real
folder name). Usage: python3 -I ices_addition.py <records_linked.csv.gz>
Weighed records only (any g_* > 0 or a recorded empty stomach), with a length. Length-weight (g, cm) as in
research_notes/"Krillintag och jaktsatt" (whiting 0.0058 L^3.10, saithe 0.0089 L^3.05, Norway pout
0.0065 L^3.17, cod 0.0076 L^3.08, haddock 0.0083 L^3.028). Pooled 5-cm classes (1980-82, 1987, 1991) +2.5 cm
(saithe 10-cm classes +5).
Content S = prey wet mass per stomach / body mass, empties included (%BW). The GER ration scales as sqrt(S)
at a fixed length and temperature (WGSAM 2024 Table 3-7), so the ratio of mean sqrt(S) is also printed
(Jensen: a pool mean overstates sqrt slightly).
"""
import csv, gzip, math, sys
from collections import defaultdict

G = ["g_U", "g_chaet", "g_cop", "g_crustu", "g_fish", "g_hyp", "g_krill", "g_mysid", "g_other", "g_ozoo", "g_pshrimp"]
LW = {"whiting": (0.0058, 3.10), "saithe": (0.0089, 3.05), "norway pout": (0.0065, 3.17),
      "cod": (0.0076, 3.08), "haddock": (0.0083, 3.028)}
LMIN = {"whiting": 20.0}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


strata = defaultdict(lambda: {True: [], False: []})
hauls = defaultdict(lambda: {True: [], False: []})
counted = defaultdict(lambda: {True: [0, 0, 0], False: [0, 0, 0]})  # recs, stomachs, recs with fish prey
with gzip.open(sys.argv[1], "rt") as fh:
    for r in csv.DictReader(fh):
        sp = r["sp"].strip().lower()
        if sp not in LW:
            continue
        L0 = f(r["L_cm"])
        if not L0 or L0 < LMIN.get(sp, 0):
            continue
        N = f(r["Number"]) or 1.0
        kc, gk = f(r["krill_count"]) or 0.0, f(r["g_krill"]) or 0.0
        krill = kc > 0 or gk > 0
        tot = sum((f(r[g]) or 0.0) for g in G)
        key = (sp, r["year"], r["quarter"], r["area"])
        if tot <= 0:
            # counted-only or empty: keep fish-prey occurrence for counted records with any prey
            if (f(r["all_count"]) or 0) > 0:
                c = counted[key][krill]
                c[0] += 1; c[1] += N; c[2] += (f(r["occ_fish"]) or 0) > 0
            empty = r["StomachEmpty"].strip() in ("1", "1.0") or (f(r["fed"]) == 0 and (f(r["all_count"]) or 0) == 0)
            if not empty:
                continue
        classed = r["year"] in ("1980", "1981", "1982", "1987", "1991") and abs(L0 - round(L0 / 5) * 5) < 1e-6 and N > 1
        L = L0 + (5.0 if sp == "saithe" else 2.5) if classed else L0
        a, b = LW[sp]
        W = a * L ** b
        S = 100 * tot / N / W
        Sk = 100 * gk / N / W
        strata[key][krill].append((N, S, Sk, S - Sk, tot > 0))
        hauls[(sp, r["year"], r["quarter"], r["area"], r["tblHaulID"])][krill].append((N, S, Sk, S - Sk, tot > 0))

print("Weighed records (incl. recorded empties), whiting >= 20 cm. Per stratum: krill-bearing (K) vs krill-free (0):")
print("records, stomachs, stomach-weighted mean total content %BW, mean non-krill %BW, mean sqrt-ration ratio K/0")
rows = []
for key in sorted(strata):
    d = strata[key]
    if not d[True] or not d[False]:
        continue
    if not any(x[4] for x in d[True] + d[False]):
        continue  # empties only: the year was counted, not weighed
    out = {}
    for k in (True, False):
        v = d[k]
        n = sum(x[0] for x in v)
        out[k] = (len(v), n, sum(x[0] * x[1] for x in v) / n, sum(x[0] * x[3] for x in v) / n,
                  sum(x[0] * math.sqrt(max(x[1], 0)) for x in v) / n)
    K, Z = out[True], out[False]
    ratio_tot = K[2] / Z[2] if Z[2] > 0 else float("inf")
    ratio_nk = K[3] / Z[3] if Z[3] > 0 else float("inf")
    ratio_sqrt = K[4] / Z[4] if Z[4] > 0 else float("inf")
    rows.append((key, K, Z, ratio_tot, ratio_nk, ratio_sqrt))
    print(f"{key[0]:12s} {key[1]} Q{key[2]} {key[3]:13s}  K: rec {K[0]:3d} stom {K[1]:5.0f} tot {K[2]:5.2f} nonkrill {K[3]:5.2f}"
          f" | 0: rec {Z[0]:3d} stom {Z[1]:5.0f} tot {Z[2]:5.2f}  | tot K/0 {ratio_tot:5.2f}  nonkrill K/0 {ratio_nk:5.2f}  sqrt-ration K/0 {ratio_sqrt:5.2f}")

print()
print("Same strata, fed records only (records with any prey mass; empty records dropped from both groups, since")
print("holding krill selects recently fed fish):")
for key in sorted(strata):
    d = strata[key]
    K = [x for x in d[True] if x[4]]
    Z = [x for x in d[False] if x[4]]
    if not K or not Z:
        continue
    nk, nz = sum(x[0] for x in K), sum(x[0] for x in Z)
    tk = sum(x[0] * x[1] for x in K) / nk
    tz = sum(x[0] * x[1] for x in Z) / nz
    ok = sum(x[0] * x[3] for x in K) / nk
    print(f"{key[0]:12s} {key[1]} Q{key[2]} {key[3]:13s}  K: rec {len(K):3d} tot {tk:5.2f} nonkrill {ok:5.2f} | 0: rec {len(Z):3d} tot {tz:5.2f}"
          f"  | tot K/0 {tk / tz:5.2f}  nonkrill K/0 {ok / tz:5.2f}")

print()
print("Counted-only records with prey (no weights): share of records holding fish prey, krill-bearing vs krill-free")
for key in sorted(counted):
    c = counted[key]
    if c[True][0] < 3 or c[False][0] < 3:
        continue
    pk = c[True][2] / c[True][0]
    p0 = c[False][2] / c[False][0]
    print(f"{key[0]:12s} {key[1]} Q{key[2]} {key[3]:13s}  K: rec {c[True][0]:3d} fish-occ {pk:4.2f} | 0: rec {c[False][0]:3d} fish-occ {p0:4.2f}")

print()
print("Within-haul pairs: hauls holding both krill-bearing and krill-free weighed records of the species.")
print("Per stratum: number of hauls, stomachs K / 0, haul-averaged ratio of mean total content K/0 (geometric mean")
print("over hauls with a non-zero krill-free total), mean non-krill K/0, and the hauls where K > 0 in total")
agg = defaultdict(list)
for hk, d in hauls.items():
    if not d[True] or not d[False]:
        continue
    if not any(x[4] for x in d[True] + d[False]):
        continue
    def m(v, i):
        n = sum(x[0] for x in v)
        return sum(x[0] * x[i] for x in v) / n, n
    tk, nk = m(d[True], 1)
    t0, n0 = m(d[False], 1)
    ok, _ = m(d[True], 3)
    agg[hk[:4]].append((tk, t0, ok, nk, n0))
for key in sorted(agg):
    v = agg[key]
    good = [x for x in v if x[1] > 0]
    if not good:
        continue
    gm = math.exp(sum(math.log(max(x[0], 1e-4) / x[1]) for x in good) / len(good))
    gm_nk = math.exp(sum(math.log(max(x[2], 1e-4) / x[1]) for x in good) / len(good))
    up = sum(x[0] > x[1] for x in good)
    print(f"{key[0]:12s} {key[1]} Q{key[2]} {key[3]:13s} hauls {len(v):3d} (usable {len(good):3d}) stom K {sum(x[3] for x in v):4.0f} / 0 {sum(x[4] for x in v):4.0f}"
          f"  total K/0 geo-mean {gm:5.2f}  non-krill(K)/total(0) geo-mean {gm_nk:5.2f}  hauls with K > 0: {up}/{len(good)}")
