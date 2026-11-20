"""Stomach-side krill occupancy: between-haul vs within-haul krill occurrence and ration, ICES stomach DB.

Input: research_notes/"Krill over grunda bottnar"/stomachs/tables/records_linked.csv.gz (o-umlaut in the real
name; built by that note's scripts from the ICES stomach database). Read-only.

Unit of analysis [D]:
- The HAUL (tblHaulID) is the sampling unit for between-haul occurrence. A haul counts if at least one record of
  the species has prey rows (anyprey) or a fed stomach; it is krill-positive if any record has occ_krill.
- Within-haul occurrence uses single-fish records only (Number == 1), so pooled records cannot hide zeros.
- Pooled records (Number > 1; 1980s) give only an upper bound on the share of fish with krill.
- Strata: area x quarter x year (NE/NW North Sea, Skagerrak, Kattegat). The 2022 IBTS uploads are dropped for
  whiting (mostly records without prey rows; flagged in "Krillparen mot vintermagar"), reported separately.
- Ration index per record: sqrt(S/S_max) with S = krill g per stomach (pool mean; counts x 0.19 g when not
  weighed), S_max 2.5 %BW (the central band of "Krillintag och jaktsatt"). The band factor cancels in ratios.
- Effective occupancy from rations: E[ration | all hauls] / E[ration | krill-positive hauls], hauls weighted by
  stomachs. Also against the krill-dominated mean (krill >= 0.8 of counted prey) at the same central band.
- Intra-haul correlation (ICC) of krill presence (single-fish records, hauls with >= 2 fed fish): one-way ANOVA
  estimator on the 0/1 variable.

Usage: python3 -I stomach_occupancy.py <records_linked.csv.gz>
"""
import csv, gzip, math, sys
from collections import defaultdict

LW = {"whiting": (0.0058, 3.10), "saithe": (0.0089, 3.05), "norway pout": (0.0065, 3.17),
      "haddock": (0.0083, 3.028), "cod": (0.0076, 3.08)}
AREAS = {"NE North Sea", "NW North Sea", "Skagerrak", "Kattegat"}
KRILL_G = 0.19
SMAX = 0.025


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def ration(r, sp):
    """Central-band krill ration index per stomach, or None if krill present but unquantified."""
    L0, N = f(r["L_cm"]), f(r["Number"]) or 1.0
    if not L0:
        return None
    classed = r["year"] in ("1980", "1981", "1982", "1987", "1991") and abs(L0 - round(L0 / 5) * 5) < 1e-6 and N > 1
    L = L0 + (5.0 if sp == "saithe" else 2.5) if classed else L0
    W = LW[sp][0] * L ** LW[sp][1]
    occ = f(r["occ_krill"]) or 0
    gk, kc = f(r["g_krill"]), f(r["krill_count"])
    if occ and not gk and not kc:
        return None
    S = (gk if gk else (kc * KRILL_G if kc else 0.0)) / N
    return math.sqrt(S / (SMAX * W)) if S > 0 else 0.0


def length_ok(r, sp):
    L = f(r["L_cm"])
    if sp == "whiting":
        return L is not None and L >= 20
    return True


hauls = defaultdict(lambda: {"recs": []})
with gzip.open(sys.argv[1], "rt") as fh:
    for r in csv.DictReader(fh):
        sp = r["sp"].strip().lower()
        if sp not in LW or r["area"] not in AREAS or not length_ok(r, sp):
            continue
        if sp == "whiting" and r["year"] == "2022":
            sp = "whiting_2022"
        key = (sp, r["tblHaulID"])
        h = hauls[key]
        h.update(area=r["area"], q=r["quarter"], year=r["year"], bed=r["bed_true"] or r["bed"], rect=r["rect"])
        h["recs"].append(r)

# per-haul summaries
H = []
dropped_noprey = defaultdict(int)
unquant = defaultdict(int)
for (sp, hid), h in hauls.items():
    recs = h["recs"]
    if not any(r["anyprey"] == "True" for r in recs):
        dropped_noprey[sp] += 1      # no prey rows at all: emptiness unknown (e.g. saithe 1980 uploads)
        continue
    spl = "whiting" if sp == "whiting_2022" else sp
    pos = any((f(r["occ_krill"]) or 0) > 0 for r in recs)
    nst = sum(f(r["Number"]) or 1.0 for r in recs)
    single = [r for r in recs if (f(r["Number"]) or 1.0) == 1.0 and r["anyprey"] == "True"]
    sk = sum(1 for r in single if (f(r["occ_krill"]) or 0) > 0)
    rs = [(ration(r, spl), f(r["Number"]) or 1.0) for r in recs]
    rq = [(x, n) for x, n in rs if x is not None]
    if any(x is None for x, _ in rs):
        unquant[sp] += 1             # krill present but unquantified (or no length): haul left out of ration ratios
        rat = None
    else:
        rat = sum(x * n for x, n in rq) / sum(n for _, n in rq) if rq else None
    # leave-one-out within-haul FO: each single fish whose haul-mates (other single fish) include a krill eater
    loo_n = loo_k = 0
    for i, r in enumerate(single):
        mates = sk - (1 if (f(r["occ_krill"]) or 0) > 0 else 0)
        if mates > 0:
            loo_n += 1
            loo_k += 1 if (f(r["occ_krill"]) or 0) > 0 else 0
    # krill-dominated records in the haul
    dom = []
    for r in recs:
        kc, allc = f(r["krill_count"]), f(r["all_count"])
        if kc and allc and kc / allc >= 0.8:
            x = ration(r, spl)
            if x is not None:
                dom.append((x, f(r["Number"]) or 1.0))
    H.append(dict(sp=sp, area=h["area"], q=h["q"], year=h["year"], bed=h["bed"], rect=h["rect"], pos=pos, nst=nst,
                  n1=len(single), k1=sk, rat=rat, dom=dom, loo_n=loo_n, loo_k=loo_k))


def icc(groups):
    """ANOVA ICC for 0/1 data; groups = list of (n, k)."""
    g = [(n, k) for n, k in groups if n >= 2]
    if len(g) < 3:
        return None, len(g)
    N = sum(n for n, _ in g)
    K = sum(k for _, k in g)
    p = K / N
    msb = sum(n * (k / n - p) ** 2 for n, k in g) / (len(g) - 1)
    msw = sum(k * (1 - k / n) for n, k in g) / (N - len(g))
    n0 = (N - sum(n * n for n, _ in g) / N) / (len(g) - 1)
    if msb + (n0 - 1) * msw == 0:
        return None, len(g)
    return (msb - msw) / (msb + (n0 - 1) * msw), len(g)


def block(rows, label):
    n = len(rows)
    if n == 0:
        return
    npos = sum(r["pos"] for r in rows)
    st = sum(r["nst"] for r in rows)
    # within-haul single-fish FO
    pos1 = [r for r in rows if r["pos"] and r["n1"] >= 2]
    n1 = sum(r["n1"] for r in pos1)
    k1 = sum(r["k1"] for r in pos1)
    all1 = [r for r in rows if r["n1"] >= 1]
    N1 = sum(r["n1"] for r in all1)
    K1 = sum(r["k1"] for r in all1)
    ic, ng = icc([(r["n1"], r["k1"]) for r in rows if r["n1"] >= 2])
    ln = sum(r["loo_n"] for r in rows)
    lk = sum(r["loo_k"] for r in rows)
    # rations
    rr = [r for r in rows if r["rat"] is not None]
    wall = sum(r["nst"] for r in rr)
    mall = sum(r["rat"] * r["nst"] for r in rr) / wall if wall else float("nan")
    rp = [r for r in rr if r["pos"]]
    wpos = sum(r["nst"] for r in rp)
    mpos = sum(r["rat"] * r["nst"] for r in rp) / wpos if wpos else float("nan")
    dom = [d for r in rows for d in r["dom"]]
    wd = sum(n for _, n in dom)
    mdom = sum(x * n for x, n in dom) / wd if wd else float("nan")
    s = (f"{label:44s} hauls {n:4d} pos {npos:4d} P_h {npos / n:4.2f} stom {st:6.0f} | "
         f"single-fish FO all {K1 / N1 if N1 else float('nan'):4.2f} (n {N1:4d}) in pos hauls {k1 / n1 if n1 else float('nan'):4.2f} (n {n1:4d}, hauls {len(pos1):3d}) "
         f"LOO FO {lk / ln if ln else float('nan'):4.2f} (n {ln:3d}) ICC {ic if ic is not None else float('nan'):5.2f} (hauls {ng:3d}) | ration hauls {len(rr):3d} all {mall:5.3f} pos {mpos:5.3f} "
         f"all/pos {mall / mpos if mpos and mpos == mpos and mpos > 0 else float('nan'):4.2f} dom {mdom:5.2f} all/dom {mall / mdom if mdom == mdom and mdom > 0 else float('nan'):4.2f}")
    print(s)


print(__doc__.split("Usage")[0])
print("Columns: hauls with fed fish of the species; krill-positive hauls; P_h = share positive; stomachs;"
      " single-fish FO overall and inside positive hauls (hauls with >= 2 single fish); ICC; mean ration index"
      " (sqrt(S/S_max), GER-Cmax units, central band) over all hauls, over positive hauls, their ratio, the krill-dominated"
      " record mean and all/dom.\n")
print("LOO FO = share of single fish with krill among single fish whose haul-mates include a krill eater (removes the"
      " selection of the fish that made the haul positive).\n")
for sp in ["whiting", "saithe", "norway pout", "cod", "haddock", "whiting_2022"]:
    rows = [r for r in H if r["sp"] == sp]
    if not rows:
        continue
    print(f"=== {sp}")
    block(rows, "ALL")
    strata = defaultdict(list)
    for r in rows:
        strata[(r["area"], r["q"], r["year"])].append(r)
    for k in sorted(strata):
        if len(strata[k]) >= 3:
            block(strata[k], f"{k[0]} Q{k[1]} {k[2]}")
    beds = defaultdict(list)
    for r in rows:
        beds[r["bed"]].append(r)
    for k in sorted(beds):
        if len(beds[k]) >= 3:
            block(beds[k], f"bed {k}")
    # within- vs between-rectangle split of haul krill presence (ICES rectangle ~ 55 x 55 km, same year and quarter)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["rect"], r["year"], r["q"])].append(1.0 if r["pos"] else 0.0)
    multi = [v for v in groups.values() if len(v) >= 2]
    if multi:
        allv = [x for v in multi for x in v]
        m = sum(allv) / len(allv)
        tot = sum((x - m) ** 2 for x in allv)
        within = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in multi)
        mixed = sum(1 for v in multi if 0 < sum(v) < len(v))
        null = (len(allv) - len(multi)) / (len(allv) - 1)   # expected within share if presence ignored the rectangle
        print(f"  rectangle x year x quarter groups with >= 2 hauls: {len(multi)} ({len(allv)} hauls, P_h {m:.2f});"
              f" groups with mixed krill presence {mixed}; share of presence variance WITHIN rectangles {within / tot if tot else float('nan'):.2f}"
              f" (null without clustering (N-G)/(N-1) = {null:.2f})")
    # stratum-level P_h distribution among strata with >= 5 hauls
    ph = sorted(sum(v["pos"] for v in s) / len(s) for s in strata.values() if len(s) >= 5)
    if ph:
        print(f"  strata with >= 5 hauls: {len(ph)}; P_h min {ph[0]:.2f} median {ph[len(ph) // 2]:.2f} max {ph[-1]:.2f};"
              f" strata with P_h in (0.1, 0.9): {sum(0.1 < x < 0.9 for x in ph)}")
    print()

print("hauls dropped for no prey rows at all:", dict(dropped_noprey))
print("hauls left out of the ration ratios (krill present but unquantified, or no length):", dict(unquant))
