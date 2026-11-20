"""Krill stomach content (%BW) of single gadoid stomachs in the ICES stomach records already
linked in research_notes/"Krill over grunda bottnar"/stomachs/tables/records_linked.csv.gz
(o-umlaut in the real folder name).

Purpose (Q1): the "maximum observed" krill meal of whiting, saithe, Norway pout, cod and haddock in
the field, and the krill content of krill-dominated stomachs, as %BW, by length class. A stomach
snapshot is not a ration: the conversion to a ration is done in ration_bounds.py.

Usage: python3 -I ices_krill_content.py <records_linked.csv.gz>
Length-weight (g, cm), as used elsewhere in research_notes: whiting 0.0058 L^3.10, saithe
0.0089 L^3.05, Norway pout 0.0065 L^3.17, haddock 0.0083 L^3.028, cod 0.0076 L^3.08 [J].
"""
import csv
import gzip
import sys
from collections import defaultdict

LW = {"whiting": (0.0058, 3.10), "saithe": (0.0089, 3.05), "Norway pout": (0.0065, 3.17),
      "haddock": (0.0083, 3.028), "cod": (0.0076, 3.08)}
ALIASES = {"whiting": "whiting", "saithe": "saithe", "pout": "Norway pout", "norway pout": "Norway pout",
           "haddock": "haddock", "cod": "cod"}
GROUPS = ["g_U", "g_chaet", "g_cop", "g_crustu", "g_fish", "g_hyp", "g_krill", "g_mysid", "g_other",
          "g_ozoo", "g_pshrimp"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def quant(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def lclass(sp, L):
    edges = {"whiting": [15, 20, 25, 30, 35, 45], "saithe": [20, 30, 40, 50, 60, 80, 120],
             "Norway pout": [8, 12, 15, 18, 25], "haddock": [15, 25, 35, 45, 60],
             "cod": [15, 25, 35, 50, 70, 120]}[sp]
    for a, b in zip(edges[:-1], edges[1:]):
        if a <= L < b:
            return f"{a}-{b}"
    return None


def main(path):
    sp_seen = defaultdict(int)
    data = defaultdict(list)       # (sp, lclass) -> list of (krill %BW, total %BW, krill share, month, bed)
    with gzip.open(path, "rt") as fh:
        for row in csv.DictReader(fh):
            sp_raw = (row["sp"] or "").strip().lower()
            sp_seen[sp_raw] += 1
            sp = ALIASES.get(sp_raw)
            if sp is None:
                continue
            if row["Number"] not in ("1", "1.0"):
                continue
            L = f(row["L_cm"])
            if not L:
                continue
            gk = f(row["g_krill"])
            if gk is None or gk <= 0:
                continue
            tot = sum(f(row[g]) or 0.0 for g in GROUPS)
            a, b = LW[sp]
            W = a * L ** b
            lc = lclass(sp, L)
            if lc is None:
                continue
            data[(sp, lc)].append((100 * gk / W, 100 * tot / W, gk / tot if tot > 0 else 0,
                                   row["month"], row["bed_true"] or row["bed"]))
    print("species names seen (top):", sorted(sp_seen.items(), key=lambda x: -x[1])[:12])
    print()
    print("Single stomachs with krill mass > 0 (wet g). Columns: n; krill content %BW median / p90 / p99 / max;"
          " in krill-dominated stomachs (krill >= 0.8 of the mass): n, total content %BW median / p90 / max")
    for sp in LW:
        for key in sorted((k for k in data if k[0] == sp), key=lambda k: float(k[1].split("-")[0])):
            v = data[key]
            kr = [x[0] for x in v]
            dom = [x[1] for x in v if x[2] >= 0.8]
            print(f"{sp:12s} {key[1]:>7s} cm  n {len(v):4d}  krill %BW {quant(kr, .5):5.2f} / {quant(kr, .9):5.2f}"
                  f" / {quant(kr, .99):5.2f} / {max(kr):5.2f}   dominated n {len(dom):4d}  total %BW "
                  f"{quant(dom, .5):5.2f} / {quant(dom, .9):5.2f} / {max(dom) if dom else float('nan'):5.2f}")
    print()
    print("Same, krill-bearing stomachs by quarter (all lengths): n, krill %BW median / p90 / max")
    for sp in LW:
        byq = defaultdict(list)
        for key, v in data.items():
            if key[0] != sp:
                continue
            for x in v:
                m = f(x[3])
                if m:
                    byq[int((m - 1) // 3) + 1].append(x[0])
        for q in sorted(byq):
            v = byq[q]
            print(f"{sp:12s} Q{q}  n {len(v):4d}  {quant(v, .5):5.2f} / {quant(v, .9):5.2f} / {max(v):5.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
