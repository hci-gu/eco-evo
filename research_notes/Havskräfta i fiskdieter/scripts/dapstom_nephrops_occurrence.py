"""Frequency of occurrence of Nephrops in fish stomachs from the Cefas DAPSTOM public portal downloads.

Usage: python3 -I dapstom_nephrops_occurrence.py <dapstom_dir> <out_csv>

<dapstom_dir> (../data/dapstom): CSV files downloaded 2026-10-26 from the Cefas DAPSTOM portal
  https://www.cefas.co.uk/data-and-publications/fish-stomach-records/ (endpoint
  /umbraco/CefasPlugins/DapstomSurface/Download?searchType=predator&species=<code>&years=...&areas=IVa):
  pred_<CODE>_<area>.csv = every prey record of that predator in that ICES division (all years >= 1950
  offered by the portal; cod IVa all years). The portal export has NO prey mass and NO positions, only
  prey presence per stomach (Predator ID), predator mean length, date, division, pooled flag.
Frequency of occurrence (FO) = individual (non-pooled) stomachs with Nephrops / individual stomachs
with any food (records whose only prey is EMPTY are counted as empty). Pooled records are excluded.
FO is not a mass share; it is reported only as a cross-check of the stomach-mass data.
"""
import csv, os, sys, collections

src, out = sys.argv[1], sys.argv[2]
NEPH = {"NEPHROPS NORVEGICUS"}
EMPTY = {"EMPTY"}


def lcls(L, sp):
    try:
        L = float(L)
    except ValueError:
        return "na"
    if sp in ("COD", "HAD", "WHG"):
        edges = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 70), (70, 999)]
    else:
        edges = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 999)]
    for a, b in edges:
        if a <= L < b:
            return f"{a:02d}-{b if b < 999 else ''}"
    return "na"


def period(y):
    y = int(y)
    return "pre1950" if y < 1950 else "1950-79" if y < 1980 else "1980-89" if y < 1990 else \
        "1990-99" if y < 2000 else "2000+"


rows_out = []
for fn in sorted(os.listdir(src)):
    if not (fn.startswith("pred_") and fn.endswith(".csv")):
        continue
    sp, area = fn[5:-4].split("_", 1)
    stom = {}
    with open(os.path.join(src, fn), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["Pooled"].strip().lower() != "n":
                continue
            pid = r["Predator ID"]
            s = stom.setdefault(pid, {"y": r["Year"], "m": r["Date"][3:5], "L": r["Mean length of predator"],
                                      "food": False, "neph": False})
            prey = r["Prey Latin name"].strip().upper()
            if prey in NEPH:
                s["neph"] = True
            if r["Prey common name"].strip().upper() not in EMPTY:
                s["food"] = True
    agg = collections.defaultdict(lambda: [0, 0])
    for s in stom.values():
        if not s["food"]:
            continue
        q = "Q" + str((int(s["m"]) - 1) // 3 + 1) if s["m"].isdigit() else "Q?"
        for per in (period(s["y"]), "all"):
            for L in (lcls(s["L"], sp), "all"):
                for qq in (q, "all"):
                    a = agg[(per, L, qq)]
                    a[0] += 1; a[1] += s["neph"]
    for (per, L, qq), (n, k) in sorted(agg.items()):
        rows_out.append([sp, area, per, L, qq, n, k, round(100 * k / n, 2)])

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["predator_code", "ices_division", "period", "length_cm", "quarter", "stomachs_with_food",
                "with_nephrops", "FO_pct"])
    w.writerows(rows_out)
for r in rows_out:
    if r[4] == "all" and (r[3] == "all" or r[6] > 0):
        print(*r)
