"""A17 (SHARK, WP2 90 um, 25-0 m) monthly abundance by taxon and Calanus stage.

Usage: python3 -I a17_stage_season.py <path to SHARK a17_zoo.txt> [median]

Purpose (research note "Djurplanktonets diapaus"): when do Calanus C5 leave
the 0-25 m layer (descent to diapause), and when are the small taxa with
resting eggs or overwintering stages (Centropages hamatus, Acartia, Temora,
Pseudocalanus, Microcalanus) present in the surface layer.

Method: one value per visit (sample id), zero where a taxon was not reported
in a counted visit; mean (or median) over visits per calendar month (2007-2023).
Carbon (ug C/m3 rows only, 2019-2023): Calanus stage carbon, mean of visits.
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
MEDIAN = len(sys.argv) > 2 and sys.argv[2] == "median"
csv.field_size_limit(10 ** 9)

CAL = {"Calanus finmarchicus", "Calanus helgolandicus", "Calanus",
       "Calanus hyperboreus", "Calanus glacialis"}


def stage_group(code):
    c = (code or "").replace(" ", "")
    if c == "NP":
        return "NP"
    if c in ("C1", "C2", "C3", "C1-C3"):
        return "C1-3"
    if c in ("C4", "C4-C5"):
        return "C4"
    if c == "C5":
        return "C5"
    if c == "AD":
        return "AD"
    return "other"


def small_group(name):
    for key in ("Centropages hamatus", "Centropages typicus", "Acartia",
                "Temora", "Pseudocalanus", "Paracalanus", "Oithona",
                "Microcalanus", "Metridia", "Euphausiacea", "Meganyctiphanes",
                "Thysanoessa", "Evadne", "Podon", "Penilia"):
        if name.startswith(key):
            return "Euphausiids" if key in ("Euphausiacea", "Meganyctiphanes",
                                            "Thysanoessa") else key
    if name == "Centropages":
        return "Centropages spp."
    return None


visits = {}                       # sample_id -> month
ab = defaultdict(float)           # (sample_id, key) -> ind/m3
carb = defaultdict(float)         # (sample_id, key) -> ug C/m3
carb_visits = {}
with open(path, newline="", encoding="utf-8") as fh:
    rd = csv.DictReader(fh, delimiter="\t")
    for r in rd:
        # sample_id is empty for 2007-2018 visits: key on date + id + depth
        sid = "|".join((r["sample_date"], r["sample_id"],
                        r["sample_min_depth_m"], r["sample_max_depth_m"]))
        try:
            month = int(r["visit_month"])
            val = float(r["value"])
        except ValueError:
            continue
        name = r["scientific_name"]
        par, unit = r["parameter"], r["unit"]
        keys = []
        if name in CAL:
            keys.append("Calanus " + stage_group(r["dev_stage_code"]))
            keys.append("Calanus all")
            sp = {"Calanus finmarchicus": "Cfin", "Calanus helgolandicus":
                  "Chel", "Calanus": "Cspp"}.get(name, "Coth")
            keys.append(sp + " " + stage_group(r["dev_stage_code"]))
            keys.append(sp + " all")
        if name.startswith("Pseudocalanus"):
            keys.append("Pseu " + stage_group(r["dev_stage_code"]))
        g = small_group(name)
        if g:
            keys.append(g)
        if par == "Abundance" and unit == "ind/m3":
            visits[sid] = month
            for k in keys:
                ab[(sid, k)] += val
        elif par == "Carbon content" and unit == "ug/m3":
            carb_visits[sid] = month
            carb[(sid, "TOTAL")] += val
            for k in keys:
                carb[(sid, k)] += val

SPLIT = [sp + " " + st for sp in ("Cfin", "Chel", "Cspp")
         for st in ("C1-3", "C4", "C5", "AD", "all")]
SPLIT += ["Pseu C1-3", "Pseu C4", "Pseu C5", "Pseu AD"]
ab_keys = SPLIT + ["Calanus C1-3", "Calanus C4", "Calanus C5", "Calanus AD",
           "Calanus NP", "Calanus all", "Metridia", "Pseudocalanus",
           "Microcalanus", "Paracalanus", "Temora", "Acartia",
           "Centropages hamatus", "Centropages typicus", "Centropages spp.",
           "Oithona", "Evadne", "Podon", "Penilia", "Euphausiids"]


def monthly(store, vis, keys):
    out = {}
    for k in keys:
        row = []
        for m in range(1, 13):
            sids = [s for s, mm in vis.items() if mm == m]
            vals = [store.get((s, k), 0.0) for s in sids]
            if MEDIAN:
                vals = sorted(vals)
                n = len(vals)
                row.append((vals[n // 2] if n % 2 else
                            0.5 * (vals[n // 2 - 1] + vals[n // 2]))
                           if vals else float("nan"))
            else:
                row.append((sum(vals) / len(vals)) if vals else float("nan"))
        out[k] = row
    nvis = [sum(1 for mm in vis.values() if mm == m) for m in range(1, 13)]
    return out, nvis


months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
res, nvis = monthly(ab, visits, ab_keys)
print("A17 0-25 m ABUNDANCE, ind/m3, " + ("median" if MEDIAN else "mean") + " of visits per month (2007-2023)")
print(f"{'taxon':22s}" + "".join(f"{m:>8s}" for m in months))
print(f"{'n visits':22s}" + "".join(f"{n:8d}" for n in nvis))
for k in ab_keys:
    print(f"{k:22s}" + "".join(f"{v:8.1f}" for v in res[k]))

ck = SPLIT + ["TOTAL", "Calanus C1-3", "Calanus C4", "Calanus C5", "Calanus AD",
      "Calanus all", "Metridia"]
cres, cn = monthly(carb, carb_visits, ck)
print()
print("A17 0-25 m CARBON, ug C/m3, " + ("median" if MEDIAN else "mean") + " of visits per month (2019-2023)")
print(f"{'taxon':22s}" + "".join(f"{m:>8s}" for m in months))
print(f"{'n visits':22s}" + "".join(f"{n:8d}" for n in cn))
for k in ck:
    print(f"{k:22s}" + "".join(f"{v:8.0f}" for v in cres[k]))
print(f"{'C5 share of Calanus C':22s}" + "".join(
    f"{(c5 / a if a > 0 else float('nan')):8.2f}"
    for c5, a in zip(cres["Calanus C5"], cres["Calanus all"])))
print(f"{'Calanus share of total':22s}" + "".join(
    f"{(a / t if t > 0 else float('nan')):8.2f}"
    for a, t in zip(cres["Calanus all"], cres["TOTAL"])))
print(f"{'Calanus C5 / TOTAL C':22s}" + "".join(
    f"{(c5 / t if t > 0 else float('nan')):8.2f}"
    for c5, t in zip(cres["Calanus C5"], cres["TOTAL"])))
print(f"{'resident (no Cal/Met)':22s}" + "".join(
    f"{(t - a - m):8.0f}"
    for t, a, m in zip(cres["TOTAL"], cres["Calanus all"], cres["Metridia"])))
print(f"{'C1-C4 share of Cal. C':22s}" + "".join(
    f"{((x + y) / a if a > 0 else float('nan')):8.2f}"
    for x, y, a in zip(cres["Calanus C1-3"], cres["Calanus C4"],
                       cres["Calanus all"])))
