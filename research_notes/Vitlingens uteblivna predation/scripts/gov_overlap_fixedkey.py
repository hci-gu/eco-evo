"""COPY of ../../Vitlingens predatorer/scripts/gov_overlap.py with ONE change: the haul key carries ShootLat (as
../../Sejens djupläge/scripts/extract_gov.py does). The original key (year, quarter, ship, haul number) collides
across the North Sea: CPUE rows of a same-numbered haul elsewhere were added to the box haul (e.g. 2016 Q1 26D4
haul 18 at 57.8805 N received the whiting of haul 18 at 54.4449 N). Everything else unchanged.
Usage: python3 -I gov_overlap_fixedkey.py <datras dir> [box|region]
"""
"""Haul-level spatial overlap of whiting with its fish predators (cod >= 20 cm, saithe, hake) and the
day/night GOV catch of whiting, NS-IBTS 2000-2020 Q1/Q3, in and around the Mareld box.

Usage: python3 -I gov_overlap.py <datras dir> [box|region]
  datras dir: research_notes/"Fiskgrupper i rutan"/data/datras (HH_*.xml.gz, CPUEL_*.xml.gz; cached,
  parsed as XML text only). box = 57.7-58.4 N, 9.9-11.3 E (tools/make_depth_map.py bbox); region =
  57.5-58.9 N, 8.5-11.8 E (as gov_by_bed.py in "Krill i Skagerrak"/a_revisit).
Parser as research_notes/"Krill i Skagerrak"/a_revisit/scripts/gov_by_bed.py. Biomass per haul =
sum over length classes of n/h x W(L), W = a L^b (g, cm): whiting 0.0058/3.10, saithe 0.0089/3.05
(as gov_by_bed.py), cod 0.0077/3.07, hake 0.0051/3.13 (FishBase North Sea means [S], J).
GOV catch rates are wing-swept availability near the bed (headline ~4-5 m), q = 1 for every length.

Overlap indices, per quarter (hauls of that quarter; every valid haul counts, zeros included):
  Schoener D = 1 - 0.5 sum_i |p_i - q_i|, p, q = each species' share of its own summed kg/h.
  Lloyd's interspecific overlap L = mean(x_i y_i) / (mean x mean y): the prey density a predator
    tonne meets relative to the mean prey density (1 = independent, the equal-cell assumption of the
    engine at the start; > 1 co-occur, < 1 segregate). This is the factor an encounter rate a x B
    carries when predators and prey are not evenly spread.
  The same L is given for whiting split < 20 cm / >= 20 cm and for 10-28 cm (inside cod's window
  at ratio ~0.40 for 25-70 cm cod).
Day/night: mean kg/h of whiting by DayNight (HH field) and bed depth class, quarter pooled and by Q.
"""
import gzip
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{ices.dk.local/DATRAS}"
SP = {126438: ("whiting", 0.0058, 3.10), 126441: ("saithe", 0.0089, 3.05), 126436: ("cod", 0.0077, 3.07),
      126484: ("hake", 0.0051, 3.13)}
d = Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "box"
LAT, LON = ((57.7, 58.4), (9.9, 11.3)) if mode == "box" else ((57.5, 58.9), (8.5, 11.8))
print(f"region {mode}: {LAT} N, {LON} E")

hauls = {}
for f in sorted(d.glob("HH_*.xml.gz")):
    with gzip.open(f) as fh:
        for _, el in ET.iterparse(fh):
            k = {c.tag.replace(NS, ""): (c.text or "").strip() for c in el}
            if "HaulVal" in k and "ShootLat" in k:
                if k["HaulVal"] != "V" or not k.get("Gear", "").startswith("GOV"):
                    el.clear(); continue
                lat, lon = float(k["ShootLat"]), float(k["ShootLong"])
                if LAT[0] <= lat <= LAT[1] and LON[0] <= lon <= LON[1]:
                    key = (k["Year"], k["Quarter"], k["Ship"].strip(), int(k["HaulNo"]), "%.4f" % float(k["ShootLat"]))
                    hauls[key] = dict(depth=float(k["Depth"]), q=int(k["Quarter"]), dn=k["DayNight"],
                                      kg={"whiting": 0.0, "wh_small": 0.0, "wh_large": 0.0, "wh_10_28": 0.0, "saithe": 0.0,
                                          "cod20": 0.0, "hake": 0.0})
                el.clear()
for f in sorted(d.glob("CPUEL_*.xml.gz")):
    with gzip.open(f) as fh:
        for _, el in ET.iterparse(fh):
            if el.tag != NS + "Cls_Datras_CPUE_Length":
                continue
            r = {c.tag.replace(NS, ""): (c.text or "").strip() for c in el}
            el.clear()
            sp = int(r["AphiaID"]) if r["AphiaID"].lstrip("-").isdigit() else None
            if sp not in SP:
                continue
            h = hauls.get((r["Year"], r["Quarter"], r["Ship"].strip(), int(r["HaulNo"]), "%.4f" % float(r["ShootLat"])))
            if h is None:
                continue
            name, a, b = SP[sp]
            L = float(r["LngtClas"]) / 10.0
            n = float(r["CPUE_number_per_hour"])
            if L <= 0 or n <= 0:
                continue
            kg = n * a * L ** b / 1000.0
            if name == "whiting":
                h["kg"]["whiting"] += kg
                h["kg"]["wh_small" if L < 20 else "wh_large"] += kg
                if 10 <= L < 28:
                    h["kg"]["wh_10_28"] += kg  # whiting inside cod's window at ratio ~0.40
            elif name == "cod":
                if L >= 20:
                    h["kg"]["cod20"] += kg
            else:
                h["kg"][name] += kg

H = list(hauls.values())
print(f"valid GOV hauls: {len(H)}; Q1 {sum(h['q'] == 1 for h in H)}, Q3 {sum(h['q'] == 3 for h in H)}; "
      f"day {sum(h['dn'] == 'D' for h in H)}, night {sum(h['dn'] == 'N' for h in H)}")


def schoener(x, y):
    sx, sy = sum(x), sum(y)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return 1 - 0.5 * sum(abs(a / sx - b / sy) for a, b in zip(x, y))


def lloyd(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    if mx <= 0 or my <= 0:
        return float("nan")
    return (sum(a * b for a, b in zip(x, y)) / n) / (mx * my)


def boot(fn, x, y, nb=1000):
    random.seed(2)
    idx = list(range(len(x)))
    v = []
    for _ in range(nb):
        s = random.choices(idx, k=len(idx))
        r = fn([x[i] for i in s], [y[i] for i in s])
        if not math.isnan(r):
            v.append(r)
    v.sort()
    return v[int(0.05 * len(v))], v[int(0.95 * len(v)) - 1]


print("\nOverlap of whiting (prey) with each predator, per quarter; L with 90 % haul-bootstrap interval")
print(f"{'quarter':8s} {'predator':8s} {'prey':9s} {'n':>4s} {'Schoener':>8s} {'Lloyd L':>8s} {'L p5':>6s} {'L p95':>6s}"
      f" {'pred kg/h':>9s} {'prey kg/h':>9s}")
for q in (1, 3, None):
    sub = [h for h in H if (q is None or h["q"] == q)]
    for pred in ("cod20", "saithe", "hake"):
        for prey in ("whiting", "wh_small", "wh_large", "wh_10_28"):
            x = [h["kg"][pred] for h in sub]
            y = [h["kg"][prey] for h in sub]
            lo, hi = boot(lloyd, x, y)
            print(f"{('Q' + str(q)) if q else 'Q1+Q3':8s} {pred:8s} {prey:9s} {len(sub):4d} {schoener(x, y):8.3f} "
                  f"{lloyd(x, y):8.3f} {lo:6.2f} {hi:6.2f} {sum(x) / len(x):9.2f} {sum(y) / len(y):9.2f}")
# Lloyd per quarter averaged with equal quarter weights (the pooled L mixes the quarters' means)

EDGES = [0, 75, 100, 150, 250, 1000]
print("\nMean kg/h by bed depth class (all valid hauls, Q1+Q3) and the share of each species' summed catch rate")
print(f"{'bed m':>10s} {'n':>4s} " + " ".join(f"{s:>9s}" for s in ("whiting", "wh_small", "wh_large", "cod20", "saithe", "hake")))
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    s = [h for h in H if lo <= h["depth"] < hi]
    if not s:
        continue
    print(f"{lo:4d}-{hi:<5d} {len(s):4d} " + " ".join(
        f"{sum(h['kg'][k] for h in s) / len(s):9.2f}" for k in ("whiting", "wh_small", "wh_large", "cod20", "saithe", "hake")))

print("\nWhiting day vs night (GOV kg/h), by quarter and bed depth class; ratio = night/day mean")
for q in (1, 3, None):
    for lo, hi in ((0, 100), (100, 150), (150, 1000), (0, 1000)):
        s = [h for h in H if lo <= h["depth"] < hi and (q is None or h["q"] == q)]
        dd = [h for h in s if h["dn"] == "D"]
        nn = [h for h in s if h["dn"] == "N"]
        if not dd or not nn:
            print(f"  {('Q' + str(q)) if q else 'Q1+Q3':6s} bed {lo}-{hi} m: day {len(dd)} night {len(nn)} hauls (no pair)")
            continue
        for k in ("whiting", "wh_small", "wh_large"):
            md = sum(h["kg"][k] for h in dd) / len(dd)
            mn = sum(h["kg"][k] for h in nn) / len(nn)
            print(f"  {('Q' + str(q)) if q else 'Q1+Q3':6s} bed {lo:3d}-{hi:<4d} {k:9s} day n {len(dd):3d} {md:8.2f} "
                  f"night n {len(nn):3d} {mn:8.2f}  ratio {mn / md if md > 0 else float('nan'):5.2f}")
