"""Saithe (SMS predator 19) observed relative stomach contents by weight from the
North Sea SMS key run 2023 (ICES WGSAM), file summary_stom.out
(https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023).
stomcon = observed relative stomach content weight (sums to 1 per predator group);
stomcon.hat = model fit. Prey 0 = 'other food' (everything not an SMS species:
krill, flatfish, Maurolicus, poor cod, blue whiting, ...).
Usage: python3 -I sms_saithe_stomachs.py <summary_stom.out>
"""
import sys, collections
PREY = {0: "other", 16: "cod", 17: "whiting", 18: "haddock", 21: "herring",
        22: "N.sandeel", 23: "S.sandeel", 24: "Nor.pout", 25: "sprat"}
rows = open(sys.argv[1]).read().split("\n")
hdr = rows[0].split()
ix = {h: i for i, h in enumerate(hdr)}
grp = collections.defaultdict(lambda: collections.defaultdict(float))
nh = {}
plen_size = collections.defaultdict(list)
for line in rows[1:]:
    f = line.split()
    if len(f) < len(hdr) or f[ix["Predator.no"]] != "19":
        continue
    key = (int(f[0]), int(f[1]), int(f[ix["Predator.length"]]))
    prey = int(f[ix["Prey.no"]])
    grp[key][prey] += float(f[ix["stomcon"]])
    grp[key][("hat", prey)] += float(f[ix["stomcon.hat"]])
    nh[key] = (float(f[ix["N.haul"]]), float(f[ix["N.samples"]]))
    if prey == 18:
        plen_size[key].append((float(f[ix["Prey.length.mean"]]), float(f[ix["stomcon"]])))
cols = [0, 17, 18, 21, 22, 24]
print("year q predlen(mm) n_haul n_samples | obs % by weight: " + " ".join(PREY[c] for c in cols) + " | hat haddock | haddock prey mean length (mm, weight-weighted)")
tot = collections.defaultdict(lambda: collections.defaultdict(float))
for key in sorted(grp):
    g = grp[key]
    s = sum(g[c] for c in cols)
    pl = plen_size.get(key, [])
    w = sum(x[1] for x in pl)
    ml = sum(x[0] * x[1] for x in pl) / w if w > 0 else float("nan")
    print(key[0], key[1], key[2], "%.0f %.0f |" % nh[key], " ".join("%5.1f" % (100 * g[c] / s) for c in cols),
          "| %5.1f | %.0f" % (100 * g[("hat", 18)], ml))
    # aggregates weighted by number of hauls
    for agg in (("year", key[0]), ("len", "<400" if key[2] < 400 else ("400-599" if key[2] < 600 else ">=600")), ("all", "all")):
        for c in cols:
            tot[agg][c] += nh[key][0] * g[c] / s
        tot[agg]["w"] += nh[key][0]
print("\nAggregates weighted by N.haul (obs % by weight):")
for agg in sorted(tot, key=str):
    t = tot[agg]
    print(agg, "hauls %.0f" % t["w"], " ".join("%s %.1f" % (PREY[c], 100 * t[c] / t["w"]) for c in cols))

# Quarter-3 haddock share by year (0-group haddock season), haul-weighted, with duplicated
# length classes removed (1986/1987 rows for 300/350 and 500/600 mm are identical copies).
print("\nQ3 haddock share by year, duplicates removed (haul-weighted):")
for yr in (1981, 1986, 1987, 1991):
    seen = set(); num = den = 0.0
    for key in sorted(grp):
        if key[0] != yr or key[1] != 3:
            continue
        g = grp[key]; s = sum(g[c] for c in cols)
        sig = tuple(round(g[c] / s, 4) for c in cols)
        if sig in seen:
            continue
        seen.add(sig)
        num += nh[key][0] * g[18] / s; den += nh[key][0]
    print(yr, "hauls %.0f haddock %.1f %%" % (den, 100 * num / den))

# Model fit vs observation for haddock, haul-weighted by year (duplicates removed).
print("\nHaddock share, observed vs SMS fit (stomcon.hat), haul-weighted, duplicates removed:")
for yr in (1981, 1986, 1987, 1991):
    seen = set(); o = h = den = 0.0
    for key in sorted(grp):
        if key[0] != yr:
            continue
        g = grp[key]; s = sum(g[c] for c in cols)
        sig = (key[1],) + tuple(round(g[c] / s, 4) for c in cols)
        if sig in seen:
            continue
        seen.add(sig)
        o += nh[key][0] * g[18] / s; h += nh[key][0] * g[("hat", 18)]; den += nh[key][0]
    print(yr, "hauls %.0f observed %.1f %% fitted %.1f %%" % (den, 100 * o / den, 100 * h / den))
