"""HERAS (RV Dana, July) herring backscatter near the bed, on the grid's footing (reconcile step A).

Usage: python3 -I reconcile_heras_profile.py <ices_zip_dir> <biotic_derived_dir> <depth_npz> <out_dir>
  ices_zip_dir      research_notes/Ekolodsdata planktivorer/data/ices_zip (untrusted, parsed only)
  biotic_derived_dir research_notes/Ekolodsdata planktivorer/data/derived (haul tables)
  depth_npz         maps/depth.npz
  out_dir           tables/ (writes reconcile_heras_by_class.csv)

What differs from Vitlingens fiskbyten/scripts/step2_heras_bottom_share.py [D]:
  1. Vertical shares use only the cruises with 1-m channels (2015-2020). The 2010-14 cruises have
     channels up to 50 m wide and 2021+ one layer; both are padded to 154 m, so step 2 put every
     2010-14 log, whatever its bed depth, into ">=150 (bed not seen)" with the 104-154 m channel as
     its "bottom 10 m".
  2. Bed depth: the deepest 1-m channel's lower bound when the bed is inside the integration (< 150 m;
     it matches the grid cell's EMODnet mean depth within ~1 m, printed below, and the lowest 1-2
     channels hold ~0 backscatter: they are the 1 m backstep + dead zone); otherwise the grid cell's
     EMODnet mean depth (maps/depth.npz) for logs inside the grid.
  3. Footing: logs inside the model grid (SWEREF 99 TM E 211-271 km, N 6414-6474 km) as well as the
     larger download bbox (57.7-58.4 N, 9.9-11.3 E) step 2 called "box".
  4. Layers measured from the bed: last 10 m (0-10 m above the bed, of which the lowest ~2 m are
     blind), last 4 m (the GOV opening, ~4.2 m), 10-20 m, and a per-metre profile.
  5. MIX = clupeids + other fish (2015-20 has MIX only). The herring share of the NEAR-BED backscatter
     is estimated from Dana EXPO bottom hauls (catch weight -> numbers -> sigma_bs) by bed depth.
  6. Column NASC per grid log (all cruises 2010-26, any channel width) by cell depth class, for the
     biomass weighting, and t/km2 per NASC unit from the herring lengths in the Dana hauls.
"""
import collections
import csv
import io
import math
import os
import sys
import zipfile
from datetime import datetime

import numpy as np

sys.path.insert(0, "/home/simon/eco-evo/tools")
from make_fishing_maps import CELL, E0, GRID, N0, wgs84_to_sweref99tm  # noqa: E402  (repo code)

zdir, ddir, depth_npz, out = sys.argv[1:5]
Z = np.load(depth_npz)["depth"].astype(float)
BBOX = (57.7, 58.4, 9.9, 11.3)
CLASSES = [(0, 80), (80, 110), (110, 130), (130, 150), (150, 200), (200, 300), (300, 1e4)]
LAB = ["<80", "80-110", "110-130", "130-150", "150-200", "200-300", ">=300"]
NMI2_KM2 = 1.852 ** 2


def cell_of(lat, lon):
    e, n = wgs84_to_sweref99tm(np.array(lon), np.array(lat))
    col = int(math.floor((float(e) - E0) / CELL))
    row = GRID - 1 - int(math.floor((float(n) - N0) / CELL))
    return (row, col) if (0 <= col < GRID and 0 <= row < GRID) else None


def elev(lat, lon, t):
    n = t.timetuple().tm_yday
    hrs = t.hour + t.minute / 60 + lon / 15.0
    dec = math.radians(23.44) * math.sin(2 * math.pi * (284 + n) / 365.0)
    ha = math.radians(15 * (hrs - 12))
    p = math.radians(lat)
    s = math.sin(p) * math.sin(dec) + math.cos(p) * math.cos(dec) * math.cos(ha)
    return math.degrees(math.asin(max(-1, min(1, s))))


def cls_of(z):
    return next(i for i, (a, b) in enumerate(CLASSES) if a <= z < b)


# ---------------------------------------------------------------- parse the zips
logs = []   # dict per log
for fn in sorted(os.listdir(zdir)):
    if not fn.endswith(".zip"):
        continue
    with zipfile.ZipFile(os.path.join(zdir, fn)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            continue
        t = z.read(names[0]).decode("utf-8-sig", errors="replace")
    hdr, cruise = {}, None
    raw = collections.defaultdict(list)
    for row in csv.reader(io.StringIO(t)):
        if len(row) < 2:
            continue
        if row[1] == "Header":
            hdr[row[0]] = row[2:]
            continue
        if row[0] == "Cruise":
            cruise = dict(zip(hdr["Cruise"], row[2:]))
        if row[0] != "Data":
            continue
        r = dict(zip(hdr["Data"], row[2:]))
        try:
            lat, lon = float(r["LogLatitude"]), float(r["LogLongitude"])
            up, lo = float(r["SampleChannelDepthUpper"]), float(r["SampleChannelDepthLower"])
            v = float(r["DataValue"] or 0)
        except (ValueError, KeyError):
            continue
        if r.get("LogValidity") not in ("V", ""):
            continue
        if not (BBOX[0] <= lat <= BBOX[1] and BBOX[2] <= lon <= BBOX[3]):
            continue
        raw[(r["LogTime"], lat, lon)].append((up, lo, r.get("DataSaCategory", ""), v))
    if cruise is None or cruise.get("CruiseSurvey") != "HERAS":
        continue
    cid = cruise.get("CruiseLocalID", fn)
    for (lt, lat, lon), ch in raw.items():
        try:
            tt = datetime.strptime(lt[:16].replace("T", " "), "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        width = max(c[1] - c[0] for c in ch)
        deepest = max(c[1] for c in ch)
        # fish backscatter: MIX (2010-20) or HER+CLU+SPR+MIX (2021+); vertical work uses 2015-20 MIX only
        prof = collections.Counter()
        for up, lo, cat, v in ch:
            if cat in ("MIX", "HER", "CLU", "SPR") and v > 0:
                prof[(up, lo)] += v
        cell = cell_of(lat, lon)
        logs.append(dict(cruise=cid, year=tt.year, lat=lat, lon=lon, e=elev(lat, lon, tt), width=width,
                         deepest=deepest, prof=prof, cell=cell, zc=(Z[cell] if cell else None)))

print(f"HERAS logs in the bbox: {len(logs)}; inside the grid: {sum(1 for L in logs if L['cell'])}")
fine = [L for L in logs if L["width"] <= 1.01]
print(f"1-m channel logs (2015-20): {len(fine)}; cruises {sorted(set(L['cruise'] for L in fine))}")

# bed depth and class for the 1-m logs
for L in fine:
    L["seen"] = L["deepest"] < 150
    if L["seen"]:
        L["bed"] = L["deepest"]
    elif L["zc"] is not None:
        L["bed"] = max(L["zc"], 155.0)
    else:
        L["bed"] = None          # deeper than ~155 m, unknown
    L["k"] = cls_of(L["bed"]) if L["bed"] is not None else None

# check of the bed proxy against the grid cell depth (1-m logs, bed seen, inside the grid)
chk = np.array([(L["zc"], L["bed"]) for L in fine if L["seen"] and L["cell"]])
d = chk[:, 1] - chk[:, 0]
print(f"bed proxy (deepest channel) - cell mean depth, seen logs in grid (n {len(d)}): median {np.median(d):+.1f} m, "
      f"p10 {np.percentile(d, 10):+.1f}, p90 {np.percentile(d, 90):+.1f}")
nseen_cell = collections.Counter((cls_of(L["zc"]), L["seen"]) for L in fine if L["cell"])
print("1-m logs in grid by CELL depth class: seen / not seen = " + ", ".join(
    f"{LAB[k]} {nseen_cell[(k, True)]}/{nseen_cell[(k, False)]}" for k in range(len(LAB))))


def layer_stats(sel):
    """Pooled NASC shares (sum over logs) for 1-m logs with the bed seen."""
    tot = b10 = b8 = b3 = b1 = b4 = b20 = 0.0
    prof = np.zeros(30)        # NASC per 1-m bin of height above the deepest channel (0 = lowest seen m)
    colm = 0.0                 # column metres integrated
    for L in sel:
        dp = L["deepest"]
        for (up, lo), v in L["prof"].items():
            tot += v
            h = dp - lo        # metres between this channel's lower bound and the deepest channel's
            if h < 10:
                b10 += v
            if h < 8:
                b8 += v
            if h < 3:
                b3 += v
            if h < 4:
                b4 += v
            if 10 <= h < 20:
                b20 += v
            if h < 1:
                b1 += v
            if h < 30:
                prof[int(h)] += v
        colm += dp - 7.0
    layer_stats.extra = (b4, b20)
    return tot, b10, b8, b3, b1, prof, colm


def run(region, sun):
    rows = []
    sel_all = [L for L in fine if L["e"] > 6 if sun == "day>6"] if sun == "day>6" else \
        [L for L in fine if 0 < L["e"] <= 6]
    if region == "grid":
        sel_all = [L for L in sel_all if L["cell"]]
    print(f"\n--- {region}, sun {sun}: 1-m logs {len(sel_all)} ---")
    print("class    logs seen | NASC/log | share last10 / last8 / last3 / last1 m (pooled)  | per-cruise last10 min-max "
          "| last-1m density / column mean per m | deep logs: share 144-154 / 134-154 / 104-154 m")
    for k in range(len(LAB)):
        sel = [L for L in sel_all if L["k"] == k]
        if not sel:
            continue
        seen = [L for L in sel if L["seen"]]
        nasc = sum(sum(L["prof"].values()) for L in sel) / len(sel)
        line = f"{LAB[k]:8s} {len(sel):4d} {len(seen):4d} | {nasc:8.0f} | "
        rec = dict(region=region, sun=sun, cls=LAB[k], logs=len(sel), seen=len(seen), nasc_per_log=nasc)
        if len(seen) >= 3:
            tot, b10, b8, b3, b1, prof, colm = layer_stats(seen)
            b4, b20 = layer_stats.extra
            per = []
            for c in sorted(set(L["cruise"] for L in seen)):
                tt, bb = layer_stats([L for L in seen if L["cruise"] == c])[:2]
                if tt > 0:
                    per.append(bb / tt)
            rel = (b1 / len(seen)) / (tot / colm) if tot > 0 else float("nan")
            line += (f"{b10 / tot:.3f} / {b8 / tot:.3f} / {b3 / tot:.3f} / {b1 / tot:.3f} | "
                     f"{min(per):.2f}-{max(per):.2f} (n {len(per)}) | {rel:5.1f} | last4 {b4 / tot:.3f}, 10-20 m {b20 / tot:.3f}")
            rec.update(last4=b4 / tot, h10_20=b20 / tot, last10=b10 / tot, last8=b8 / tot, last3=b3 / tot, last1=b1 / tot,
                       last10_min=min(per), last10_max=max(per), rel_last1=rel,
                       prof=" ".join(f"{p / tot:.4f}" for p in prof[:20]))
        deep = [L for L in sel if not L["seen"]]
        if deep:
            tot = sum(sum(L["prof"].values()) for L in deep)
            s144 = sum(v for L in deep for (u, l), v in L["prof"].items() if u >= 144)
            s134 = sum(v for L in deep for (u, l), v in L["prof"].items() if u >= 134)
            s104 = sum(v for L in deep for (u, l), v in L["prof"].items() if u >= 104)
            if tot > 0:
                line += f" | deep n {len(deep)}: {s144 / tot:.3f} / {s134 / tot:.3f} / {s104 / tot:.3f}"
                rec.update(deep_logs=len(deep), deep_144_154=s144 / tot, deep_134_154=s134 / tot,
                           deep_104_154=s104 / tot)
        print(line)
        rows.append(rec)
    # profile above the bed, pooled over bed-seen classes
    for k in range(4):
        r = next((x for x in rows if x["cls"] == LAB[k] and "prof" in x), None)
        if r:
            p = [float(v) for v in r["prof"].split()]
            print(f"  profile {LAB[k]:8s} share per 1-m bin above the lowest seen metre (0..9): "
                  + " ".join(f"{v:.3f}" for v in p[:10]) + f" | 10-19 m sum {sum(p[10:20]):.3f}")
    return rows


rows = []
# per-cruise sums (grid, day > 6 deg, bed seen) for the bootstrap in reconcile_combine.py
for k in range(4):
    for c in sorted(set(L["cruise"] for L in fine)):
        sel = [L for L in fine if L["cell"] and L["e"] > 6 and L["k"] == k and L["seen"] and L["cruise"] == c]
        if not sel:
            continue
        tot, b10, b8, b3, b1, prof, colm = layer_stats(sel)
        b4, b20 = layer_stats.extra
        rows.append(dict(region="grid_cruise", sun="day>6", cls=LAB[k], cruise=c, logs=len(sel), sum_tot=tot,
                         sum_last10=b10, sum_last4=b4, sum_bin2=prof[2], sum_bin3=prof[3], sum_bin0=prof[0],
                         sum_bin1=prof[1]))
for region in ("bbox", "grid"):
    for sun in ("day>6", "low0-6"):
        rows += run(region, sun)

# ---------------------------------------------------------------- column NASC per grid log, all cruises
print("\n--- column fish NASC per log inside the grid by CELL depth class, day > 6 deg, all cruises 2010-26 ---")
print("class   logs  mean NASC  (per-cruise mean of means)  cell share of grid")
zz = Z.ravel()
area = [np.mean((zz >= a) & (zz < b)) for a, b in CLASSES]
for k in range(len(LAB)):
    sel = [L for L in logs if L["cell"] and L["e"] > 6 and cls_of(L["zc"]) == k]
    if not sel:
        print(f"{LAB[k]:8s} {0:4d}")
        continue
    m = np.mean([sum(L["prof"].values()) for L in sel])
    per = collections.defaultdict(list)
    for L in sel:
        per[L["cruise"]].append(sum(L["prof"].values()))
    mm = np.mean([np.mean(v) for v in per.values()])
    print(f"{LAB[k]:8s} {len(sel):4d} {m:9.0f}  ({mm:7.0f}, {len(per)} cruises)  {area[k]:.3f}")
    rows.append(dict(region="grid_allcruises", sun="day>6", cls=LAB[k], logs=len(sel), nasc_per_log=m,
                     nasc_cruise_mean=mm, cell_share=area[k]))

print("same, 2015-20 cruises only (the years of the vertical shares):")
for k in range(len(LAB)):
    sel = [L for L in logs if L["cell"] and L["e"] > 6 and cls_of(L["zc"]) == k and 2015 <= L["year"] <= 2020]
    if sel:
        m = np.mean([sum(L["prof"].values()) for L in sel])
        print(f"{LAB[k]:8s} {len(sel):4d} {m:9.0f}")
        rows.append(dict(region="grid_1520", sun="day>6", cls=LAB[k], logs=len(sel), nasc_per_log=m))

# ---------------------------------------------------------------- Dana trawl hauls: herring t/km2 per NASC
# and the herring share of near-bed backscatter in EXPO bottom hauls
CLU = {"126417": -71.2, "126425": -71.2}
GAD = {k: -67.5 for k in ["126444", "126438", "126437", "126436", "126441", "126440", "126446", "126439",
                          "126435", "126484", "126822", "126715", "126716", "127312", "126461", "126450"]}
B20 = dict(CLU, **GAD)
hauls = {}
for r in csv.DictReader(open(os.path.join(ddir, "biotic_region_hauls.tsv")), delimiter="\t"):
    if not (r["cruise"].startswith("26D") or r["cruise"].startswith("Dana")) or r["validity"] not in ("V", ""):
        continue
    try:
        lat, lon = float(r["lat"]), float(r["lon"])
    except ValueError:
        continue
    bot = float(r["bottom"]) if r["bottom"] else (float(r["max_depth"]) if r["max_depth"] else None)
    hauls[(r["cruise"], r["haul"])] = dict(gear=r["gear"], lat=lat, lon=lon, bottom=bot, W=collections.Counter())
for r in csv.DictReader(open(os.path.join(ddir, "biotic_region_catch.tsv")), delimiter="\t"):
    k = (r["cruise"], r["haul"])
    if k in hauls and r["species"] in B20:
        hauls[k]["W"][r["species"]] += float(r["weight_kg"] or 0)
ind = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(ddir, "biotic_region_individuals.tsv")), delimiter="\t"):
    if r["species"] in B20:
        try:
            L = float(r["length_mm"]) / 10
        except ValueError:
            continue
        if 2 < L < 120:
            ind[r["species"]].append((r["cruise"], r["haul"], L))
pooled = {}
for sp, lst in ind.items():
    Ls = np.array([x[2] for x in lst])
    pooled[sp] = Ls
HLW = (0.0022305830962786404, 3.3701933505454997)   # herring L-W, Fiskgrupper i rutan/tables/length_weight_all.csv


def sig(L, b20):
    return 10 ** ((20 * np.log10(L) + b20) / 10)


Lh = pooled["126417"]
w = HLW[0] * Lh ** HLW[1] / 1e6       # t
s = sig(Lh, -71.2)
t_per_nasc = (w.sum() / (4 * math.pi * s.sum())) / NMI2_KM2
print(f"\nherring in Dana hauls (bbox+margin, all years): n {len(Lh)}, mean L {Lh.mean():.1f} cm; "
      f"t/km2 per unit herring NASC = {t_per_nasc:.5f}")
rows.append(dict(region="conversion", cls="herring", t_per_nasc=t_per_nasc, n=len(Lh)))


def mean_sig_w(sp, haul_key):
    lst = [x[2] for x in ind.get(sp, []) if (x[0], x[1]) == haul_key]
    L = np.array(lst) if len(lst) >= 5 else pooled.get(sp)
    if L is None or len(L) == 0:
        return None
    wl = 0.01 * L ** 3 / 1e3                 # kg, generic fallback weight for gadoids
    if sp == "126417":
        wl = HLW[0] * L ** HLW[1] / 1e3
    return sig(L, B20[sp]).mean(), wl.mean()


print("\nDana EXPO bottom hauls (bbox + 0.3 deg margin): herring share of fish backscatter (n sigma_bs) and of the "
      "planktivore+gadoid catch weight, by bed depth")
print("bed class  hauls  herring share of backscatter (pooled / haul median)  herring share of weight")
for k in range(len(LAB)):
    acc = []
    for key, h in hauls.items():
        if h["gear"] not in ("EXP", "BOT") or h["bottom"] is None or cls_of(h["bottom"]) != k:
            continue
        if not (BBOX[0] - 0.3 <= h["lat"] <= BBOX[1] + 0.3 and BBOX[2] - 0.3 <= h["lon"] <= BBOX[3] + 0.3):
            continue
        bs = {}
        for sp, W in h["W"].items():
            st = mean_sig_w(sp, key)
            if st and W > 0:
                bs[sp] = W / st[1] * st[0]
        if bs:
            acc.append((bs.get("126417", 0.0), sum(bs.values()), h["W"].get("126417", 0.0), sum(h["W"].values())))
    if acc:
        a = np.array(acc)
        med = np.median(a[:, 0] / a[:, 1])
        print(f"{LAB[k]:9s} {len(a):5d}  {a[:, 0].sum() / a[:, 1].sum():.2f} / {med:.2f}  {a[:, 2].sum() / a[:, 3].sum():.2f}")
        rows.append(dict(region="expo", cls=LAB[k], hauls=len(a), her_bs_share=a[:, 0].sum() / a[:, 1].sum(),
                         her_bs_median=med, her_w_share=a[:, 2].sum() / a[:, 3].sum()))

keys = sorted({k for r in rows for k in r}, key=lambda x: (x not in ("region", "sun", "cls"), x))
with open(os.path.join(out, "reconcile_heras_by_class.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys)
    wr.writeheader()
    for r in rows:
        wr.writerow(r)
print(f"\nwrote {os.path.join(out, 'reconcile_heras_by_class.csv')}")
