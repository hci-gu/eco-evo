"""Deep light in the box from VOTO glider radiometry: dark floor, slab K, deepest usable depth, daytime light.

Usage: python3 -I voto_k.py > ../tables/voto_k_stdout.txt   (also writes ../tables/voto_k.json)
Input: ../data/voto/<dataset>_bin5.csv.gz (voto_download.py), ../data/voto/meta.json.

Method
- Each ERDDAP profile_index is one up- or down-cast; values are 5 m bin means (bin k = 5k..5k+5 m, verified
  against mean pressure: bin 19 -> 98.4 dbar). z = 5k + 2.5.
- Sun elevation from the cast's mean time and position (NOAA low-precision formulae).
- Dark floor: per mission and depth bin, the median of night casts (sun < -12 deg); noise = 1.4826 x MAD.
  A daytime bin is usable when its dark-corrected value E = raw - dark is >= 3 x dark AND >= 10 noise (PAR)
  or >= 10 noise (OCR channels, whose dark is near zero for 532 nm).
- Slab K per cast: least-squares slope of ln(E) on z over the bins inside the slab, all usable, >= 3 bins.
  Daytime casts only (sun > 5 deg). Statistics: median and p25-p75 over casts, per mission and pooled.
- Daytime light at depth: median over all casts with sun > 0 of the dark-corrected PAR at each depth bin,
  where the bin is usable in >= 50 % of those casts; values below the floor are not reported.
"""
import csv
import gzip
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "voto")
TAB = os.path.join(HERE, "..", "tables")
SLABS = [(20, 35), (35, 50), (50, 65), (65, 80), (80, 95), (95, 110), (110, 125), (125, 140), (140, 155), (155, 170)]
CH_PAR = "downwelling_PAR"
OCR = ("down_irradiance_380", "down_irradiance_490", "down_irradiance_532")


def sun_elev(t, lat, lon):
    """t: numpy datetime64[s]; returns elevation in degrees (NOAA approximation, ~0.5 deg)."""
    doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(int) + 1
    hour = (t - t.astype("datetime64[D]")).astype(int) / 3600.0
    g = 2 * math.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    phi = np.radians(lat)
    s = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def load(ds):
    rows = list(csv.reader(gzip.open(os.path.join(DATA, f"{ds}_bin5.csv.gz"), "rt")))
    head = rows[0]
    rows = rows[2:]
    col = {h: i for i, h in enumerate(head)}
    casts = defaultdict(dict)
    for r in rows:
        k = int(float(r[col["depth"]]))
        if k < 0:
            continue
        p = int(r[col["profile_index"]])
        c = casts[p]
        c.setdefault("bins", {})[k] = {h: (float(r[i]) if h not in ("time",) and r[i] != "NaN" else
                                           (r[i] if h == "time" else np.nan)) for h, i in col.items()}
    out = []
    for p, c in casts.items():
        b = c["bins"]
        ks = sorted(b)
        times = np.array([np.datetime64(b[k]["time"].replace("Z", "")) for k in ks], dtype="datetime64[s]")
        tmean = times.min() + (times.max() - times.min()) / 2
        lat = np.nanmean([b[k]["latitude"] for k in ks])
        lon = np.nanmean([b[k]["longitude"] for k in ks])
        out.append(dict(profile=p, bins=b, kmax=max(ks), t=tmean, lat=lat, lon=lon,
                        sun=float(sun_elev(np.array([tmean]), lat, lon)[0]),
                        dirn=np.nanmedian([b[k].get("profile_direction", np.nan) for k in ks])))
    return out, [h for h in head]


def darkfloor(casts, ch, kmax=60):
    """Night casts (sun < -12 deg); if fewer than 20 (June), sun < -6 deg for bins >= 50 m
    (daylight there is then < 1e-4 of the floor), with the 50-55 m dark copied to shallower bins."""
    night = [c for c in casts if c["sun"] < -12]
    kmin = 0
    if len(night) < 20:
        night = [c for c in casts if c["sun"] < -6]
        kmin = 10
    d, n = {}, {}
    for k in range(kmin, kmax):
        v = np.array([c["bins"][k][ch] for c in night if k in c["bins"] and np.isfinite(c["bins"][k][ch])])
        if len(v) >= 5:
            med = np.median(v)
            d[k], n[k] = med, 1.4826 * np.median(np.abs(v - med))
    if kmin and kmin in d:
        for k in range(kmin):
            d[k], n[k] = d[kmin], n[kmin]
    return d, n, len(night)


BED = {}


def bed_of(c):
    if not BED:
        f = os.path.join(HERE, "..", "data", "emodnet_bed.json")
        if os.path.exists(f):
            BED.update(json.load(open(f)))
    if not np.isfinite(c["lat"]):
        return np.nan
    v = BED.get(f"{round(c['lat'], 2):.2f},{round(c['lon'], 2):.2f}")
    return float(v) if v is not None else np.nan


def bed_class(b):
    if not np.isfinite(b):
        return "?"
    return "<100" if b < 100 else "100-150" if b < 150 else "150-250" if b < 250 else ">=250"


def corrected(c, ch, dark, noise, is_par):
    """Return {k: E} of usable dark-corrected bins for one cast."""
    out = {}
    for k, row in c["bins"].items():
        v = row.get(ch, np.nan)
        if not np.isfinite(v) or k not in dark:
            continue
        e = v - dark[k]
        ok = e >= 10 * max(noise[k], 1e-12)
        if is_par:
            ok = ok and e >= 3 * dark[k]
        if ok:
            out[k] = e
    return out


def slab_k(E, lo, hi):
    ks = [k for k in E if lo <= 5 * k + 2.5 < hi]
    need = [k for k in range(int(lo // 5), int(math.ceil(hi / 5))) if lo <= 5 * k + 2.5 < hi]
    if len(ks) < 3 or len(ks) < len(need):
        return None
    z = np.array([5 * k + 2.5 for k in ks])
    y = np.log([E[k] for k in ks])
    return float(-np.polyfit(z, y, 1)[0])


def q(v, ps=(25, 50, 75)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)])
    return (len(v),) + tuple(float(np.percentile(v, p)) for p in ps) if len(v) else (0,) + (np.nan,) * len(ps)


def main():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    res = dict(missions={}, pooled={})
    pooled = defaultdict(list)
    pooled_ocr = defaultdict(list)
    props = defaultdict(list)
    daymed = defaultdict(lambda: defaultdict(list))
    deepest = defaultdict(list)
    updown = defaultdict(list)
    print("VOTO glider light in the box: per mission (dark floor from night casts, K per daytime cast, sun > 5 deg)")
    for ds, m in meta.items():
        irr = m["attrs"].get("irradiance", "")
        model = "MPE-PAR" if "MPE" in irr else "OCR504"
        ser = re.search(r"'serial': '([^']+)'", irr)
        ser = ser.group(1) if ser else "?"
        casts, head = load(ds)
        if not casts:
            continue
        t0 = min(c["t"] for c in casts)
        t1 = max(c["t"] for c in casts)
        month = int(str(t0 + (t1 - t0) / 2)[5:7])
        dark, noise, nn = darkfloor(casts, CH_PAR)
        day = [c for c in casts if c["sun"] > 5]
        allday = [c for c in casts if c["sun"] > 0]
        ks = defaultdict(list)
        for c in day:
            E = corrected(c, CH_PAR, dark, noise, model == "MPE-PAR")
            for lo, hi in SLABS:
                kk = slab_k(E, lo, hi)
                if kk is not None:
                    ks[(lo, hi)].append(kk)
                    pooled[(model, lo, hi)].append(kk)
                    pooled[(model, month, lo, hi)].append(kk)
                    updown[(model, lo, hi, "down" if c["dirn"] > 0 else "up")].append(kk)
                    bc = bed_class(bed_of(c))
                    pooled[(model, "serial", ser, lo, hi)].append(kk)
                    pooled[(model, "serialbed", ser, bc, lo, hi)].append(kk)
                    pooled[(model, "bed", bc, lo, hi)].append(kk)
                    bd = bed_of(c)
                    if np.isfinite(bd) and lo >= 50:
                        hab = bd - (lo + hi) / 2
                        hc = "<25" if hab < 25 else "25-50" if hab < 50 else "50-100" if hab < 100 else ">=100"
                        bbv = np.nanmean([c["bins"][k].get("particulate_backscatter", c["bins"][k].get("backscatter", np.nan))
                                          for k in c["bins"] if lo <= 5 * k + 2.5 < hi])
                        pooled[(model, "hab", hc, lo, hi)].append(kk)
                        pooled[(model, "hab", hc, "all")].append(kk)
                        pooled[(model, "habbb", hc, "all")].append(bbv)
                    pooled[(model, "bedseason", bc, "Apr-Sep" if 4 <= month <= 9 else "Oct-Mar", lo, hi)].append(kk)
                    if lo >= 65:
                        b = [c["bins"][k] for k in c["bins"] if lo <= 5 * k + 2.5 < hi]
                        props[(model, lo, hi)].append((kk, np.nanmean([x.get("chlorophyll", np.nan) for x in b]),
                                                       np.nanmean([x.get("salinity", np.nan) for x in b]),
                                                       np.nanmean([x.get("particulate_backscatter", x.get("backscatter", np.nan)) for x in b]),
                                                       c["kmax"] * 5 + 5))
            full = [slab_k(E, lo, hi) for lo, hi in SLABS[3:8]]
            if all(x is not None for x in full):
                for (lo, hi), kk in zip(SLABS[3:8], full):
                    pooled[(model, "sub140", lo, hi)].append(kk)
            if E and c["kmax"] * 5 >= 150:
                deepest[(model, month)].append(5 * max(E) + 2.5)
        for c in allday:
            E = corrected(c, CH_PAR, dark, noise, model == "MPE-PAR")
            bc = bed_class(bed_of(c))
            for k in c["bins"]:
                if k in dark:
                    daymed[(model, month)][k].append(E.get(k, np.nan))
                    daymed[(model, "bed " + bc)][k].append(E.get(k, np.nan))
        d100 = dark.get(19, np.nan)
        d150 = dark.get(29, np.nan)
        print(f"\n{ds} [{model} s/n {ser}] {str(t0)[:10]}..{str(t1)[:10]} casts {len(casts)}, night {nn}, day>5deg {len(day)};"
              f" dark @100 m {d100:.3g} (noise {noise.get(19, np.nan):.2g}), @150 m {d150:.3g}  (instrument units)")
        line = []
        for lo, hi in SLABS:
            n, a, b, cc = q(ks[(lo, hi)])
            if n:
                line.append(f"{lo}-{hi}: {b:.3f} [{a:.3f}-{cc:.3f}] n{n}")
        print("   K_PAR " + "; ".join(line))
        mres = dict(model=model, serial=ser, month=month, start=str(t0), end=str(t1), dark=dark,
                    K={f"{lo}-{hi}": q(ks[(lo, hi)]) for lo, hi in SLABS})
        if model == "OCR504":
            for ch in OCR:
                dch, nch, _ = darkfloor(casts, ch)
                kc = defaultdict(list)
                for c in day:
                    E = corrected(c, ch, dch, nch, False)
                    for lo, hi in SLABS:
                        kk = slab_k(E, lo, hi)
                        if kk is not None:
                            kc[(lo, hi)].append(kk)
                            pooled_ocr[(ch, lo, hi)].append(kk)
                            pooled_ocr[(ch, month, lo, hi)].append(kk)
                            pooled_ocr[(ch, "bed", bed_class(bed_of(c)), lo, hi)].append(kk)
                line = []
                for lo, hi in SLABS:
                    n, a, b, cc = q(kc[(lo, hi)])
                    if n:
                        line.append(f"{lo}-{hi}: {b:.3f} n{n}")
                print(f"   K_{ch[-3:]} (dark @100 m {dch.get(19, np.nan):.3g}, noise {nch.get(19, np.nan):.2g}) " + "; ".join(line))
                mres["K_" + ch[-3:]] = {f"{lo}-{hi}": q(kc[(lo, hi)]) for lo, hi in SLABS}
        res["missions"][ds] = mres

    for model in ("MPE-PAR", "OCR504"):
        print(f"\n=== Pooled K_PAR, {model} (median [p25-p75] n casts) ===")
        for lo, hi in SLABS:
            n, a, b, c = q(pooled[(model, lo, hi)])
            if n:
                print(f"   {lo:3d}-{hi:3d} m: {b:.3f} [{a:.3f}-{c:.3f}] n{n}")
                res["pooled"][f"{model} {lo}-{hi}"] = (n, a, b, c)
        print("   same casts usable through 140 m (selection check):")
        for lo, hi in SLABS[3:8]:
            n, a, b, c = q(pooled[(model, "sub140", lo, hi)])
            if n:
                print(f"     {lo:3d}-{hi:3d} m: {b:.3f} [{a:.3f}-{c:.3f}] n{n}")
                res["pooled"][f"{model} sub140 {lo}-{hi}"] = (n, a, b, c)
        print("   by sensor serial (median K, n >= 5), all beds / beds 150-250 m:")
        for ser in sorted({k[2] for k in pooled if len(k) == 5 and k[0] == model and k[1] == "serial"}):
            for lab, key in (("all", lambda lo, hi: (model, "serial", ser, lo, hi)),
                             ("150-250", lambda lo, hi: (model, "serialbed", ser, "150-250", lo, hi))):
                line = []
                for lo, hi in SLABS:
                    n, a, b, c = q(pooled[key(lo, hi)])
                    if n >= 5:
                        line.append(f"{lo}-{hi} {b:.3f} n{n}")
                if line:
                    print(f"     s/n {ser} {lab:>7s}: " + "; ".join(line))
        print("   by EMODnet bed depth class under the cast (median [p25-p75] n):")
        for bc in ("<100", "100-150", "150-250", ">=250", "?"):
            line = []
            for lo, hi in SLABS:
                n, a, b, c = q(pooled[(model, "bed", bc, lo, hi)])
                if n >= 5:
                    line.append(f"{lo}-{hi} {b:.3f} [{a:.3f}-{c:.3f}] n{n}")
                    res["pooled"][f"{model} bed {bc} {lo}-{hi}"] = (n, a, b, c)
            if line:
                print(f"     bed {bc:>7s}: " + "; ".join(line))
        print("   by height of the slab mid above the EMODnet bed (median [p25-p75] n; slabs >= 50 m):")
        for hc in ("<25", "25-50", "50-100", ">=100"):
            n, a, b, c = q(pooled[(model, "hab", hc, "all")])
            nb, ab, bb_, cb = q(pooled[(model, "habbb", hc, "all")])
            line = []
            for lo, hi in SLABS[2:]:
                n2, a2, b2, c2 = q(pooled[(model, "hab", hc, lo, hi)])
                if n2 >= 10:
                    line.append(f"{lo}-{hi} {b2:.3f} n{n2}")
            if n:
                print(f"     {hc:>6s} m above bed: all slabs {b:.3f} [{a:.3f}-{c:.3f}] n{n}; backscatter median {bb_:.2e}"
                      f" (instrument units, mixed sensors) | " + "; ".join(line))
                res["pooled"][f"{model} hab {hc}"] = (n, a, b, c)
        print("   by bed class and half-year (median K, n >= 5):")
        for bc in ("100-150", "150-250", ">=250"):
            for hy in ("Apr-Sep", "Oct-Mar"):
                line = []
                for lo, hi in SLABS[2:]:
                    n, a, b, c = q(pooled[(model, "bedseason", bc, hy, lo, hi)])
                    if n >= 5:
                        line.append(f"{lo}-{hi} {b:.3f} n{n}")
                        res["pooled"][f"{model} bed {bc} {hy} {lo}-{hi}"] = (n, a, b, c)
                if line:
                    print(f"     bed {bc:>7s} {hy}: " + "; ".join(line))
        print(f"   by month (median K, n):")
        for mo in range(1, 13):
            line = []
            for lo, hi in SLABS:
                n, a, b, c = q(pooled[(model, mo, lo, hi)])
                if n >= 5:
                    line.append(f"{lo}-{hi} {b:.3f} n{n}")
            if line:
                print(f"     month {mo:2d}: " + "; ".join(line))
        print("   up vs down casts (median K):")
        for lo, hi in SLABS:
            u = q(updown[(model, lo, hi, "up")])
            d = q(updown[(model, lo, hi, "down")])
            if u[0] or d[0]:
                print(f"     {lo}-{hi}: up {u[2]:.3f} n{u[0]}, down {d[2]:.3f} n{d[0]}")
    print("\n=== Pooled OCR-504 spectral K (median [p25-p75] n casts) ===")
    for ch in OCR:
        for lo, hi in SLABS:
            n, a, b, c = q(pooled_ocr[(ch, lo, hi)])
            if n:
                print(f"   {ch[-3:]} nm {lo:3d}-{hi:3d} m: {b:.3f} [{a:.3f}-{c:.3f}] n{n}")
                res["pooled"][f"Ed{ch[-3:]} {lo}-{hi}"] = (n, a, b, c)
    print("   by EMODnet bed class (median K, n >= 5):")
    for ch in ("down_irradiance_490", "down_irradiance_532"):
        for bc in ("<100", "100-150", "150-250", ">=250"):
            line = []
            for lo, hi in SLABS:
                n, a, b, c = q(pooled_ocr[(ch, "bed", bc, lo, hi)])
                if n >= 5:
                    line.append(f"{lo}-{hi} {b:.3f} n{n}")
                    res["pooled"][f"Ed{ch[-3:]} bed {bc} {lo}-{hi}"] = (n, a, b, c)
            if line:
                print(f"     {ch[-3:]} bed {bc:>7s}: " + "; ".join(line))
    print("   by month (median K, n >= 5):")
    for ch in ("down_irradiance_490", "down_irradiance_532"):
        for mo in range(1, 13):
            line = []
            for lo, hi in SLABS:
                n, a, b, c = q(pooled_ocr[(ch, mo, lo, hi)])
                if n >= 5:
                    line.append(f"{lo}-{hi} {b:.3f} n{n}")
            if line:
                print(f"     {ch[-3:]} month {mo:2d}: " + "; ".join(line))

    print("\n=== K_PAR (MPE-PAR) at >= 65 m against water properties in the slab (Spearman rho over casts) ===")
    for lo, hi in SLABS:
        v = props[("MPE-PAR", lo, hi)]
        if len(v) < 20:
            continue
        a = np.array(v, dtype=float)
        def rho(x, y):
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                return np.nan
            rx = np.argsort(np.argsort(x[ok]))
            ry = np.argsort(np.argsort(y[ok]))
            return float(np.corrcoef(rx, ry)[0, 1])
        print(f"   {lo}-{hi} m n{len(a)}: chl median {np.nanmedian(a[:, 1]):.3f} mg/m3 (rho {rho(a[:, 0], a[:, 1]):+.2f}), "
              f"S {np.nanmedian(a[:, 2]):.2f} (rho {rho(a[:, 0], a[:, 2]):+.2f}), bb {np.nanmedian(a[:, 3]):.2e} "
              f"(rho {rho(a[:, 0], a[:, 3]):+.2f}), dive depth {np.nanmedian(a[:, 4]):.0f} m (rho {rho(a[:, 0], a[:, 4]):+.2f})")

    print("\n=== Deepest usable PAR bin, daytime casts (sun > 5 deg) that dived >= 150 m: median [p10-p90] ===")
    for (model, mo), v in sorted(deepest.items()):
        n, a, b, c = q(v, (10, 50, 90))
        print(f"   {model} month {mo:2d}: {b:.0f} m [{a:.0f}-{c:.0f}] n{n}")

    print("\n=== Daytime (sun > 0) median of measured PAR, umol m-2 s-1 (dark-corrected; '<floor' = usable in < 50 % of casts) ===")
    zs = (50, 65, 80, 100, 120, 140, 150, 160)
    print("   model         month " + " ".join(f"{z:>9d}m" for z in zs))
    res["daymed"] = {}
    for (model, mo) in sorted(daymed, key=lambda x: (x[0], str(x[1]).zfill(2))):
        line = []
        for z in zs:
            k = int(z // 5)
            v = np.array(daymed[(model, mo)].get(k, []), dtype=float)
            if len(v) < 10:
                line.append("        -")
                continue
            frac = np.isfinite(v).mean()
            if frac < 0.5:
                line.append("   <floor")
                continue
            vv = np.where(np.isfinite(v), v, 0.0)
            scale = 1e4 if model == "MPE-PAR" else 1.0
            med = float(np.median(vv)) * scale
            line.append(f"{med:10.3g}")
            res["daymed"][f"{model} {mo} {z}"] = med
        print(f"   {model:7s} {str(mo):>11s} " + " ".join(line))
    json.dump(res, open(os.path.join(TAB, "voto_k.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
