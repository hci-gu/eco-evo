"""Maturity in and around the box from NS-IBTS CA/HL records (datras_fetch.py output).

Usage: python3 -I datras_maturity.py DATA_DIR [DATA_DIR ...]   (each holds hh.csv, hl.csv, ca.csv)

Prints, per species:
 A. CA records by quarter, box (44G0, 44G1, 45G0, 45G1) vs region, with the maturity-stage
    composition of the sexed/staged fish (immature, developing, spawning, spent, resting).
 B. Maturity ogive (share mature by length class) per quarter, region-pooled, years pooled.
 C. Mature share of the species' BIOMASS in the box per quarter: box HL numbers-at-length per
    hour (all valid hauls in the four rectangles, years pooled) x weight-at-length (log-log fit to
    CA IndWgt, region) x the region ogive of that quarter.
 D. Spawning-stage fish in the box by month (the direct evidence of spawning in the box).
 E. Weight at length of mature vs immature fish by quarter and sex (residual of the region
    log-log fit, %): a Q1-Q3 difference in mature females over immature fish = gonad (+liver) mass.

Stage mapping (DATRAS scales): old IBTS 4-stage (scale -9, codes 1-4: immature, maturing,
spawning, spent; codes 5-6 appear only in SE 2008-11 and are counted as mature, stage unknown),
M6 (61 immature, 62 developing, 63 spawning, 64 spent, 65 resting / skipped, 66 abnormal) and
SMSF (A immature, B developing, C spawning capable, D regressing / regenerating, E omitted
spawning, F abnormal). Abnormal and unstaged fish are left out.
"""
import sys

import numpy as np
import pandas as pd

BOX = {"44G0", "44G1", "45G0", "45G1"}
SPECIES = {
    126436: "cod", 126437: "haddock", 126438: "whiting", 126441: "saithe", 126444: "Norway pout",
    126439: "blue whiting", 126417: "herring", 126425: "sprat", 127023: "mackerel",
    126822: "horse mackerel", 126484: "hake", 126461: "ling", 126440: "pollack",
    126555: "anglerfish", 105923: "spurdog", 105913: "velvet belly", 127137: "long rough dab",
    127139: "dab", 127143: "plaice", 127136: "witch", 105865: "starry ray", 101170: "hagfish",
    127140: "lemon sole",
}
STAGE = {"1": "imm", "61": "imm", "A": "imm",
         "2": "dev", "62": "dev", "B": "dev",
         "3": "spawn", "63": "spawn", "C": "spawn",
         "4": "spent", "64": "spent", "D": "spent",
         "65": "rest", "E": "rest", "5": "mat?", "6": "mat?"}
KEYS = ["Year", "Quarter", "Country", "Ship", "Gear", "HaulNo", "StNo"]


def load(dirs):
    hh, hl, ca = [], [], []
    for d in dirs:
        hh.append(pd.read_csv(f"{d}/hh.csv", dtype=str))
        hl.append(pd.read_csv(f"{d}/hl.csv", dtype=str))
        ca.append(pd.read_csv(f"{d}/ca.csv", dtype=str))
    hh, hl, ca = (pd.concat(x, ignore_index=True) for x in (hh, hl, ca))
    for df in (hh, hl, ca):
        for c in df.columns:
            df[c] = df[c].str.strip()
    return hh, hl, ca


def length_cm(code, cls):
    cls = pd.to_numeric(cls, errors="coerce")
    return np.where(code.isin([".", "0"]), cls / 10.0, cls)


def main():
    hh, hl, ca = load(sys.argv[1:])
    hh = hh.drop_duplicates(KEYS)
    hh["box"] = hh.StatRec.isin(BOX)
    hh["Month"] = pd.to_numeric(hh.Month, errors="coerce")
    hh["HaulDur"] = pd.to_numeric(hh.HaulDur, errors="coerce")
    hh = hh[(hh.HaulVal == "V") & (hh.Gear == "GOV")]
    print(f"valid GOV hauls: region {len(hh)}, box {hh.box.sum()}")
    print(hh[hh.box].groupby(["Quarter", "Month"]).size().rename("box hauls").to_string())
    print(hh[hh.box].groupby("Quarter").Year.agg(["min", "max", "nunique"]).to_string())
    print(f"box haul depth m: median {pd.to_numeric(hh[hh.box].Depth, errors='coerce').median()}, "
          f"range {pd.to_numeric(hh[hh.box].Depth, errors='coerce').min()}-"
          f"{pd.to_numeric(hh[hh.box].Depth, errors='coerce').max()}")
    dep = pd.to_numeric(hh[hh.box].Depth, errors="coerce")
    print(f"box hauls < 100 m: {(dep < 100).mean():.2f} (box bed area < 100 m: 0.232, maps/depth.npz)")

    ca["L"] = length_cm(ca.LngtCode, ca.LngtClass)
    ca["n"] = pd.to_numeric(ca.CANoAtLngt, errors="coerce").clip(lower=0)
    ca["w"] = pd.to_numeric(ca.IndWgt, errors="coerce")
    ca["stage"] = ca.Maturity.map(STAGE)
    ca["sp"] = pd.to_numeric(ca.Valid_Aphia, errors="coerce").map(SPECIES)
    ca = ca.merge(hh[KEYS + ["box", "Month", "StatRec"]], on=KEYS, how="left")
    ca["box"] = ca.box.fillna(False) | ca.AreaCode.isin(BOX)
    ca = ca[ca.Gear == "GOV"]

    hl = hl.merge(hh[KEYS + ["box", "HaulDur", "DataType", "Depth", "Month"]], on=KEYS, how="inner")
    hl["L"] = length_cm(hl.LngtCode, hl.LngtClass)
    hl["N"] = pd.to_numeric(hl.HLNoAtLngt, errors="coerce") * pd.to_numeric(hl.SubFactor, errors="coerce")
    hl.loc[hl.DataType == "R", "N"] *= 60.0 / hl.loc[hl.DataType == "R", "HaulDur"]
    hl["sp"] = pd.to_numeric(hl.Valid_Aphia, errors="coerce").map(SPECIES)
    hl = hl[hl.N > 0]

    nbox = hh[hh.box].groupby("Quarter").size()
    for sp in ["haddock", "cod", "long rough dab", "dab", "plaice", "witch", "lemon sole", "starry ray",
               "hagfish", "whiting", "saithe", "hake", "ling", "pollack", "anglerfish", "spurdog",
               "velvet belly", "herring", "Norway pout", "blue whiting", "sprat", "mackerel",
               "horse mackerel"]:
        c = ca[ca.sp == sp]
        h = hl[hl.sp == sp]
        print(f"\n===== {sp}: CA records {len(c)} (box {int(c.box.sum())}), HL box rows {int(h.box.sum())}")
        if len(c) == 0 and len(h) == 0:
            continue
        # A. stage composition
        st = c[c.stage.notna()]
        if len(st):
            tab = st.groupby(["Quarter", "box", "stage"]).n.sum().unstack(fill_value=0)
            tab["N"] = tab.sum(axis=1)
            print("A. staged fish by quarter / box, numbers")
            print(tab.to_string())
        # weight-length fit (region, all quarters)
        wl = c[(c.w > 0) & (c.L > 0)]
        if len(wl) > 20:
            b, lna = np.polyfit(np.log(wl.L), np.log(wl.w), 1)
        else:
            b, lna = 3.0, np.log(0.01)
        print(f"weight-length (region CA, n {len(wl)}): W g = {np.exp(lna):.5f} L^{b:.3f}")
        # B. ogive per quarter (2 cm classes)
        st2 = st[st.stage != "mat?"].copy() if len(st) else st
        if len(st):
            st = st.copy()
            st["mat"] = st.stage != "imm"
            st["L2"] = (st.L // 2) * 2
        og1 = None
        if len(st) and (st.Quarter == "1").sum() >= 20:
            s1q = st[st.Quarter == "1"]
            og1 = s1q.groupby("L2").apply(lambda d: pd.Series({"n": d.n.sum(), "pm": (d.n * d.mat).sum() / d.n.sum()}))
        for q in sorted(hh.Quarter.unique()):
            sq = st[st.Quarter == q] if len(st) else st
            hq = h[(h.Quarter == q) & h.box]
            if len(sq) < 20 and len(hq) == 0:
                continue
            og = sq.groupby("L2").apply(lambda d: pd.Series({"n": d.n.sum(), "pm": (d.n * d.mat).sum() / d.n.sum()})) \
                if len(sq) >= 20 else None
            line = ""
            if og is not None:
                ok = og[(og.n >= 10) & (og.pm >= 0.5)]
                l50 = ok.index.min() if len(ok) else np.nan
                line = f"L50 ~{l50} cm (first 2-cm class with n >= 10 and >= 50 % mature), staged n {int(sq.n.sum())}"
            # C. mature biomass share in the box
            if len(hq) and og is not None:
                hq = hq.copy()
                hq["L2"] = (hq.L // 2) * 2
                hq["wt"] = hq.N * np.exp(lna) * hq.L ** b
                pm = og.pm.reindex(sorted(set(hq.L2) | set(og.index))).sort_index()
                pm = pm.interpolate(limit_direction="both")
                hq["pm"] = hq.L2.map(pm)
                share = (hq.wt * hq.pm).sum() / hq.wt.sum()
                kg_h = hq.wt.sum() / 1000.0 / nbox.get(q, np.nan)
                line += f"; box: {nbox.get(q, 0)} hauls, {kg_h:.2f} kg/h mean, mature share of biomass {share:.2f}"
                # per period
                per = []
                for y0, y1 in ((1991, 2004), (2005, 2014), (2015, 2025)):
                    hp = hq[(pd.to_numeric(hq.Year) >= y0) & (pd.to_numeric(hq.Year) <= y1)]
                    if hp.wt.sum() > 0:
                        per.append(f"{y0}-{str(y1)[2:]} {(hp.wt * hp.pm).sum() / hp.wt.sum():.2f}")
                line += " (" + ", ".join(per) + ")"
                if og1 is not None:
                    pm1d = og1.pm.reindex(sorted(set(hq.L2) | set(og1.index))).sort_index()
                    pm1d = pm1d.interpolate(limit_direction="both")
                    hq["pm1"] = hq.L2.map(pm1d)
                    hq["dep"] = pd.to_numeric(hq.Depth, errors="coerce")
                    parts = []
                    for lo, hi in ((0, 100), (100, 400)):
                        dd = hq[(hq.dep >= lo) & (hq.dep < hi)]
                        if dd.wt.sum() > 0:
                            parts.append(f"{lo}-{hi} m {(dd.wt * dd.pm1).sum() / dd.wt.sum():.2f} "
                                         f"({dd.wt.sum() / 1000 / hh[hh.box & (hh.Quarter == q) & (pd.to_numeric(hh.Depth, errors='coerce') >= lo) & (pd.to_numeric(hh.Depth, errors='coerce') < hi)].shape[0]:.1f} kg/h)")
                    line += "; Q1-ogive share by haul depth: " + ", ".join(parts)
                    # area weighting: the box bed area is 0.232 < 100 m, 0.768 >= 100 m (maps/depth.npz)
                    num = den = 0.0
                    for (lo, hi), area in (((0, 100), 0.232), ((100, 400), 0.768)):
                        dd = hq[(hq.dep >= lo) & (hq.dep < hi)]
                        nh = hh[hh.box & (hh.Quarter == q) & (pd.to_numeric(hh.Depth, errors="coerce") >= lo)
                                & (pd.to_numeric(hh.Depth, errors="coerce") < hi)].shape[0]
                        if nh:
                            dens = dd.wt.sum() / nh
                            num += area * (dd.wt * dd.pm1).sum() / nh
                            den += area * dens
                    if den > 0:
                        line += f"; AREA-WEIGHTED Q1-ogive share {num / den:.2f}"
                if q != "1" and og1 is not None:
                    pm1 = og1.pm.reindex(sorted(set(hq.L2) | set(og1.index))).sort_index()
                    pm1 = pm1.interpolate(limit_direction="both")
                    s1 = (hq.wt * hq.L2.map(pm1)).sum() / hq.wt.sum()
                    line += f"; with the Q1 ogive {s1:.2f}"
            elif len(hq):
                line += f"; box: biomass present ({hq.N.sum():.0f} fish/h summed) but < 20 staged fish"
            print(f"Q{q}: {line}")
            if og is not None and sp in ("haddock", "cod", "whiting", "saithe", "Norway pout", "herring",
                                         "long rough dab", "plaice", "dab", "witch", "sprat", "mackerel"):
                print("   ogive: " + " ".join(f"{int(k)}:{v:.2f}" for k, v in og.pm.items() if og.n[k] >= 5))
        # D. spawning-stage fish in the box by month
        if len(st):
            sb = st[st.box & st.stage.isin(["spawn", "spent"])]
            if len(sb):
                print("D. spawning / spent fish in the BOX by quarter, month, stage (numbers):")
                print(sb.groupby(["Quarter", "Month", "stage"]).n.sum().unstack(fill_value=0).to_string())
            else:
                print("D. no spawning / spent fish in the box")
        # E. weight residual mature vs immature by quarter and sex
        if len(wl) > 50 and len(st):
            e = st[(st.w > 0) & (st.L > 0)].copy()
            e["res"] = np.log(e.w) - (lna + b * np.log(e.L))
            e["cls"] = np.where(e.stage == "imm", "imm", "mature")
            e = e[e.Sex.isin(["F", "M"])]
            if len(e):
                r = e.groupby(["Quarter", "Sex", "cls"]).agg(n=("n", "sum"), res=("res", "mean"),
                                                              L=("L", "median"))
                r["res %"] = (np.exp(r.res) - 1) * 100
                print("E. weight at length vs the region fit, %, by quarter / sex / maturity:")
                print(r[["n", "L", "res %"]].round(1).to_string())
                # E2: matched lengths - mature minus immature residual within each 2-cm class
                # holding both (>= 3 fish each), averaged over classes weighted by the mature n
                e["L2"] = (e.L // 2) * 2
                out = []
                for (qq, sx), d in e.groupby(["Quarter", "Sex"]):
                    g = d.groupby(["L2", "cls"]).agg(n=("n", "sum"), res=("res", "mean")).unstack()
                    g = g.dropna()
                    if ("n", "mature") not in g.columns or ("n", "imm") not in g.columns:
                        continue
                    g = g[(g[("n", "imm")] >= 3) & (g[("n", "mature")] >= 3)]
                    if len(g):
                        dif = g[("res", "mature")] - g[("res", "imm")]
                        wgt = g[("n", "mature")]
                        out.append(f"Q{qq} {sx}: {(np.exp((dif * wgt).sum() / wgt.sum()) - 1) * 100:+.1f} % "
                                   f"({len(g)} classes {int(g.index.min())}-{int(g.index.max())} cm, "
                                   f"mature n {int(wgt.sum())})")
                if out:
                    print("E2. mature minus immature weight at the SAME length: " + "; ".join(out))


main()
