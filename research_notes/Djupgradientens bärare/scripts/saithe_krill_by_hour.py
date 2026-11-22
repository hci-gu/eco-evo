"""Saithe (and hake) krill occurrence and freshness by time of day, ICES stomach DB.

Usage (run with python3 -I):
    python3 -I saithe_krill_by_hour.py <records_linked.csv.gz> <ices_stomach_dir>

Inputs are read only:
  records_linked.csv.gz : research_notes/"Krill over grunda bottnar"/stomachs/tables (one row per predator record,
                          occ_krill / g_krill / fed / hour / lat / lon / bed classes)
  ices_stomach_dir      : research_notes/"Fodoniva och mattnad i falt"/data/A_ices_stomach/all
                          (PreyInformation.csv carries DigestionStage per prey row; HaulInformation.csv Time)

Sun elevation is computed from lat, lon, date and the haul Time read as UTC (ICES convention; not verified per
upload). Phases: day (sun > 0 deg), twilight (-12 .. 0 deg), night (< -12 deg). Times recorded exactly as
"1200" are reported separately (suspected filler).
"""
import sys
import math
import numpy as np
import pandas as pd

rec_path, ices_dir = sys.argv[1], sys.argv[2]
R = pd.read_csv(rec_path, low_memory=False)
H = pd.read_csv(f"{ices_dir}/HaulInformation.csv", low_memory=False, dtype={"Time": str})
P = pd.read_csv(f"{ices_dir}/PreyInformation.csv", low_memory=False, dtype={"Time": str, "DigestionStage": str})

KRILL_APHIA = None  # resolved below from the krill flag


def sun_elev(lat, lon, year, month, day, hh, mm):
    # NOAA-style low-precision solar position, UTC input
    n = pd.Timestamp(year=int(year), month=int(month), day=int(day)).dayofyear
    g = 2 * math.pi / 365.0 * (n - 1 + (hh + mm / 60.0 - 12) / 24.0)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    tst = hh * 60 + mm + eqt + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cosz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return 90 - math.degrees(math.acos(max(-1, min(1, cosz))))


H["tblHaulID"] = H["tblHaulID"].astype(str)
R["tblHaulID_orig"] = R["tblHaulID_orig"].astype(str)
tmap = dict(zip(H.tblHaulID, H.Time))

for sp in ["saithe", "hake"]:
    s = R[(R.sp == sp)].copy()
    s["Time"] = s.tblHaulID_orig.map(tmap)
    s = s[s.Time.notna() & s.Time.str.fullmatch(r"\d{3,4}")]
    s["hh"] = s.Time.str.zfill(4).str[:2].astype(int)
    s["mm"] = s.Time.str.zfill(4).str[2:].astype(int)
    s["filler"] = s.Time.str.zfill(4) == "1200"
    s["elev"] = [sun_elev(a, b, y, m, d, h, mi) for a, b, y, m, d, h, mi in
                 zip(s.lat, s.lon, s.year, s.month, s.day, s.hh, s.mm)]
    s["phase"] = np.where(s.elev > 0, "day", np.where(s.elev > -12, "twilight", "night"))
    fed = s[s.anyprey.astype(str) == "True"]  # records with any prey row (the "fed" column is a count)
    print(f"\n===== {sp}: records with a usable Time {len(s)}, fed records {len(fed)} =====")
    print("Time == 1200 (suspected filler):", int(s.filler.sum()), "of", len(s))
    for lab, sub in [("all fed", fed), ("fed, Time != 1200", fed[~fed.filler]),
                     ("fed, 35-70 cm, Time != 1200", fed[~fed.filler & fed.L_cm.between(35, 70)])]:
        g = sub.groupby("phase").agg(n_rec=("occ_krill", "size"), fo_krill=("occ_krill", "mean"),
                                     n_fish_in_records=("Number", "sum"))
        print(f"\n-- {lab}: krill FO (share of fed records with krill) by sun phase")
        print(g.to_string())
        hh_ = sub.groupby(["phase", "tblHaulID"]).occ_krill.max().groupby("phase").agg(["size", "mean"])
        print(f"-- {lab}: haul-level krill FO by phase (size = hauls)")
        print(hh_.to_string())
        sub = sub.assign(hbin=(sub.hh // 3) * 3)
        g2 = sub.groupby("hbin").agg(n_rec=("occ_krill", "size"), fo_krill=("occ_krill", "mean"))
        print(f"-- {lab}: by 3-h UTC bin")
        print(g2.to_string())
        # weight share where prey weights exist
        w = sub[sub[[c for c in sub.columns if c.startswith("g_")]].notna().any(axis=1)].copy()
        if len(w):
            gc = [c for c in w.columns if c.startswith("g_")]
            w["tot"] = w[gc].fillna(0).sum(axis=1)
            w = w[w.tot > 0]
            w["ks"] = w.g_krill.fillna(0) / w.tot
            print(f"-- {lab}: krill weight share in records with prey weights, by phase "
                  f"(mean of per-record shares; pooled = sum krill g / sum g)")
            print(w.groupby("phase").apply(lambda x: pd.Series({"n": len(x), "mean_share": x.ks.mean(),
                                                                 "pooled": x.g_krill.fillna(0).sum() / x.tot.sum()}),
                                           include_groups=False).to_string())
    # confounding: phase x quarter, krill FO
    sub = fed[~fed.filler]
    print(f"\n-- {sp}: fed, Time != 1200: krill FO by quarter x phase (size, mean)")
    print(sub.groupby(["quarter", "phase"]).occ_krill.agg(["size", "mean"]).unstack(1).to_string())
    print(f"-- {sp}: fed, Time != 1200: records by 3-h UTC bin x quarter, and krill FO")
    sub2 = sub.assign(hbin=(sub.hh // 3) * 3)
    print(sub2.groupby(["hbin", "quarter"]).occ_krill.agg(["size", "mean"]).unstack(1).to_string())
    # by bed class and phase
    print(f"\n-- {sp}: fed, Time != 1200: krill FO by bed class (bed_rectmed) x phase")
    sub = fed[~fed.filler]
    print(sub.groupby(["bed_rectmed", "phase"]).occ_krill.agg(["size", "mean"]).unstack(1).to_string())
    print(f"-- {sp}: years, months, areas of fed records with Time != 1200")
    print(sub.groupby(["year"]).size().to_string())
    print(sub.groupby(["area", "quarter"]).size().unstack(1).fillna(0).astype(int).to_string())

    # digestion stage of krill prey rows, by phase
    pids = set(fed.tblPredatorInformationID)
    pk = P[P.tblPredatorInformationID.isin(pids)].copy()
    kr = fed[fed.occ_krill == 1].tblPredatorInformationID
    print(f"\n-- {sp}: prey rows for fed records {len(pk)}; DigestionStage values:",
          pk.DigestionStage.value_counts(dropna=False).head(8).to_dict())
    # krill rows: Aphia of euphausiids (Euphausiidae 110671, Meganyctiphanes norvegica 110690,
    # Thysanoessa 110675/110676/110677/110681..., Euphausiacea 1128)
    eu = {"110671", "1128", "110673", "110690", "110707", "110708", "110706", "110681", "110695", "110687"}
    pk["AphiaIDPrey"] = pk.AphiaIDPrey.astype(str).str.replace(r"\.0$", "", regex=True).str.strip('"')
    pkk = pk[pk.AphiaIDPrey.isin(eu)].merge(fed[["tblPredatorInformationID", "phase", "hh", "filler"]],
                                            on="tblPredatorInformationID")
    print(f"-- {sp}: euphausiid prey rows {len(pkk)}; by phase x DigestionStage")
    if len(pkk):
        print(pkk.groupby(["phase", "DigestionStage"], dropna=False).size().unstack(1).fillna(0).astype(int)
              .to_string())
        print("   by filler flag:", pkk.groupby("filler").size().to_dict())
