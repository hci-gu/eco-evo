"""Empty-stomach share and mean stomach content from the ICES stomach database.

Input: the full download of https://stomachdata.ices.dk/api/download (unzipped
into data/A_ices_stomach/all/). Run with `python3 -I scripts/A_ices_stomach_summary.py`.

Per species x region x quarter x length class (North Sea + Skagerrak only):
- stomachs sampled (sum of Number; records with StomachEmpty given),
- empty share = sum StomachEmpty / sum Number,
- regurgitated share,
- mean content per non-regurgitated stomach (g wet, sum of prey Weight / (N - regurgitated)),
  also as % of predator body weight (IndWgt, mean weight of the fish in the record),
- for whiting: mean content / Smax, Smax = 0.0352 L^1.68 e^(0.004 T) g (Rindorf & Gislason
  2005 appendix, from Andersen & Riis-Vestergaard 2003); T is not in the data, the
  temperature term is negligible (e^(0.004*8) = 1.03) and is set at 8 C.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "A_ices_stomach" / "all"
OUT = ROOT / "tables" / "A_ices_stomach_summary.csv"

SPECIES = {126436: "cod", 126437: "haddock", 126438: "whiting", 126441: "saithe",
           127143: "plaice", 127139: "dab", 127137: "long rough dab", 127141: "witch"}
LCLASS = [0, 20, 30, 40, 50, 70, 1000]


def region(lat, lon):
    """North Sea (IV) and Skagerrak/Kattegat (IIIa) only; everything else NaN."""
    if not (51.0 <= lat <= 62.0) or lon < -4.0:
        return None
    if lon <= 7.0:
        return "North Sea"
    if lat >= 57.0 and lon <= 11.5:
        return "Skagerrak-Kattegat"
    if lat < 57.0 and lon <= 8.5:
        return "North Sea"
    return None


def main():
    kw = dict(low_memory=False, on_bad_lines="skip")
    pred = pd.read_csv(DATA / "PredatorInformation.csv", **kw)
    prey = pd.read_csv(DATA / "PreyInformation.csv", **kw)
    haul = pd.read_csv(DATA / "HaulInformation.csv", **kw)

    pred["species"] = pd.to_numeric(pred.AphiaIDPredator, errors="coerce").map(SPECIES)
    pred = pred[pred.species.notna()].copy()
    haul["lat"] = pd.to_numeric(haul.ShootLat, errors="coerce")
    haul["lon"] = pd.to_numeric(haul.ShootLong, errors="coerce")
    haul["region"] = [region(a, b) if np.isfinite(a) and np.isfinite(b) else None
                      for a, b in zip(haul.lat, haul.lon)]
    pred = pred.merge(haul[["tblHaulID", "region"]], on="tblHaulID", how="left")
    pred = pred[pred.region.notna()].copy()

    for c in ["Number", "StomachEmpty", "Regurgitated", "IndWgt", "Length", "Month"]:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")
    # Length is in cm (checked per species and year); pooled records give the lower
    # bound of a class of width MeasurementIncrement -> use the class midpoint.
    inc = pd.to_numeric(pred.MeasurementIncrement, errors="coerce").fillna(1.0)
    pred["L_cm"] = pred.Length + inc / 2.0
    # Body weight: some years report IndWgt in kg (median < 5) -> g. Then a log-log
    # length-weight fit per species on single-fish records, used for every record
    # (1981-1991 records carry no weight).
    pred["IndWgt"] = pred.IndWgt * np.where(
        pred.groupby(["species", "Year"]).IndWgt.transform("median") < 5.0, 1000.0, 1.0)
    lw = {}
    fulton = 100.0 * pred.IndWgt / pred.Length ** 3
    ok = (pred.Number == 1) & (pred.Length >= 10) & fulton.between(0.5, 1.8)
    for sp, g in pred[ok].groupby("species"):
        b, loga = np.polyfit(np.log(g.Length + 0.5), np.log(g.IndWgt), 1)
        lw[sp] = (np.exp(loga), b, len(g))
    print("length-weight fits W = a L^b (g, cm):", {k: (round(v[0], 5), round(v[1], 3), v[2]) for k, v in lw.items()})
    pred["IndWgt"] = [lw[sp][0] * L ** lw[sp][1] if sp in lw else np.nan
                      for sp, L in zip(pred.species, pred.L_cm)]
    # Implausible lengths (mm entered as cm in 2021-2023 uploads): whiting > 60 cm,
    # cod > 150 cm are dropped.
    bad = ((pred.species == "whiting") & (pred.L_cm > 60)) | ((pred.species == "cod") & (pred.L_cm > 150))
    print("dropped implausible-length records:", int(bad.sum()))
    pred = pred[~bad].copy()
    pred["quarter"] = ((pred.Month - 1) // 3 + 1).astype("Int64")
    pred = pred[pred.StomachEmpty.notna() & pred.Number.notna()].copy()
    pred["Regurgitated"] = pred.Regurgitated.fillna(0)

    prey["W"] = pd.to_numeric(prey.Weight, errors="coerce")
    content = prey.groupby("tblPredatorInformationID").W.agg(["sum", "count", lambda s: s.isna().sum()])
    content.columns = ["prey_g", "n_prey_rec", "n_prey_noW"]
    pred = pred.merge(content, left_on="tblPredatorInformationID", right_index=True, how="left")
    pred["prey_g"] = pred.prey_g.fillna(0.0)
    pred["n_prey_noW"] = pred.n_prey_noW.fillna(0)
    # Records with prey but missing weights would bias content low: flag them.
    pred["weight_ok"] = pred.n_prey_noW == 0
    # A non-empty, non-regurgitated record with zero summed prey weight means the
    # prey were not weighed (datasets that only record occurrence).
    pred["has_food"] = (pred.Number - pred.StomachEmpty - pred.Regurgitated) > 0

    pred["Lclass"] = pd.cut(pred.L_cm, LCLASS, right=False,
                            labels=["<20", "20-29", "30-39", "40-49", "50-69", ">=70"])
    smax = 0.0352 * pred.L_cm ** 1.68 * np.exp(0.004 * 8.0)
    pred["smax_g"] = np.where(pred.species == "whiting", smax * pred.Number, np.nan)
    pred["bw_g"] = pred.IndWgt * pred.Number

    rows = []
    keys = ["species", "region", "quarter", "Lclass"]
    for k, g in pred.groupby(keys, observed=True):
        n = g.Number.sum()
        if n < 20:
            continue
        gw = g[g.weight_ok]
        fed = gw[gw.has_food]
        weighed = (fed.prey_g > 0).mean() if len(fed) else np.nan
        if not (weighed >= 0.9):
            gw = gw.iloc[0:0]
        n_ok = (gw.Number - gw.Regurgitated).clip(lower=0).sum()
        mean_g = gw.prey_g.sum() / n_ok if n_ok > 0 else np.nan
        bw = gw.bw_g.sum() / gw.Number.sum() if gw.IndWgt.notna().all() and len(gw) else np.nan
        sm = gw.smax_g.sum() / gw.Number.sum() if len(gw) else np.nan
        rows.append(dict(zip(keys, k)) | dict(
            years=f"{int(g.Year.min())}-{int(g.Year.max())}",
            n_stomachs=int(n),
            empty_share=round(g.StomachEmpty.sum() / n, 3),
            regurgitated_share=round(g.Regurgitated.sum() / n, 3),
            mean_content_g=round(mean_g, 2),
            mean_bw_g=round(bw, 0) if np.isfinite(bw) else np.nan,
            content_pct_bw=round(100 * mean_g / bw, 3) if np.isfinite(bw) and bw > 0 else np.nan,
            content_over_smax=round(mean_g / sm, 3) if np.isfinite(sm) and sm > 0 else np.nan,
            share_records_weight_ok=round(len(gw) / len(g), 2),
            fed_records_weighed=round(weighed, 2) if np.isfinite(weighed) else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(out.to_string(index=False))

    # Species x quarter, all sizes pooled, and species totals.
    print()
    for lvl in (["species", "region", "quarter"], ["species", "region"]):
        agg = pred.groupby(lvl, observed=True).apply(
            lambda g: pd.Series(dict(n=g.Number.sum(),
                                     empty=g.StomachEmpty.sum() / g.Number.sum(),
                                     regurg=g.Regurgitated.sum() / g.Number.sum(),
                                     years=f"{int(g.Year.min())}-{int(g.Year.max())}")),
            include_groups=False)
        print(agg.to_string())
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
