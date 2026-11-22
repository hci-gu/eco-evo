"""Overview of saithe (AphiaID 126441) stomachs in the ICES stomach DB download held in
research_notes/"Födonivå och mättnad i fält"/data/A_ices_stomach/all (ICES data licence).
Per year/quarter/area: hauls, stomachs, fish prey weight by prey taxon.
    python3 -I saithe_db_overview.py
"""
import pandas as pd, numpy as np
D = "/home/simon/eco-evo/research_notes/Födonivå och mättnad i fält/data/A_ices_stomach/all/"
SAI = "126441"
pr = pd.read_csv(D + "PredatorInformation.csv", dtype=str, on_bad_lines="skip")
pr = pr[pr.AphiaIDPredator == SAI].copy()
hl = pd.read_csv(D + "HaulInformation.csv", dtype=str, on_bad_lines="skip")
py = pd.read_csv(D + "PreyInformation.csv", dtype=str, on_bad_lines="skip")
py = py[py.AphiaIDPredator == SAI].copy()
for c in ("Number", "Length", "Year", "Month"):
    pr[c] = pd.to_numeric(pr[c], errors="coerce")
pr = pr.merge(hl[["tblHaulID", "ICESrectangle", "Depth", "ShootLat", "ShootLong", "Survey"]], on="tblHaulID", how="left")
pr["Q"] = ((pr.Month - 1) // 3 + 1)
print("saithe predator rows", len(pr), "stomachs", pr.Number.sum(), "hauls", pr.tblHaulID.nunique())
print(pr.groupby(["Year", "Q"]).agg(rows=("Number", "size"), stomachs=("Number", "sum"),
                                    hauls=("tblHaulID", "nunique"), Lmed=("Length", "median")).to_string())
py["Weight"] = pd.to_numeric(py.Weight, errors="coerce")
py["Count"] = pd.to_numeric(py.Count, errors="coerce")
top = py.groupby("AphiaIDPrey").agg(n=("Weight", "size"), g=("Weight", "sum")).sort_values("g", ascending=False)
print("\nprey AphiaIDs by summed weight (top 25)")
print(top.head(25).to_string())
print("\nunits of Length / Weight:", pr.get("MeasurementIncrement").value_counts().head().to_dict(), py.UnitWgt.value_counts().head().to_dict())
print("GravMethod", py.GravMethod.value_counts().head().to_dict())
