"""Net-feeding checks on the 2026 saithe-pout pairing (see saithe_pout_2026.py).
1. DigestionStage of pout in each haul vs the pout catch (haul 152: no pout in the catch).
2. Per-fish pout content (%BW) vs the haul's pout catch; fish above 2.5 / 5 %BW.
3. Ration implied in the top hauls vs the field Q/B 2.83 /yr (0.78 %BW/d).
4. Type II fit on GOV hauls only (Walker q is a GOV catchability).
Usage: python3 -I net_feeding_check.py <repo root>
"""
import math, os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(sys.argv[1])
NOTE = os.path.join(ROOT, "research_notes", "Sejens mötesfaktor")
DB = os.path.join(ROOT, "research_notes", "Födonivå och mättnad i fält", "data", "A_ices_stomach", "all")
py = pd.read_csv(os.path.join(DB, "PreyInformation.csv"), dtype=str, on_bad_lines="skip")
py = py[(py.AphiaIDPredator == "126441") & (py.Year == "2026")].copy()
py["g"] = pd.to_numeric(py.Weight, errors="coerce") / 1000.0
py["HaulNo"] = pd.to_numeric(py.HaulNo)
pr = pd.read_csv(os.path.join(DB, "PredatorInformation.csv"), dtype=str, on_bad_lines="skip")
pr = pr[(pr.AphiaIDPredator == "126441") & (pr.Year == "2026")].copy()
pr["HaulNo"] = pd.to_numeric(pr.HaulNo); pr["W"] = pd.to_numeric(pr.IndWgt)
H = pd.read_csv(os.path.join(NOTE, "tables", "saithe_pout_2026_hauls.csv")).set_index("HaulNo")
p = py[py.AphiaIDPrey == "126444"]
t = p.groupby(["HaulNo", "DigestionStage"]).g.sum().unstack(fill_value=0.0)
t = t.join(H[["Gear", "catch_g", "D", "P"]], how="left")
print("1. pout prey mass (g) by DigestionStage per haul, with the haul's pout catch (g/haul) and density (t/km2)")
print(t.sort_values("catch_g").to_string(float_format=lambda x: f"{x:.4g}"))
print("   prey lengths of pout (cm, stage 0 / stage 1):",
      p[p.DigestionStage == "0"].Length.value_counts().head(8).to_dict(), "/",
      p[p.DigestionStage == "1"].Length.value_counts().head(8).to_dict())
fp = p.groupby("tblPredatorInformationID").g.sum()
pr["S_pout_pct"] = 100 * pr.tblPredatorInformationID.map(fp).fillna(0.0) / pr.W
pr = pr.drop(columns=["Gear"]).join(H[["catch_g", "Gear"]], on="HaulNo")
print("\n2. per-fish pout content (%BW): fish with pout", int((pr.S_pout_pct > 0).sum()), "of", len(pr),
      "; > 2.5 %BW", int((pr.S_pout_pct > 2.5).sum()), "; > 5 %BW", int((pr.S_pout_pct > 5).sum()),
      "; max", round(pr.S_pout_pct.max(), 1))
for lo, hi in ((0, 1000), (1000, 10000), (10000, 1e9)):
    s = pr[(pr.catch_g >= lo) & (pr.catch_g < hi)]
    print(f"   haul pout catch {lo:>6.0f}-{hi:<9.0f} g: fish {len(s):3d}, with pout {int((s.S_pout_pct > 0).sum()):3d}, "
          f"mean pout %BW {s.S_pout_pct.mean():.2f}, > 5 %BW {int((s.S_pout_pct > 5).sum())}")
print("\n3. implied pout ration in the top hauls (%BW/d) vs field Q/B 2.83 /yr = 0.78 %BW/d (all food, annual mean)")
print((100 * H.P).sort_values(ascending=False).head(8).round(2).to_dict())
print(f"   fish-weighted mean pout ration over all 29 hauls {100 * np.average(H.P, weights=H.n_fish):.2f} %BW/d")
# 4. GOV-only fit
def t2(Hs):
    P_, D_, c_ = Hs.P.values, Hs.D.values, Hs.chain_day.values
    best = (np.inf, None, None)
    for lp in np.linspace(math.log(1e-4), math.log(0.2), 100):
        pm = math.exp(lp)
        for lf in np.linspace(math.log(1e-4), math.log(100.0), 250):
            f = math.exp(lf)
            sse = float(((P_ - f * c_ * D_ / (1 + f * c_ * D_ / pm)) ** 2).sum())
            if sse < best[0]:
                best = (sse, f, pm)
    return best[1], best[2]
for lab, sel in (("GOV only", H.Gear == "GOV"), ("JTDS only", H.Gear == "JTDS")):
    Hs = H[sel]
    f, pm = t2(Hs)
    print(f"\n4. {lab}: n {len(Hs)}, pout D range {Hs.D.min():.3g}-{Hs.D.max():.3g} t/km2; Type II factor {f:.3g}, "
          f"Pmax {pm:.3g}; ratio factor {Hs.P.sum() / (Hs.chain_day * Hs.D).sum():.3g}")
