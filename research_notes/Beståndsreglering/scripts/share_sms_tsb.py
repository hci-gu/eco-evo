"""Total-stock-biomass to SSB ratio per species from the ICES WGSAM North Sea SMS key run 2023.

Usage: python3 -I share_sms_tsb.py <sms2023_summary.out> <out_csv>
TSB = annual mean over quarters of sum over ages of BIO; SSB = Q1 SSB (sum over ages).
Both in tonnes for the SMS area (North Sea + Skagerrak for most species). Years 2001-2022.
"""
import sys

import pandas as pd

src, out = sys.argv[1:3]
NAMES = {16: "cod", 17: "whiting", 18: "haddock", 19: "saithe", 21: "herring", 22: "N.sandeel",
         23: "S.sandeel", 24: "Norway pout", 25: "sprat", 26: "plaice", 27: "sole"}
S = pd.read_csv(src, sep=r"\s+")
S = S[S["Species.n"].isin(NAMES) & S.Year.between(2001, 2022)]
tsb = S.groupby(["Species.n", "Year", "Quarter"]).BIO.sum().groupby(["Species.n", "Year"]).mean()
ssb = S[S.Quarter == 1].groupby(["Species.n", "Year"]).SSB.sum()
# juvenile (age 0 + 1) share of the annual-mean biomass
juv = S[S.Age <= 1].groupby(["Species.n", "Year", "Quarter"]).BIO.sum().groupby(["Species.n", "Year"]).mean()
D = pd.DataFrame(dict(TSB=tsb, SSB_Q1=ssb, juv01=juv)).reset_index()
D["ratio"] = D.TSB / D.SSB_Q1
R = D.groupby("Species.n").agg(TSB_0122=("TSB", "mean"), SSB_0122=("SSB_Q1", "mean"),
                               ratio_mean=("ratio", "mean"), ratio_min=("ratio", "min"),
                               ratio_max=("ratio", "max"), juv01=("juv01", "mean"))
R["juv01_share"] = R.juv01 / R.TSB_0122
R.insert(0, "species", R.index.map(NAMES))
R.to_csv(out)
print(R.round(3).to_string())
