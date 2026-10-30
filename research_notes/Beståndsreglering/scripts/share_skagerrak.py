"""Skagerrak (3a.20) share of each species' North Sea + 3a survey index by stage (WP3).

Usage: python3 -I share_skagerrak.py <share_survey.csv> <out_csv>
Skagerrak share = index(3a20) / index(4+3a) = share_box(4+3a) / share_box(3a20) (same box numerator).
Also the box share of the Skagerrak index. Climatology 2001-25, R and S box estimators averaged.
"""
import sys

import pandas as pd

src, out = sys.argv[1:3]
V = pd.read_csv(src)
V = V[~V.stage.str.endswith("_kg")]
P = V.pivot_table(index=["species", "stage", "quarter"], columns="area", values="index")
P["skag_of_4+3a"] = P["3a20 (Skagerrak)"] / P["4+3a"]
P["kat_of_4+3a"] = (P["3a (20+21)"] - P["3a20 (Skagerrak)"]) / P["4+3a"]
B = V[V.area == "3a20 (Skagerrak)"].set_index(["species", "stage", "quarter"])[["share_R", "share_S"]]
P["box_of_skag"] = B.mean(axis=1)
P = P[["skag_of_4+3a", "kat_of_4+3a", "box_of_skag"]]
P.to_csv(out)
print("Skagerrak sea area / (4+3a) sea area = %.3f (ETOPO; box 3600 km2 = 0.104 of 3a20)" % (34526 / 692376))
print((P * 100).round(2).to_string())
