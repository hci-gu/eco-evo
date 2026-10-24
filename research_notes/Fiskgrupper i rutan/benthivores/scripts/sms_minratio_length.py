"""Convert SMS 2023 key-run minimum predator/prey WEIGHT ratios (the largest prey seen in the 1981-91
stomachs) into prey/predator LENGTH ratios with the region's length-weight fits.
Usage: python3 -I sms_minratio_length.py <length_weight_all.csv>
Ratios typed from data/lit/sms2023_sizepref/min_max_size_pref.out (rows: grey gurnard, cod, haddock).
"""
import csv, sys
lw = {r["species"]: (float(r["a"]), float(r["b"])) for r in csv.DictReader(open(sys.argv[1]))}
NAME = {"cod": "Gadus morhua", "haddock": "Melanogrammus aeglefinus", "grey_gurnard": "Eutrigla gurnardus",
        "whiting": "Merlangius merlangus", "herring": "Clupea harengus", "norway_pout": "Trisopterus esmarkii",
        "sprat": "Sprattus sprattus", "sandeel": "Ammodytes marinus"}
MINR = {"grey_gurnard": {"cod": 9.037, "whiting": 5.322, "haddock": 19.818, "herring": 8.299, "sandeel": 11.590, "norway_pout": 8.325, "sprat": 35.075},
        "cod": {"cod": 3.633, "whiting": 4.491, "haddock": 6.984, "herring": 2.744, "sandeel": 5.513, "norway_pout": 4.186, "sprat": 11.318},
        "haddock": {"sandeel": 4.162, "norway_pout": 8.688}}
def W(sp, L): a, b = lw[NAME[sp]]; return a * L ** b
def L(sp, w): a, b = lw[NAME[sp]]; return (w / a) ** (1 / b)
print("predator,pred_cm,prey,min_weight_ratio,max_prey_cm,length_ratio")
for pred, d in MINR.items():
    for Lp in (25, 35, 50):
        for prey, r in d.items():
            lp = L(prey, W(pred, Lp) / r)
            print(f"{pred},{Lp},{prey},{r},{lp:.1f},{lp / Lp:.2f}")
