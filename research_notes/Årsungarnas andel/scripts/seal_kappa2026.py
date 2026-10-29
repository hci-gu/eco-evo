"""Harbour seal prey sizes from Kappa et al. 2026 (Mar. Mamm. Sci. 42: e70069), Hvaler, northern
Skagerrak, 73 scats, September 2020: mean prey weight per species from the paper's totals.

Hand-typed from the paper (text and Fig. 2, read off the bars; no download is read): 2671 identified
otoliths represent 33.7 kg of fish (13 g per otolith; 1977-78: 57 g). Per species: share of otoliths
(%) and of biomass (%). Mean weight = biomass share x 33.7 kg / (otolith share x 2671). Lengths from
the length-weight table (Fiskgrupper i rutan/tables/length_weight_all.csv constants below). Otolith
sizes are not corrected for erosion in the paper, so true sizes are larger (often 10-30 %).

Usage: python3 -I seal_kappa2026.py
"""
import math

TOTAL_KG, N_OTO = 33.7, 2671
# species: (otolith %, biomass %, a, b) - a, b in g and cm
SP = {"haddock/pollack/saithe": (12.8, 18.7, 0.005451, 3.14816),
      "cod": (1.7, 10.7, 0.005786, 3.139519),
      "whiting": (3.6, 4.8, 0.0063, 3.1),
      "Norway pout": (21.8, 11.3, 0.005527, 3.092817),
      "herring": (0.5, 1.4, 0.002231, 3.370193),
      "long rough dab": (0.6, 1.2, 0.0105, 3.08),
      "blue whiting": (7.9, 3.5, 0.0045, 3.1)}
for sp, (o, bm, a, b) in SP.items():
    w = bm / 100 * TOTAL_KG * 1000 / (o / 100 * N_OTO)
    L = (w / a) ** (1 / b)
    print(f"{sp:24s} mean weight {w:6.1f} g -> length {L:5.1f} cm (x1.25 erosion {L * 1.25:5.1f})")

# juvenile share of the benthivore part of the take (haddock half of h/p/s, J; cod by a lognormal with
# CV 0.3 around its mean length, J below 27.1 cm; long rough dab and plaice adult as in the stage table)
def p_below(mean_L, cut, cv=0.3):
    s = math.sqrt(math.log(1 + cv * cv))
    mu = math.log(mean_L) - s * s / 2
    return 0.5 * (1 + math.erf((math.log(cut) - mu) / (s * math.sqrt(2))))


cod_L = (SP["cod"][1] / 100 * TOTAL_KG * 1000 / (SP["cod"][0] / 100 * N_OTO) / 0.005786) ** (1 / 3.139519)
had, cod, other = 0.5 * 18.7, 10.7, 1.2 + 0.2
x = 0.155  # box benthivore J/B, size-defined, 2001-25 mean
for lab, Lc in (("as measured", cod_L), ("erosion x1.25", cod_L * 1.25)):
    je = (had + cod * p_below(Lc, 27.1)) / (had + cod + other)
    fa = x * (1 - je) / (je * (1 - x))  # f_A with f_J = 1 at the box J/B
    print(f"{lab}: cod {Lc:.1f} cm, juvenile share of the benthivore take {je:.2f}; f_A at box J/B {fa:.3f}")
