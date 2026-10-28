"""Interference algebra for the benthivore group (no model run, pure arithmetic).

Compares, for the four benthivore pairs with the saturation levels x = a*h*B_visible
measured in the trained run (infauna 59, epifauna 38, herring 5.3, whiting 0.25):

  A. per-pair Beddington-DeAngelis (the engine):   f_j/f_j(0) = (1+x_j)/(1+x_j+I)
  B. shared-denominator multi-prey BD:            f_j/f_j(0) = (1+S)/(1+S+I), S = sum x_k
  C. Crowley-Martin (per pair or shared):          f_j/f_j(0) = 1/(1+I)
  plus the local Hassell-Varley exponent m = -dln f/dln P = I/(1+x+I) (BD) or I/(1+I) (CM),

with I = w*P and P = k x (grid mean) for k = 1, 5, 13, 30. Also the ceiling share
x/(1+x+I) quoted in the brief, and the eat-weighted intake cut under the policy's
diet (weights = share of intake at I = 0; illustrative).

Writes tables/interference_algebra.csv and prints a text table.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tables", "interference_algebra.csv")

PAIRS = {"infauna": 59.0, "epifauna": 38.0, "herring(planktivores)": 5.3,
         "whiting": 0.25}
W = 0.45
MEANS = {"run_mean_1.0": 1.0, "start_1.53": 1.53}
MULTS = [1, 5, 13, 30]


def main():
    S = sum(PAIRS.values())
    rows = []
    for mean_name, mean in MEANS.items():
        for k in MULTS:
            P = k * mean
            I = W * P
            for prey, x in PAIRS.items():
                bd_pair = (1 + x) / (1 + x + I)
                bd_shared = (1 + S) / (1 + S + I)
                cm = 1.0 / (1 + I)
                rows.append(dict(
                    grid_mean=mean_name, multiple=k, P_t_km2=round(P, 2),
                    I=round(I, 2), prey=prey, x=x,
                    ceiling_share_I0=round(x / (1 + x), 3),
                    ceiling_share=round(x / (1 + x + I), 3),
                    cut_per_pair_BD=round(1 - bd_pair, 3),
                    cut_shared_BD=round(1 - bd_shared, 3),
                    cut_CM=round(1 - cm, 3),
                    m_local_per_pair_BD=round(I / (1 + x + I), 3),
                    m_local_shared_BD=round(I / (1 + S + I), 3),
                    m_local_CM=round(I / (1 + I), 3)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"S = {S}")
    hdr = ("mean", "k", "P", "I", "prey", "x", "f/fmax", "cutPair", "cutShared",
           "cutCM", "mPair")
    print("%-14s %3s %6s %6s %-22s %5s %6s %7s %9s %6s %6s" % hdr)
    for r in rows:
        print("%-14s %3d %6.2f %6.2f %-22s %5.2f %6.3f %7.3f %9.3f %6.3f %6.3f" % (
            r["grid_mean"], r["multiple"], r["P_t_km2"], r["I"], r["prey"], r["x"],
            r["ceiling_share"], r["cut_per_pair_BD"], r["cut_shared_BD"],
            r["cut_CM"], r["m_local_per_pair_BD"]))
    # The brief's measured I's
    print("\nAt measured biomass-weighted I values:")
    for I in (3.6, 8.0, 9.5, 22.0):
        print(f"I={I}: " + ", ".join(
            f"{p} {x/(1+x+I):.2f} (I0 {x/(1+x):.2f}, cut {1-(1+x)/(1+x+I):.2f})"
            for p, x in PAIRS.items()) + f"; shared cut {1-(1+S)/(1+S+I):.3f}; CM cut {1-1/(1+I):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
