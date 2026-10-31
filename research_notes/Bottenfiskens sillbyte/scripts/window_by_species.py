"""Per-species prey-size windows of the benthivores -> planktivores pair, and the pool stage access
[f_J, f_A] under three species weightings (no model output read).

Usage: python3 -I window_by_species.py <stage_length_biomass.csv>

Same data and window rule as research_notes/"Arsungarnas andel" (A-ring) scripts/stage_access.py:
prey stage lengths (q-corrected GOV kg/h by 1-cm class, J = below length at age 1 in Q3; Q1/Q3 2001-25,
Q2/Q4 1991-96; annual value = mean of the four quarters); a member's window = prey length <= ratio x
predator length over the predator's box length distribution (cod >= 20 cm from the same table, as
stage_access.py; haddock >= 20 cm likewise).

Species weights inside the pool:
Blue whiting has no rows in the stage table, so it drops out of every window share (as in
stage_access.py); it is above the cod window (p10 17.5 cm) except for cod >= ~60 cm.
  POOL   pool biomass weights by quarter (stage_access.py; reproduces [0.494, 0.136])
  ENC    POOL x the encounter overlap Fbar_j (the a's own species weighting; Attackrater ur motesteori
         benthivores_encounter.md section 1c / 2.3): Fbar_j = w_day F_day + w_night F_night with
         w_day = 0.5 x 0.36, w_night = 0.5 x 0.09 (daylight share ~0.5 over the year x the mean
         R^2/R_sat^2 by day / night, section 2.3), F from the encounter's POOL table (central)
  ENC'   as ENC with the GOV-revised daytime near-bed shares of this note (herring 0.34, sprat 0.5,
         blue whiting 0.6; evidence.md section 3)
"""
import sys

import warnings

import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
import pandas as pd

SL = pd.read_csv(sys.argv[1])
POOL_W = {1: {"herring": 0.88, "blue whiting": 0.0, "Norway pout": 0.11, "sprat": 0.01},
          2: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030},
          3: {"herring": 0.78, "blue whiting": 0.15, "Norway pout": 0.04, "sprat": 0.03},
          4: {"herring": 0.788, "blue whiting": 0.106, "Norway pout": 0.076, "sprat": 0.030}}
PERIOD = {1: "2001-25", 2: "1991-96", 3: "2001-25", 4: "1991-96"}
SPP = ["herring", "Norway pout", "blue whiting", "sprat"]
W_DAY, W_NIGHT = 0.5 * 0.36, 0.5 * 0.09
F_CUR = {"herring": (0.15, 0.02), "blue whiting": (0.3, 0.15), "Norway pout": (0.7, 0.4), "sprat": (0.1, 0.01)}
F_REV = {"herring": (0.34, 0.02), "blue whiting": (0.6, 0.15), "Norway pout": (0.7, 0.4), "sprat": (0.5, 0.01)}


def fbar(F):
    return {s: W_DAY * d + W_NIGHT * n for s, (d, n) in F.items()}


def pred_len(sp, Lmin=20):
    p = SL[(SL.species == sp) & (SL.period == "2001-25")].groupby("L_cm")[["J", "A"]].sum().sum(axis=1)
    p = p[p.index >= Lmin]
    return p.index.values + 0.5, (p / p.sum()).values


PRED = {"cod": pred_len("cod"), "haddock": pred_len("haddock")}
Lc, wc = PRED["cod"]
print(f"cod >= 20 cm, biomass-weighted length: p10/p50/p90 "
      + " / ".join(f"{Lc[np.searchsorted(np.cumsum(wc), q)]:.1f}" for q in (0.1, 0.5, 0.9)) + " cm")
Lh, wh = PRED["haddock"]
print(f"haddock >= 20 cm: p10/p50/p90 "
      + " / ".join(f"{Lh[np.searchsorted(np.cumsum(wh), q)]:.1f}" for q in (0.1, 0.5, 0.9)) + " cm")


# length-dependent cod ratios (SMS 2023 key run, mass-weighted p95 prey/predator length;
# tables/sms_diet_pred16.txt and research_notes/"Arsungarnas andel"/tables/sms_prey_size_stdout.txt)
RATIO_L = {"Ldep_allP": [(30, 0.40), (45, 0.34), (65, 0.30), (999, 0.27)],
           "Ldep_herring": [(30, 0.38), (40, 0.42), (50, 0.39), (70, 0.35), (999, 0.21)]}


def rat(ratio, Lp):
    if isinstance(ratio, str):
        for top, r in RATIO_L[ratio]:
            if Lp < top:
                return r
    return ratio


def inside(q, sp, stage, pred, ratio):
    e = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
    if e.empty or e[stage].sum() <= 0:
        return np.nan, 0.0
    L = e.L_cm.values + 0.5
    v = e[stage].values / e[stage].sum()
    Ls, ws = PRED[pred]
    return sum(wp * v[L <= rat(ratio, Lp) * Lp].sum() for Lp, wp in zip(Ls, ws)), e[stage].sum()


def per_species(pred, ratio):
    """annual-mean f_J, f_A and J share per species (mean over quarters where the stage exists)."""
    out = {}
    for sp in SPP:
        fj, fa, js = [], [], []
        for q in (1, 2, 3, 4):
            a, bj = inside(q, sp, "J", pred, ratio)
            b, ba = inside(q, sp, "A", pred, ratio)
            fj.append(a)
            fa.append(b)
            js.append(bj / (bj + ba) if bj + ba > 0 else np.nan)
        out[sp] = (np.nanmean(fj), np.nanmean(fa), np.nanmean(js), fj, fa)
    return out


def pool_access(pred, ratio, weights):
    fJ, fA = [], []
    for q in (1, 2, 3, 4):
        tj = ta = ij = ia = 0.0
        for sp in SPP:
            w = POOL_W[q][sp] * weights.get(sp, 1.0)
            if w <= 0:
                continue
            a, _ = inside(q, sp, "J", pred, ratio)
            b, _ = inside(q, sp, "A", pred, ratio)
            if a == a:
                tj += w
                ij += w * a
            if b == b:
                ta += w
                ia += w * b
        fJ.append(ij / tj)
        fA.append(ia / ta)
    return np.mean(fJ), np.mean(fA)


print("\nPer-species window shares (annual mean of Q1-Q4): f_J, f_A, species J share of biomass")
for pred, ratios in (("cod", (0.30, 0.35, 0.40, "Ldep_allP", "Ldep_herring")), ("haddock", (0.30,))):
    for r in ratios:
        ps = per_species(pred, r)
        print(f"  {pred} ratio {r}: " + "; ".join(f"{sp} [{v[0]:.3f}, {v[1]:.3f}] J {v[2]:.2f}" for sp, v in ps.items()))

fc, fr = fbar(F_CUR), fbar(F_REV)
for lab, fb in (("current", fc), ("revised", fr)):
    tot = sum(POOL_W[2][s] * fb[s] for s in SPP)
    print(f"\nFbar_j ({lab} overlaps): " + ", ".join(f"{s} {fb[s]:.4f}" for s in SPP)
          + f"; pool sum s_j Fbar_j {tot:.4f}; encounter share of each species "
          + ", ".join(f"{s} {POOL_W[2][s] * fb[s] / tot:.3f}" for s in SPP))
print(f"  ratio of pool sums revised / current: "
      f"{sum(POOL_W[2][s] * fr[s] for s in SPP) / sum(POOL_W[2][s] * fc[s] for s in SPP):.2f}")

print("\nPool stage access [f_J, f_A] and effective f at J/B 0.38:")
for pred, r in (("cod", 0.30), ("cod", 0.35), ("cod", 0.40), ("cod", "Ldep_allP"), ("cod", "Ldep_herring"),
                ("haddock", 0.30)):
    for lab, wts in (("POOL", {}), ("ENC", fc), ("ENC'", fr)):
        j, a = pool_access(pred, r, wts)
        print(f"  {pred:7s} ratio {str(r):12s} {lab:5s} [{j:.3f}, {a:.3f}]  f_eff {0.38 * j + 0.62 * a:.3f}")
