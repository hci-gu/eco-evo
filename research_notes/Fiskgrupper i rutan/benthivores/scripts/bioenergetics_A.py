"""Evaluate Cmax and SMR functions for the benthivorous_fish members.

All literature parameters are typed in here with their source (no download is read).
Usage: python3 -I bioenergetics_A.py <out_dir>
Writes <out_dir>/bioenergetics_A_cmax.csv and <out_dir>/bioenergetics_A_smr.csv and prints both.

Conventions
- Cmax in kJ per kg fish per day (gross prey energy) and in % body weight per day of a
  reference prey of 4.184 kJ/g wet (the FB4 cod prey energy density), so species compare.
- SMR in mg O2 / kg / h; MJ per tonne per 6 h tick = mg O2/kg/h * 6 h * 13.6 J/mg / 1000.
  13.6 J/mg O2 is the oxycalorific coefficient used by Fonds et al. (1985) and assumed here.
- Annual mean over the model's 50 m series = mean of the 12 monthly Q10 factors (not the
  factor at the 8.4 C mean).
"""
import csv
import math
import os
import sys

T_EVAL = [6.0, 7.0, 8.4, 10.0, 11.7]
T50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]  # Jan..Dec, 50 m
T_BOTTOM = 6.5  # 100-300 m Skagerrak bottom water, ~6-7 C (stable); judgement midpoint
REF_ED = 4.184  # kJ/g wet, reference prey
J_PER_MG_O2 = 13.6


def mj_t_tick(mg_kg_h):
    return mg_kg_h * 6.0 * J_PER_MG_O2 / 1000.0


def pct_bw(kj_kg_d, ed=REF_ED):
    return kj_kg_d / ed / 10.0


# ---------------------------------------------------------------- Cmax functions
def kitchell(T, CQ, CTO, CTM):
    """FB4 consumption temperature equation 2 (Kitchell et al. 1977)."""
    if T >= CTM:
        return 0.0
    V = (CTM - T) / (CTM - CTO)
    Z = math.log(CQ) * (CTM - CTO)
    Y = math.log(CQ) * (CTM - CTO + 2.0)
    X = (Z ** 2 * (1.0 + (1.0 + 40.0 / Y) ** 0.5) ** 2) / 400.0
    return V ** X * math.exp(X * (1.0 - V))


def cod_cmax_kj(T, W):
    """Hansson et al. 1996 via FB4: g prey/g fish/d = 0.099 W^-0.195 f(T); prey 4184 J/g."""
    g_g_d = 0.099 * W ** -0.195 * kitchell(T, 2.41, 13.7, 21.0)
    return g_g_d * 1000.0 * 4.184


# Fonds et al. 1985 (ICES CM 1985/G:54) Table 8: DFCmax, J (physiological) per W^0.78 per day.
FONDS_T = [2.0, 6.0, 10.0, 14.0, 18.0]
FONDS_DFC = [78.0, 247.0, 453.0, 528.0, 762.0]
FONDS_GROSS = 23.3 / 20.4  # J-max / J-fys per mg AFD of mussel meat (same paper)


def loglin(T, xs, ys):
    for i in range(len(xs) - 1):
        if xs[i] <= T <= xs[i + 1]:
            f = (T - xs[i]) / (xs[i + 1] - xs[i])
            return math.exp(math.log(ys[i]) + f * (math.log(ys[i + 1]) - math.log(ys[i])))
    raise ValueError(T)


def plaice_cmax_kj(T, W, fit=False):
    """kJ gross per kg per day. fit=True uses the paper's Table 9 fit 79 e^(0.133T) W^0.78."""
    a = 79.0 * math.exp(0.133 * T) if fit else loglin(T, FONDS_T, FONDS_DFC)
    j_fys_d = a * W ** 0.78
    return j_fys_d * FONDS_GROSS / W  # J/g/d == kJ/kg/d


def arr(T, TA, Tref_C):
    return math.exp(TA / (Tref_C + 273.15) - TA / (T + 273.15))


def dab_over_plaice(T):
    """DEB {p_Am} ratio (Freitas et al. 2010, females, Tref 15 C): dab 317 TA 4000, plaice 599 TA 7000."""
    return (317.0 * arr(T, 4000.0, 15.0)) / (599.0 * arr(T, 7000.0, 15.0))


def deb_ingest_kj(T, W, pAm15, TA, kapX=0.8):
    """DEB max ingestion per kg: {p_Am} V^(2/3) / kap_X, V ~ W (cm3 ~ g, reserve ignored)."""
    return pAm15 * arr(T, TA, 15.0) * W ** (2.0 / 3.0) / kapX / W  # J/g/d


# ---------------------------------------------------------------- SMR functions
def q10_factor(T, Tref, q10):
    return q10 ** ((T - Tref) / 10.0)


def annual_mean_factor(Tref, q10, series=T50):
    return sum(q10_factor(t, Tref, q10) for t in series) / len(series)


# Plaice SMR: Fonds et al. 1985 Table 7, DOCmin mg O2 per day per W^0.78 (unfed, daytime lowest).
PL_T = [2.0, 6.0, 10.0, 14.0]
PL_DOC = [1.82, 2.67, 2.80, 4.53]


def fit_exp(xs, ys):
    n = len(xs)
    lx = [math.log(y) for y in ys]
    mx = sum(xs) / n
    my = sum(lx) / n
    b = sum((x - mx) * (l - my) for x, l in zip(xs, lx)) / sum((x - mx) ** 2 for x in xs)
    a = math.exp(my - b * mx)
    return a, b


def main(out):
    os.makedirs(out, exist_ok=True)
    rows = []
    # check against the prior note (cod 500 g, 7 C: 2.13 %BW/d)
    chk = cod_cmax_kj(7.0, 500.0) / 4.184 / 10.0
    print(f"check cod 500 g 7 C: {chk:.2f} %BW/d (prior note 2.13)")
    for W in (100.0, 300.0, 1000.0):
        for T in T_EVAL:
            k = cod_cmax_kj(T, W)
            rows.append(["cod/haddock-central", "Hansson 1996 FB4", W, T, round(k, 1), round(pct_bw(k), 2)])
    for W in (50.0, 150.0, 400.0):
        for T in T_EVAL:
            k = plaice_cmax_kj(T, W)
            kf = plaice_cmax_kj(T, W, fit=True)
            rows.append(["plaice", "Fonds 1985 Table 8 (log-lin interp)", W, T, round(k, 1), round(pct_bw(k), 2)])
            rows.append(["plaice", "Fonds 1985 Table 9 fit", W, T, round(kf, 1), round(pct_bw(kf), 2)])
            kd = deb_ingest_kj(T, W, 599.0, 7000.0)
            rows.append(["plaice", "DEB Freitas 2010 {pAm}/0.8", W, T, round(kd, 1), round(pct_bw(kd), 2)])
            r = dab_over_plaice(T)
            rows.append(["dab", f"plaice Fonds x DEB ratio {r:.2f}", W, T, round(k * r, 1), round(pct_bw(k * r), 2)])
            kdd = deb_ingest_kj(T, W, 317.0, 4000.0)
            rows.append(["dab", "DEB Freitas 2010 {pAm}/0.8", W, T, round(kdd, 1), round(pct_bw(kdd), 2)])
    with open(os.path.join(out, "bioenergetics_A_cmax.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "source", "mass_g", "T_C", "Cmax_kJ_per_kg_d", "Cmax_pctBW_d_at_4.184kJ_g"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")

    # temperature ratios of Cmax (item 6)
    print("\nCmax ratio 12 C / 6 C and 11.7 / 6.6:")
    print(" cod FB4:", round(cod_cmax_kj(12, 300) / cod_cmax_kj(6, 300), 2),
          round(cod_cmax_kj(11.7, 300) / cod_cmax_kj(6.6, 300), 2))
    print(" plaice Fonds table:", round(plaice_cmax_kj(12, 150) / plaice_cmax_kj(6, 150), 2),
          round(plaice_cmax_kj(11.7, 150) / plaice_cmax_kj(6.6, 150), 2))
    print(" plaice Fonds fit:", round(math.exp(0.133 * 6), 2), round(math.exp(0.133 * 5.1), 2))
    print(" dab DEB TA 4000:", round(arr(12, 4000, 15) / arr(6, 4000, 15), 2))
    print(" plaice DEB TA 7000:", round(arr(12, 7000, 15) / arr(6, 7000, 15), 2))
    print(" annual mean Cmax factor over 50 m series vs 8.4 C, cod FB4:",
          round(sum(cod_cmax_kj(t, 300) for t in T50) / 12 / cod_cmax_kj(8.4, 300), 3))

    # ---------------- SMR
    a, b = fit_exp(PL_T, PL_DOC)
    q10_pl = math.exp(10 * b)
    print(f"\nplaice DOCmin fit 2-14 C: {a:.3f} e^({b:.4f} T) mg O2/d/W^0.78, Q10 {q10_pl:.2f}")
    smr = []

    def add(sp, src, W, Tref, mg_ref, q10, note=""):
        am = annual_mean_factor(Tref, q10)
        bot = q10_factor(T_BOTTOM, Tref, q10)
        at = {T: mg_ref * q10_factor(T, Tref, q10) for T in T_EVAL}
        smr.append([sp, src, W, Tref, round(mg_ref, 1), q10,
                    round(at[7.0], 1), round(at[8.4], 1),
                    round(mj_t_tick(at[8.4]), 2),
                    round(mj_t_tick(mg_ref * am), 2),
                    round(mj_t_tick(mg_ref * bot), 2), note])

    # cod: Schurmann & Steffensen 1997, 57.0 mg/kg/h at 10 C, Q10 2.2 (prior note)
    add("cod", "Schurmann & Steffensen 1997", "as measured", 10.0, 57.0, 2.2)
    # haddock: Norin et al. 2019, 84.3 at 12 C, 63 g; scaled to 300 g with b = 0.8
    for q in (2.2, 2.5, 3.07):
        add("haddock", "Norin 2019, scaled 63->300 g (b 0.8)", 300, 12.0, 84.3 * (300 / 63.0) ** -0.2, q)
    add("haddock", "Norin 2019, 1000 g (b 0.8)", 1000, 12.0, 84.3 * (1000 / 63.0) ** -0.2, 2.5)
    # plaice: Fonds 1985 DOCmin fit, per kg per h
    for W in (50.0, 150.0, 400.0):
        mg10 = a * math.exp(b * 10.0) * W ** 0.78 / W / 24.0 * 1000.0
        add("plaice", "Fonds 1985 DOCmin fit", W, 10.0, mg10, round(q10_pl, 2))
    # plaice juvenile: Lever 1974, 10 g, log R = 1.23 + 0.038 T (ug-atom O/fish/h), Q10 2.40
    mg10 = 10 ** (1.23 + 0.038 * 10.0) * 16.0 / 1000.0 / 0.010
    add("plaice", "Lever 1974 (10 g juveniles)", 10, 10.0, mg10, round(10 ** 0.38, 2))
    # witch: Moser et al. 1996, 0.067 mg O2 g^-0.7 h^-1 at 6 C (5.3 g fish); Q10 assumed 2.2
    for W in (150.0, 400.0):
        add("witch", "Moser 1996 (5 g, scaled b 0.7); Q10 assumed", W, 6.0, 0.067 * W ** -0.3 * 1000.0, 2.2)
    # DEB maintenance [p_M] as an O2 equivalent (Freitas 2010; J/cm3/d at 15 C)
    for sp, pM, TA in (("plaice", 29.8, 7000.0), ("dab", 29.8, 4000.0), ("cod", 24.5, 7400.0)):
        mg15 = pM / J_PER_MG_O2 / 24.0 * 1000.0  # J/g/d -> mg O2/kg/h
        q = math.exp(TA * 10.0 / ((10.0 + 273.15) * (20.0 + 273.15)))  # Q10 around 15 C
        add(sp, f"DEB [pM] Freitas 2010 (TA {int(TA)}, Q10~{q:.2f})", "any", 15.0, mg15, round(q, 2),
            "DEB maintenance, not measured SMR")
    with open(os.path.join(out, "bioenergetics_A_smr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "source", "mass_g", "Tref_C", "SMR_mgO2_kg_h_at_Tref", "Q10",
                    "SMR_7C", "SMR_8.4C", "MJ_t_tick_8.4C", "MJ_t_tick_annual_mean_50m",
                    "MJ_t_tick_bottom_6.5C", "note"])
        w.writerows(smr)
    print()
    for r in smr:
        print(*r, sep=" | ")
    for q in (2.0, 2.2, 2.5, 3.0):
        print(f"annual-mean factor over 50 m series relative to 8.4 C, Q10 {q}: "
              f"{annual_mean_factor(8.4, q):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
