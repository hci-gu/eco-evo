"""Evaluate Wisconsin-type Cmax functions for the planktivorous_fish members.

Cmax(W, T) = CA * W**CB * f(T)   [g prey wet weight / g fish / day]
f(T): CEQ 3 = Thornton & Lessem (1978), CEQ 2 = Kitchell et al. (1977),
both as written in the Fish Bioenergetics 4.0 manual (Deslauriers et al. 2017).

Energy: Cmax_E = Cmax * ED_ref (kJ / g fish / day), ED_ref = the prey energy
density the mass limit is taken to stand for (an assumption, see the note).
Model conversion: fraction of body mass per day on the model's zooplankton
(4.5 kJ/g) = Cmax_E / 4.5; ceiling per 6 h tick = fraction / 4;
handling time h = 1 / ceiling = 4 / fraction.

No input files; all parameters are written below with their source.
Usage: python3 -I cmax_eval.py
"""
import math

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
# Water temperature by month (C), the model's simulation_settings.temperature.
T_0_50 = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]
T_50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]
# Feeding season 1 Apr - 15 Sep (Bachiller et al. 2018): Apr-Aug whole, Sep half.
SEASON_W = [0, 0, 0, 1, 1, 1, 1, 1, 0.5, 0, 0, 0]

PREY_ED_MODEL = 4.5   # kJ/g, model zooplankton energy_content 4500 MJ/t
TICKS_PER_DAY = 4
H_NOW = 25.0


def f_tl(T, CQ, CTO, CTM, CTL, CK1, CK4):
    """Thornton & Lessem (FB4 CEQ 3)."""
    g1 = math.log(0.98 * (1 - CK1) / (CK1 * 0.02)) / (CTO - CQ)
    l1 = math.exp(g1 * (T - CQ))
    ka = CK1 * l1 / (1 + CK1 * (l1 - 1))
    g2 = math.log(0.98 * (1 - CK4) / (CK4 * 0.02)) / (CTL - CTM)
    l2 = math.exp(g2 * (CTL - T))
    kb = CK4 * l2 / (1 + CK4 * (l2 - 1))
    return ka * kb


def f_kitchell(T, CQ, CTO, CTM):
    """Kitchell et al. (FB4 CEQ 2)."""
    if T >= CTM:
        return 0.0
    v = (CTM - T) / (CTM - CTO)
    z = math.log(CQ) * (CTM - CTO)
    y = math.log(CQ) * (CTM - CTO + 2)
    x = (z ** 2 * (1 + (1 + 40 / y) ** 0.5) ** 2) / 400
    return v ** x * math.exp(x * (1 - v))


# Parameter sets (FB4 Parameters_official.csv; Rose et al. 2008 Table 1).
SETS = {
    "herring_juv": dict(CA=0.642, CB=-0.256, f=lambda T: f_tl(T, 1, 15, 17, 25, 0.1, 0.01)),
    "herring_adult": dict(CA=0.642, CB=-0.256, f=lambda T: f_tl(T, 1, 13, 15, 23, 0.1, 0.01)),
    "cod_hansson": dict(CA=0.099, CB=-0.195, f=lambda T: f_kitchell(T, 2.41, 13.7, 21.0)),
}

# Lab ad libitum rations of juvenile herring (De Silva & Balbontin 1974, via
# Bernreuther 2007 p. 89-90): 6.1-18.8 g WW fed mysids, minced mussels and
# squid; 8.1-12.7 %BW/d at 14.5 C and 3.5-5.0 %BW/d at 6.5 C. Midpoints at the
# geometric-mean mass 10.7 g, exponential in T between the two temperatures
# (extrapolated outside 6.5-14.5 C), mass exponent -0.256 (their 0.744 - 1).
LAB_W0 = math.sqrt(6.1 * 18.8)
LAB_HI, LAB_LO = (8.1 + 12.7) / 200, (3.5 + 5.0) / 200
LAB_K = math.log(LAB_HI / LAB_LO) / (14.5 - 6.5)


def lab_ration(W, T):
    T = min(T, 14.5)
    return LAB_LO * math.exp(LAB_K * (T - 6.5)) * (W / LAB_W0) ** -0.256

# Members: (parameter set, body mass g, temperature table, ED_ref kJ/g, weight)
# ED_ref: the prey energy density the mass-based Cmax is read as (see note).
MEMBERS = {
    "herring": ("herring_juv", 60.0, T_0_50, None, 0.79),
    "blue_whiting": ("cod_hansson", 100.0, T_50, None, 0.105),
    "norway_pout": ("cod_hansson", 25.0, T_50, None, 0.075),
    "sprat": ("herring_juv", 10.0, T_0_50, None, 0.03),
}


def cmax(setname, W, T):
    s = SETS[setname]
    return s["CA"] * W ** s["CB"] * s["f"](T)


def wmean(vals, w):
    return sum(v * x for v, x in zip(vals, w)) / sum(w)


def summarise(setname, W, temps):
    monthly = [cmax(setname, W, T) for T in temps]
    t_ann = sum(temps) / 12
    t_fs = wmean(temps, SEASON_W)
    return dict(
        monthly=monthly,
        t_ann=t_ann, t_fs=t_fs,
        at_t_ann=cmax(setname, W, t_ann),
        at_t_fs=cmax(setname, W, t_fs),
        mean_ann=sum(monthly) / 12,
        mean_fs=wmean(monthly, SEASON_W),
        opt=SETS[setname]["CA"] * W ** SETS[setname]["CB"],
    )


def to_model(e_kj):
    frac = e_kj / PREY_ED_MODEL
    return frac, TICKS_PER_DAY / frac


def main(ed_sets):
    print("Temperature factors f(T) (fraction of the optimum Cmax):")
    for T in [3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17]:
        print(f"  T {T:4.1f}: herring_juv {SETS['herring_juv']['f'](T):.3f}"
              f"  herring_adult {SETS['herring_adult']['f'](T):.3f}"
              f"  cod_hansson {SETS['cod_hansson']['f'](T):.3f}")
    print()
    rows = {}
    for name, (sname, W, temps, _, wt) in MEMBERS.items():
        s = summarise(sname, W, temps)
        rows[name] = s
        print(f"{name}: set {sname}, W {W:g} g, Cmax at optimum {s['opt']:.4f} g/g/d")
        print(f"  T annual mean {s['t_ann']:.2f} C, feeding season mean {s['t_fs']:.2f} C")
        print(f"  Cmax(g/g/d) at T_ann {s['at_t_ann']:.4f}, at T_fs {s['at_t_fs']:.4f},"
              f" mean of months: year {s['mean_ann']:.4f}, season {s['mean_fs']:.4f}")
        print("  monthly: " + " ".join(f"{m} {c:.3f}" for m, c in zip(MONTHS, s["monthly"])))
    # sensitivity: pout / blue whiting on flat near-bottom 7 C
    for name, W in [("norway_pout", 25.0), ("blue_whiting", 100.0)]:
        print(f"{name} at flat 7 C (near-bottom): {cmax('cod_hansson', W, 7.0):.4f} g/g/d;"
              f" at 6 C {cmax('cod_hansson', W, 6.0):.4f}; at 8 C {cmax('cod_hansson', W, 8.0):.4f}")
    # sensitivity: herring adult set
    s = summarise("herring_adult", 60.0, T_0_50)
    rows["herring_adult60"] = s
    print(f"herring (adult set, 60 g): T_ann {s['at_t_ann']:.4f}, T_fs {s['at_t_fs']:.4f},"
          f" mean months year {s['mean_ann']:.4f}, season {s['mean_fs']:.4f}")
    print("  monthly: " + " ".join(f"{m} {c:.3f}" for m, c in zip(MONTHS, s["monthly"])))
    print(f"Lab ad-lib herring ration: k {LAB_K:.4f}/C (Q10 {math.exp(10*LAB_K):.2f}), W0 {LAB_W0:.1f} g")
    for name, W in [("herring", 60.0), ("sprat", 10.0)]:
        mon = [lab_ration(W, T) for T in T_0_50]
        ma, mf = sum(mon) / 12, wmean(mon, SEASON_W)
        print(f"  {name} {W:g} g: mean of months year {ma:.4f} g/g/d, season {mf:.4f};"
              f" at T_ann {lab_ration(W, sum(T_0_50)/12):.4f}, T_fs {lab_ration(W, wmean(T_0_50, SEASON_W)):.4f}")
        print("  monthly: " + " ".join(f"{m} {c:.3f}" for m, c in zip(MONTHS, mon)))
        for ed in (3.4, 3.7, 4.0):
            fa, ha = to_model(ma * ed)
            ff, hf = to_model(mf * ed)
            print(f"    lab food ED {ed}: year {ma*ed:.3f} kJ/g/d ({100*fa:.1f} %/d on 4.5, h {ha:.0f}),"
                  f" season {mf*ed:.3f} kJ/g/d ({100*ff:.1f} %/d, h {hf:.0f})")
    # Weighted group value with the lab curve for herring and sprat (food 3.7 kJ/g)
    # and the cod proxy for pout / blue whiting (4.18 kJ/g).
    for k, lab in [("year", "mean of months, year"), ("fs", "mean of months, Apr-mid-Sep")]:
        tot = 0.0
        for name, (sname, W, temps, _, wt) in MEMBERS.items():
            if sname == "herring_juv":
                mon = [lab_ration(W, T) for T in temps]
                ed = 3.7
            else:
                mon = [cmax(sname, W, T) for T in temps]
                ed = 4.18
            v = sum(mon) / 12 if k == "year" else wmean(mon, SEASON_W)
            tot += wt * v * ed
        f, h = to_model(tot)
        print(f"  WEIGHTED lab-curve reading, {lab}: {tot:.3f} kJ/g/d = {100*f:.1f} %bw/d on 4.5 kJ/g, h {h:.0f}")
    print()

    for label, eds in ed_sets.items():
        print(f"=== Prey ED reading '{label}': " + ", ".join(f"{k} {v} kJ/g" for k, v in eds.items()))
        tot = {"ann": 0.0, "fs": 0.0, "mann": 0.0, "mfs": 0.0}
        for name, (sname, W, temps, _, wt) in MEMBERS.items():
            s = rows[name]
            ed = eds[name]
            e = {"ann": s["at_t_ann"] * ed, "fs": s["at_t_fs"] * ed,
                 "mann": s["mean_ann"] * ed, "mfs": s["mean_fs"] * ed}
            for k in tot:
                tot[k] += wt * e[k]
            fa, ha = to_model(e["mann"])
            ff, hf = to_model(e["mfs"])
            print(f"  {name:13s} kJ/g/d: T_ann {e['ann']:.3f} T_fs {e['fs']:.3f} |"
                  f" mean-of-months year {e['mann']:.3f} season {e['mfs']:.3f} |"
                  f" on 4.5 kJ/g: year {100*fa:.1f} %/d (h {ha:.0f}), season {100*ff:.1f} %/d (h {hf:.0f})")
        for k, lab in [("ann", "at annual-mean T"), ("fs", "at feeding-season-mean T"),
                       ("mann", "mean of monthly Cmax, year"), ("mfs", "mean of monthly Cmax, Apr-mid-Sep")]:
            f, h = to_model(tot[k])
            print(f"  WEIGHTED {lab:36s}: {tot[k]:.3f} kJ/g/d = {100*f:.1f} %bw/d on 4.5 kJ/g,"
                  f" ceiling {f/4:.4f} t/t/tick, h {h:.1f}; h_now/h = {H_NOW/h:.2f},"
                  f" now/Cmax = {0.72/tot[k]:.2f}")
        print()


if __name__ == "__main__":
    # Readings of the mass-based Cmax as an energy limit (kJ per g prey):
    ED_SETS = {
        # Rudstam-model herring/sprat applied to zooplankton: NEMURO.FISH CALz
        # 2.58; Arrhenius & Hansson 1993 Baltic zooplankton 2.85; Bachiller
        # copepods 3.6. Cod (Hansson 1996): prey ED filled from the source.
        "low (NEMURO 2.58 / cod 4.18)": {"herring": 2.58, "sprat": 2.58, "norway_pout": 4.18, "blue_whiting": 4.18},
        "central (Baltic zoo 2.85 / cod 4.18)": {"herring": 2.85, "sprat": 2.85, "norway_pout": 4.18, "blue_whiting": 4.18},
        "copepod (3.6 / cod 4.18)": {"herring": 3.6, "sprat": 3.6, "norway_pout": 4.18, "blue_whiting": 4.18},
        "copepod + cod on fat fish (3.6 / 5.86)": {"herring": 3.6, "sprat": 3.6, "norway_pout": 5.86, "blue_whiting": 5.86},
        "mass limit on model prey (4.5 all)": {"herring": 4.5, "sprat": 4.5, "norway_pout": 4.5, "blue_whiting": 4.5},
    }
    main(ED_SETS)
