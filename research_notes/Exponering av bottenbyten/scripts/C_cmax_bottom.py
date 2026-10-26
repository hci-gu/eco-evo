"""Cmax of the benthivorous_fish members at bottom temperature, and the group ceiling.

Usage:
  python3 -I C_cmax_bottom.py <length_biomass_longrun.csv> <out_csv> [--bottom t1 ... t12]

Cmax unit everywhere: kJ gross prey energy per kg fish per day. Energy ceiling (model,
6 h tick) = Cmax / 4 MJ/t/tick; h = E_prey (MJ/t) / ceiling.

Size: every per-kg rate is averaged over the species' long-run (2001-25) biomass-weighted length
mix, E[W^b] computed directly from length_biomass_longrun.csv with each function's own exponent
(the earlier group_params.py used E[W^-0.2] / w_ref^-0.22 for the plaice-based species).

Temperature series (monthly, C):
  bottom  default = SHARK A16/A17 1991-2020 grid-mean bottom series (caller), annual 7.40
  at_50m  7.7 7.2 6.6 6.6 6.8 7.2 8.0 8.5 10.2 11.7 10.6 9.3 (the series the old 108 used)
  p100    7.9 7.2 6.7 6.8 6.8 7.0 7.4 7.5 7.9 8.3 9.0 9.1 (100 m placeholder from the task)
Annual mean Cmax = mean of the 12 monthly Cmax values (not Cmax at the mean T).

Group weights: the earlier group_params.py dict (haddock 38.6, cod 31.4, long rough dab 12.6,
dab 4.5, plaice 3.8, witch 1.7, starry ray 1.5, hagfish 1.3, grey gurnard 0.9, lemon sole 0.9,
greater weever 0.8, others 2.5; normalised), with hagfish/ray/weever/others at the earlier
analogue values. A second weighting renormalised over the named, measured-or-analogue species
(hagfish and 'others' dropped) is printed too.

Sources of each function: see the comments; tags in the caller's report.
"""
import math
import sys

import pandas as pd

args = sys.argv[1:]
lb_path, out_csv = args[0], args[1]
BOTTOM = [7.80, 6.99, 6.59, 6.58, 6.69, 6.96, 7.22, 7.41, 7.70, 8.04, 8.36, 8.50]
if "--bottom" in args:
    i = args.index("--bottom")
    BOTTOM = [float(x) for x in args[i + 1:i + 13]]
    assert len(BOTTOM) == 12
T50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]
P100 = [7.9, 7.2, 6.7, 6.8, 6.8, 7.0, 7.4, 7.5, 7.9, 8.3, 9.0, 9.1]
SERIES = {"bottom": BOTTOM, "at_50m": T50, "p100": P100}
TFIX = [6.5, 7.0, 7.5, 8.0]

LB = pd.read_csv(lb_path)


def ew(sp, b):
    """Biomass-weighted E[W^b] (W in g) over the species' long-run length mix."""
    g = LB[LB.species == sp]
    return float((g.kgh_2001_25 * g.w_g ** b).sum() / g.kgh_2001_25.sum())


# ---------------------------------------------------------------- cod / gadoid functions
def kitchell(T, CQ=2.41, CTO=13.7, CTM=21.0):
    V = (CTM - T) / (CTM - CTO)
    Z = math.log(CQ) * (CTM - CTO)
    Y = math.log(CQ) * (CTM - CTO + 2)
    X = Z * Z * (1 + (1 + 40 / Y) ** 0.5) ** 2 / 400
    return V ** X * math.exp(X * (1 - V))


def fb4(Ewb, T):
    """Hansson et al. 1996 (FB4): 0.099 W^-0.195 f(T) g/g/d, at the 4.184 kJ/g convention."""
    return 0.099 * Ewb * kitchell(T) * 4184.0  # kJ/kg/d


def jobling(Ewb, T):
    """Jobling 1988 via Neuenfeldt et al. 2020: 0.223 e^(0.104T-0.000112T^3) W^0.802 kJ/d.
    Ewb = E[W^-0.198]; per kg = x1000."""
    return 0.223 * math.exp(0.104 * T - 0.000112 * T ** 3) * Ewb * 1000.0


# Bjornsson, Steinarsson & Oddgeirsson 2001 (ICES JMS 58:29), Tables 2-4. Fish hand-fed to
# satiation; F-H: uneaten food removed and subtracted. Signs of the polynomial coefficients were
# lost in text extraction; they are reconstructed from the reported Gmax / FCmin (checked below).
# (W g, G: |a| b c Gmax, FC: |a| b c FCmin, protein %, fat %, carb %, feeding note)
BJ = {
    "E": (108.7, (0.58, 0.269, 0.0133, 0.77), (3.77, 0.581, 0.0297, 0.92), 42.0, 26.0, 14.5,
          "dry feed (Vextra trout), 3-5 meals/d"),
    "F": (447.0, (0.29, 0.094, 0.00533, 0.70), (2.99, 0.139, 0.0092, 2.46), 14.7, 10.5, 0.0,
          "capelin 4 d + shrimp 2 d per week"),
    "G": (874.0, (0.18, 0.111, 0.00684, 0.63), (3.24, 0.244, 0.0172, 2.37), 14.9, 10.1, 0.0,
          "capelin 4 d + shrimp 2 d per week"),
    "H": (2213.0, (0.08, 0.050, 0.00294, 0.30), (3.72, 0.385, 0.0240, 2.18), 14.6, 16.6, 0.0,
          "capelin 3 d per week only"),
}
KJ_PROT, KJ_FAT, KJ_CARB = 23.6, 39.5, 17.2  # gross energy kJ/g (assumed, Brett & Groves 1979 values)


def bj_poly(k):
    W, (ga, gb, gc, gmax), (fa, fb, fc, fcmin), *_ = BJ[k]
    # growth G = a + bT - cT^2, with sign(a) chosen to reproduce Gmax = a + b^2/4c
    a_g = gmax - gb * gb / (4 * gc)
    a_g = ga if abs(a_g - ga) < abs(a_g + ga) else -ga
    # feed conversion FC = a - bT + cT^2 (FCmin = a - b^2/4c)
    a_f = fa
    return W, (a_g, gb, gc), (a_f, fb, fc)


def bj_FI(k, T):
    """Feed intake, % body weight per day = FC(T) x G(T) (wet feed per gain x specific growth)."""
    W, (ag, gb, gc), (af, fb, fc) = bj_poly(k)
    G = ag + gb * T - gc * T * T
    FC = af - fb * T + fc * T * T
    return FC * G, G, FC


def bj_E(k):
    p, f, c = BJ[k][3], BJ[k][4], BJ[k][5]
    return (p * KJ_PROT + f * KJ_FAT + c * KJ_CARB) / 100.0  # kJ/g as fed


E_EXP = 0.125  # energy evacuation ~ E^(1-0.875) (Andersen gadoid model): prey-quality normalisation


def bj_kj(k, T, normalise=False):
    fi, _, _ = bj_FI(k, T)
    e = bj_E(k)
    v = fi / 100.0 * e * 1000.0
    if normalise:
        v *= (4.184 / e) ** E_EXP
    return v


def cod_bjornsson(Ewb195, T, normalise=True):
    """Mean of the wild-cod experiments F and G (447, 874 g, 6 d/week ad lib), size-scaled with
    W^-0.195 to the species' E[W^-0.195]. Valid 1-13 C."""
    vals = [bj_kj(k, T, normalise) * Ewb195 / BJ[k][0] ** -0.195 for k in ("F", "G")]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------- flatfish functions
PHYS2GROSS = 23.3 / 20.4  # Fonds 1985: J-max / J-physiological per mg AFDW mussel
FONDS_TAB = [(2.0, 78.0), (6.0, 247.0), (10.0, 453.0), (14.0, 528.0)]  # J-phys / W^0.78 / d


def fonds_table(Ewb22, T):
    for (t0, v0), (t1, v1) in zip(FONDS_TAB, FONDS_TAB[1:]):
        if t0 <= T <= t1:
            v = v0 * (v1 / v0) ** ((T - t0) / (t1 - t0))
            return v * Ewb22 * PHYS2GROSS  # J/g/d = kJ/kg/d
    raise ValueError(T)


def fonds_fit(Ewb22, T):
    return 79.0 * math.exp(0.133 * T) * Ewb22 * PHYS2GROSS


def arr(T, TA, Tref=15.0):
    return math.exp(TA / (Tref + 273.15) - TA / (T + 273.15))


def deb_pxm(pAm, TA, Ewb33, T):
    """Freitas et al. 2010 {p_Am} (J/cm2/d at 15 C) / kap_X 0.8, structural volume ~ wet mass."""
    return pAm / 0.8 * Ewb33 * arr(T, TA)  # J/g/d = kJ/kg/d


def plaice_c(sp, T):
    e22 = ew(sp, -0.22)
    return 0.5 * (fonds_table(e22, T) + fonds_fit(e22, T))


def q10(v_ref, Tref, T, Q=2.0):
    return v_ref * Q ** ((T - Tref) / 10.0)


# ---------------------------------------------------------------- species table
W = {"haddock": 38.6, "cod": 31.4, "long rough dab": 12.6, "dab": 4.5, "plaice": 3.8,
     "witch": 1.7, "starry ray": 1.5, "hagfish": 1.3, "grey gurnard": 0.9, "lemon sole": 0.9,
     "greater weever": 0.8, "others": 2.5}
tot = sum(W.values())
W = {k: v / tot for k, v in W.items()}
NAMED = [k for k in W if k not in ("hagfish", "others")]
WN = {k: W[k] / sum(W[j] for j in NAMED) for k in NAMED}


def cod_basis(basis, sp, T):
    if basis == "fb4":
        return fb4(ew(sp, -0.195), T)
    if basis == "bjornsson":
        return cod_bjornsson(ew(sp, -0.195), T, normalise=True)
    if basis == "bjornsson_raw":
        return cod_bjornsson(ew(sp, -0.195), T, normalise=False)
    if basis == "jobling":
        return jobling(ew(sp, -0.198), T)
    raise ValueError(basis)


def species_cmax(T, basis="bjornsson", hadd=1.0, level="central"):
    """Per-species Cmax (kJ/kg/d) at temperature T."""
    i = {"low": 0, "central": 1, "high": 2}[level]
    out = {}
    out["cod"] = cod_basis(basis, "cod", T)
    out["haddock"] = hadd * cod_basis(basis, "haddock", T)
    out["grey gurnard"] = cod_basis(basis, "grey gurnard", T)
    # plaice: low DEB, central mean(Fonds table, fit), high Fonds table
    pl = (deb_pxm(599.0, 7000.0, ew("plaice", -1 / 3), T), plaice_c("plaice", T),
          fonds_table(ew("plaice", -0.22), T))
    out["plaice"] = pl[i]
    # dab: central plaice-central x DEB {p_Am} ratio(T); low DEB dab direct; high plaice-central
    ratio = (317.0 / 599.0) * arr(T, 4000.0) / arr(T, 7000.0)
    pd_c = plaice_c("dab", T)
    out["dab"] = (deb_pxm(317.0, 4000.0, ew("dab", -1 / 3), T), pd_c * ratio, pd_c)[i]
    # long rough dab, witch: fraction of plaice-central at their size at 8.4 C, Q10 2 (judgement)
    out["long rough dab"] = q10((0.4, 0.6, 0.8)[i] * plaice_c("long rough dab", 8.4), 8.4, T)
    out["witch"] = q10((0.35, 0.6, 0.8)[i] * plaice_c("witch", 8.4), 8.4, T)
    out["lemon sole"] = (deb_pxm(599.0, 7000.0, ew("lemon sole", -1 / 3), T),
                         plaice_c("lemon sole", T), fonds_table(ew("lemon sole", -0.22), T))[i]
    # earlier analogue values (annual mean, ~8.4 C), Q10 2
    out["starry ray"] = q10((30.0, 50.0, 80.0)[i], 8.4, T)
    out["hagfish"] = q10((10.0, 20.0, 40.0)[i], 8.4, T)
    out["greater weever"] = q10((60.0, 90.0, 120.0)[i], 8.4, T)
    out["others"] = q10((60.0, 90.0, 120.0)[i], 8.4, T)
    return out


def annual(fn, series):
    vals = [fn(t) for t in series]
    return sum(vals) / len(vals)


def group(T, weights=W, **kw):
    s = species_cmax(T, **kw)
    return sum(weights[k] * s[k] for k in weights)


# ---------------------------------------------------------------- output
rows = []


def main():
    print("Series annual mean T: " + ", ".join(f"{k} {sum(v)/12:.2f} C" for k, v in SERIES.items()))
    print()
    print("Effective sizes (W_eff = E[W^b]^(1/b), g):")
    for sp in ["cod", "haddock", "plaice", "dab", "long rough dab", "witch", "lemon sole",
               "grey gurnard", "starry ray"]:
        print(f"  {sp:15s} b -0.195: {ew(sp,-0.195)**(1/-0.195):7.0f}   b -0.22: "
              f"{ew(sp,-0.22)**(1/-0.22):7.0f}   b -1/3: {ew(sp,-1/3)**(-3):7.0f}")
    print()

    print("1. Bjornsson et al. 2001 ad-lib intake = FC x G (polynomials of Tables 3-4)")
    print("   gross energy of feed from Table 2 proximate composition: protein 23.6, fat 39.5, "
          "carbohydrate 17.2 kJ/g (assumed coefficients)")
    for k in BJ:
        Wk, (ag, gb, gc), (af, fb, fc) = bj_poly(k)
        tg = gb / (2 * gc); tf = fb / (2 * fc)
        print(f"  exp {k}: W {Wk:.0f} g, {BJ[k][6]}; feed {bj_E(k):.2f} kJ/g; "
              f"check Gmax {ag + gb*tg - gc*tg*tg:.3f} (paper {BJ[k][1][3]}) at {tg:.1f} C, "
              f"FCmin {af - fb*tf + fc*tf*tf:.2f} (paper {BJ[k][2][3]}) at {tf:.1f} C")
        for T in (5.0, 6.5, 7.0, 7.5, 8.0, 8.5, 10.0):
            fi, G, FC = bj_FI(k, T)
            kj = bj_kj(k, T)
            f4 = fb4(Wk ** -0.195, T)
            jb = jobling(Wk ** -0.198, T)
            print(f"     T {T:4.1f}: G {G:5.3f} %/d  FC {FC:4.2f}  FI {fi:5.2f} %BW/d = {kj:6.1f} kJ/kg/d"
                  f"  (prey-normalised to 4.184 kJ/g: {bj_kj(k,T,True):6.1f});"
                  f"  FB4 {f4:6.1f} -> x{kj/f4:.2f} (norm x{bj_kj(k,T,True)/f4:.2f});"
                  f"  Jobling {jb:6.1f} -> x{kj/jb:.2f}")
        rows.append(dict(table="bjornsson2001", item=k, W_g=Wk, T=7.0, value=bj_kj(k, 7.0),
                         note=f"FI {bj_FI(k,7.0)[0]:.2f} %BW/d x {bj_E(k):.2f} kJ/g; FB4 ratio "
                              f"{bj_kj(k,7.0)/fb4(Wk**-0.195,7.0):.2f}"))
    print("  Q10 of FI over 6.5 -> 8.5 C: " + ", ".join(
        f"{k} {(bj_FI(k,8.5)[0]/bj_FI(k,6.5)[0])**5:.2f}" for k in BJ))
    print()

    print("2. Q10 over 6.5 -> 8.5 C implied by each temperature function")
    q = lambda f: (f(8.5) / f(6.5)) ** 5
    print(f"  FB4 Kitchell (cod, haddock)         {q(kitchell):.2f}")
    print(f"  Jobling 1988 exponential-cubic       {q(lambda t: math.exp(0.104*t-0.000112*t**3)):.2f}")
    print(f"  Bjornsson F/G (447/874 g cod)        {q(lambda t: bj_FI('F',t)[0]):.2f} / {q(lambda t: bj_FI('G',t)[0]):.2f}")
    print(f"  Fonds 1985 table (plaice, 6-10 C)    {q(lambda t: fonds_table(1.0,t)):.2f}")
    print(f"  Fonds 1985 fit e^0.133T              {q(lambda t: fonds_fit(1.0,t)):.2f}")
    print(f"  DEB TA 7000 K (plaice)               {q(lambda t: arr(t,7000.0)):.2f}")
    print(f"  DEB TA 4000 K (dab)                  {q(lambda t: arr(t,4000.0)):.2f}")
    print(f"  gadoid evacuation e^0.078T / 0.083T  {math.exp(0.78):.2f} / {math.exp(0.83):.2f}")
    print()

    print("3. Per-species Cmax (kJ/kg/d) at the long-run size; low / central / high")
    hdr = "  species          " + "".join(f"{'T'+str(t):>17s}" for t in TFIX) + \
          "".join(f"{k+' ann':>17s}" for k in SERIES)
    print(hdr)
    sp_list = list(W)
    for sp in sp_list:
        cells = []
        for t in TFIX + list(SERIES):
            vals = []
            for lev in ("low", "central", "high"):
                if sp in ("cod", "grey gurnard"):
                    basis = {"low": "fb4", "central": "bjornsson", "high": "jobling"}[lev]
                    kw = dict(basis=basis, level="central")
                elif sp == "haddock":
                    kw = dict(basis={"low": "fb4", "central": "bjornsson", "high": "jobling"}[lev],
                              hadd={"low": 0.85, "central": 1.0, "high": 1.3}[lev])
                else:
                    kw = dict(level=lev)
                if isinstance(t, float):
                    v = species_cmax(t, **kw)[sp]
                else:
                    v = annual(lambda x: species_cmax(x, **kw)[sp], SERIES[t])
                vals.append(v)
                rows.append(dict(table="species", item=sp, W_g="", T=t, value=round(v, 2),
                                 note=lev))
            cells.append(f"{vals[0]:5.0f}/{vals[1]:5.0f}/{vals[2]:5.0f}")
        print(f"  {sp:15s} " + "".join(f"{c:>17s}" for c in cells))
    print("  cod/gurnard low = FB4 x1.0, central = Bjornsson F/G prey-normalised, high = Jobling 1988;")
    print("  haddock low = FB4 x0.85, central = Bjornsson x1.0, high = Jobling x1.3")
    print()
    print("  other cod bases at the cod size (kJ/kg/d): " + "; ".join(
        f"T {t}: FB4 {cod_basis('fb4','cod',t):.0f}, Bjornsson raw {cod_basis('bjornsson_raw','cod',t):.0f}, "
        f"norm {cod_basis('bjornsson','cod',t):.0f}, Jobling {cod_basis('jobling','cod',t):.0f}" for t in TFIX))
    print("  dab checks: Hall 1987 dab = 0.6 x haddock -> " + ", ".join(
        f"T {t}: {0.6*cod_basis('bjornsson','dab',t):.0f} (hadd fn at dab size) vs central "
        f"{species_cmax(t)['dab']:.0f}" for t in (7.0, 9.5)))
    print(f"  Bagge 1995 dab 95-175 g, 9.2-9.8 C: 2.23 %BW/d boiled mussel = ~100 kJ/kg/d at 4.5 kJ/g "
          f"(assumed); model dab central at 9.5 C, 130 g: "
          f"{plaice_c('dab',9.5)*(317/599)*arr(9.5,4000)/arr(9.5,7000)*(130**-0.22/ew('dab',-0.22)):.0f}")
    print()

    print("4. Group Cmax (kJ/kg/d) and ceiling (MJ/t/tick = Cmax/4)")
    scen = [
        ("old method (FB4 cod, haddock x1.3)", dict(basis="fb4", hadd=1.3, level="central")),
        ("FB4 cod, haddock x1.0", dict(basis="fb4", hadd=1.0, level="central")),
        ("Bjornsson cod, haddock x1.0 (CENTRAL)", dict(basis="bjornsson", hadd=1.0, level="central")),
        ("Bjornsson cod, haddock x1.3", dict(basis="bjornsson", hadd=1.3, level="central")),
        ("low: FB4 cod, haddock x0.85, flat low", dict(basis="fb4", hadd=0.85, level="low")),
        ("high: Jobling cod, haddock x1.3, flat high", dict(basis="jobling", hadd=1.3, level="high")),
    ]
    cols = TFIX + list(SERIES)
    print("  " + " " * 44 + "".join(f"{('T'+str(c)) if isinstance(c,float) else c+' ann':>12s}" for c in cols))
    for name, kw in scen:
        for wlab, wts in (("all", W), ("named", WN)):
            vals = []
            for c in cols:
                if isinstance(c, float):
                    v = group(c, weights=wts, **kw)
                else:
                    v = annual(lambda x: group(x, weights=wts, **kw), SERIES[c])
                vals.append(v)
                rows.append(dict(table=f"group_{wlab}", item=name, W_g="", T=c,
                                 value=round(v, 2), note=f"ceiling {v/4:.2f} MJ/t/tick"))
            print(f"  {name[:38]:38s} {wlab:5s} " + "".join(f"{v:12.1f}" for v in vals))
            print(f"  {'  ceiling MJ/t/tick':38s} {'':5s} " + "".join(f"{v/4:12.2f}" for v in vals))
    print()
    print("  h = E_prey / ceiling (ticks per unit), bottom-series annual mean:")
    for name, kw in scen:
        c = annual(lambda x: group(x, **kw), BOTTOM) / 4
        print(f"  {name[:44]:44s} ceiling {c:5.2f}: h infauna (2600) {2600/c:6.1f}, "
              f"epifauna (3750) {3750/c:6.1f}, zooplankton (4500) {4500/c:6.1f}")
    print()
    print("  old value: 108.4 kJ/kg/d, ceiling 27.1 MJ/t/tick (group_params.py, 50 m series)")

    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
