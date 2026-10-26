"""Field rations divided by a laboratory Cmax at the same size and temperature.

Run: python3 -I scripts/A_ration_over_cmax.py  (no inputs read; all numbers below are
quoted from the sources named in tables/A_evidence_notes.txt).

Cmax functions
- Cod: Fish Bioenergetics 4 (Hansson et al. 1996, Baltic cod; lab satiation data),
  Cmax = 0.099 W^-0.195 f(T) g prey / g fish / d, Kitchell f with CQ 2.41, CTO 13.7,
  CTM 21; prey 4.184 kJ/g (FB4 convention). Check against the project's table
  (benthivores evidence A): 1000 g at 7 C -> 78.0 kJ/kg/d.
- Haddock: no Cmax function exists. Shown as cod function x 1.0 and x 1.3 (the project's
  judgement, from juvenile lab comparisons).
- Whiting: Andersen & Riis-Vestergaard (2003) ad-libitum sandeel ration,
  Wmax = 0.00122 L^2.28 e^(0.076 T) g sandeel/d, sandeel 6.08 kJ/g (as quoted in Rindorf &
  Gislason 2005, appendix eq. A1).

Field rations
- SMS key run 2023 (ICES WGSAM 2024, Table 3-8; consum_ab.in): ration (kg/quarter) =
  a_q W^b, W in kg, evacuation-based (Andersen & Beyer 2005 cylinder model). Wet weight
  of prey as eaten.
- Daan (1973, ICES CM 1973/F:38 Table II): daily ration (g) of North Sea cod by length.

Temperatures: the manifest's at_50m monthly series (Skagerrak, 50 m) averaged per
quarter, plus a +1.5 C sensitivity (the SMS rations were evaluated at North Sea ambient
temperature per stratum, which is not given in the report).
"""
import math

AT50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]  # mareld2.yaml
TQ = [sum(AT50[3 * q:3 * q + 3]) / 3 for q in range(4)]
DAYS_Q = 365.25 / 4


def kitchell(T, CQ=2.41, CTO=13.7, CTM=21.0):
    V = (CTM - T) / (CTM - CTO)
    if V <= 0:
        return 0.0
    Z = math.log(CQ) * (CTM - CTO)
    Y = math.log(CQ) * (CTM - CTO + 2)
    X = Z ** 2 * (1 + (1 + 40 / Y) ** 0.5) ** 2 / 400
    return V ** X * math.exp(X * (1 - V))


def cod_cmax_kj(Wg, T):
    """kJ per kg fish per day."""
    return 0.099 * Wg ** -0.195 * kitchell(T) * 4184.0


def cod_cmax_jobling_kj(Wg, T):
    """Jobling (1988) cod Cmax as used by Neuenfeldt et al. 2020 eq. 2:
    0.223 e^(0.104T - 0.000112T^3) W^0.802, taken as kJ/d (W in g) -> kJ/kg/d."""
    return 0.223 * math.exp(0.104 * T - 0.000112 * T ** 3) * Wg ** 0.802 / (Wg / 1000.0)


def whiting_wmax_kj(Lcm, T, Wg):
    """kJ per kg fish per day from the sandeel satiation ration."""
    return 0.00122 * Lcm ** 2.28 * math.exp(0.076 * T) * 6.08 / (Wg / 1000.0)


def whiting_L(Wg):
    # length-weight W = 0.00542 L^3.105 (fit to the ICES stomach database single-fish records,
    # scripts/A_ices_stomach_summary.py)
    return (Wg / 0.00542) ** (1 / 3.105)


SMS = {  # a per quarter, b ; ration kg per quarter, W kg
    "cod": ([0.9004, 1.21211, 1.24653, 1.38951], 0.78595),
    "haddock": ([0.32311, 0.44561, 0.59414, 0.58802], 0.71443),
    "whiting": ([0.42637, 0.45491, 0.67914, 0.57439], 0.68266),
    "saithe": ([0.39357, 1.13916, 0.60354, 0.7059], 1.04466),
}


def sms_pct_bw_d(sp, q, Wkg):
    a, b = SMS[sp]
    return 100 * a[q] * Wkg ** b / DAYS_Q / Wkg


def main():
    print("check: cod Cmax 1000 g at 7 C = %.1f kJ/kg/d (table: 78.0)" % cod_cmax_kj(1000, 7.0))
    print("quarterly at_50m temperatures:", [round(t, 2) for t in TQ])
    print()
    E_FB4 = 4.184
    print("SMS ration / Cmax (wet/wet, prey at the FB4 4.184 kJ/g convention)")
    print("species  W(g)  q  T    ration%BW/d  Cmax%BW/d  ratio   ratio(T+1.5)")
    rows = []
    for sp, Ws in (("cod", [300, 1190, 2400, 5000]), ("haddock", [150, 275, 600])):
        for W in Ws:
            ann_r, ann_c = 0, 0
            for q in range(4):
                r = sms_pct_bw_d(sp, q, W / 1000)
                c = cod_cmax_kj(W, TQ[q]) / E_FB4 / 10
                c2 = cod_cmax_kj(W, TQ[q] + 1.5) / E_FB4 / 10
                mults = (1.0, 1.3) if sp == "haddock" else (1.0,)
                for m in mults:
                    lab = sp if m == 1.0 else sp + "x1.3"
                    print(f"{lab:12s} {W:5d} Q{q+1} {TQ[q]:5.2f} {r:8.3f} {c*m:9.3f} {r/(c*m):7.3f} {r/(c2*m):7.3f}")
                    rows.append((lab, W, q + 1, TQ[q], r, c * m, r / (c * m), r / (c2 * m)))
                ann_r += r / 4
                ann_c += c / 4
            print(f"{sp:12s} {W:5d} annual mean ration {ann_r:.3f} %BW/d, Cmax {ann_c:.3f}, ratio {ann_r/ann_c:.3f}"
                  + (f" (x1.3: {ann_r/(1.3*ann_c):.3f})" if sp == "haddock" else ""))
    print()
    print("Whiting: SMS ration (energy at diet E 4.5 kJ/g, range 4.0-5.5) / Andersen & Riis-Vestergaard Wmax")
    for W in (50, 120, 250, 400):
        L = whiting_L(W)
        ann = []
        for q in range(4):
            r = sms_pct_bw_d("whiting", q, W / 1000)
            rk = r * 10 * 4.5
            c = whiting_wmax_kj(L, TQ[q], W)
            ann.append((rk, c))
            print(f"whiting {W:4d} g L {L:4.1f} Q{q+1} T {TQ[q]:5.2f} ration {r:.3f}%BW/d = {rk:5.1f} kJ/kg/d, "
                  f"Wmax {c:6.1f} kJ/kg/d ({c/6.08/10:.2f}%BW/d sandeel), ratio {rk/c:.3f} "
                  f"(E 4.0: {rk/4.5*4.0/c:.3f}; E 5.5: {rk/4.5*5.5/c:.3f})")
        print(f"   annual mean ratio {sum(a for a, _ in ann)/sum(b for _, b in ann):.3f}")
    print()
    print("Saithe SMS rations, %BW/d (no lab Cmax exists):")
    for W in (500, 1500, 3000):
        print(f"saithe {W} g:", [round(sms_pct_bw_d('saithe', q, W / 1000), 3) for q in range(4)])
    print()
    print("Daan (1973) cod daily ration / FB4 Cmax (wet/wet at 4.184 kJ/g). W = 0.00684 L^3.07"
          " (length-weight used by Neuenfeldt et al. 2020) ")
    daan = {10: 0.53, 20: 2.1, 30: 4.8, 40: 8.5, 50: 13, 60: 19, 70: 26, 80: 34, 90: 43, 100: 53}
    for L, R in daan.items():
        W = 0.006838 * L ** 3.07
        out = []
        for T in (7.0, 8.4, 10.0):
            c = cod_cmax_kj(W, T) / E_FB4 * W / 1000  # g/d
            out.append(f"T{T}: {R/c:.2f}")
        print(f"L {L:3d} cm W {W:7.0f} g ration {R:5.2f} g/d = {100*R/W:.2f} %BW/d; ratio " + ", ".join(out))
    print()
    print("Second cod Cmax basis: Jobling (1988) via Neuenfeldt et al. (2020)")
    for W in (67, 300, 567, 1190, 1967, 5000):
        print(f"  W {W:5d} g: " + ", ".join(
            f"T{T}: Jobling {cod_cmax_jobling_kj(W, T):6.1f} vs FB4 {cod_cmax_kj(W, T):6.1f} kJ/kg/d "
            f"(x{cod_cmax_jobling_kj(W, T)/cod_cmax_kj(W, T):.2f})" for T in (5.0, 7.0, 10.0)))
    print("SMS cod ration (energy at diet E 4.184 / 5.0 kJ/g) / Jobling Cmax, quarterly at_50m")
    for W in (300, 1190, 5000):
        vals = []
        num = den = 0
        for q in range(4):
            r = sms_pct_bw_d("cod", q, W / 1000) * 10  # g/kg/d
            c = cod_cmax_jobling_kj(W, TQ[q])
            vals.append(f"Q{q+1} {r*4.184/c:.2f}/{r*5.0/c:.2f}")
            num += r; den += c
        print(f"  {W:5d} g: " + "  ".join(vals) + f"  annual {num*4.184/den:.2f}/{num*5.0/den:.2f}")
    print("Daan cod ration (E 4.184 / 5.0 kJ/g) / Jobling Cmax")
    for L, R in daan.items():
        W = 0.006838 * L ** 3.07
        out = []
        for T in (7.0, 10.0):
            c = cod_cmax_jobling_kj(W, T) * W / 1000  # kJ/d
            out.append(f"T{T}: {R*4.184/c:.2f}/{R*5.0/c:.2f}")
        print(f"  L {L:3d} cm: " + ", ".join(out))
    print()
    print("Daan stomach content 0.00016 L^3 g as % BW:",
          {L: round(100 * 0.00016 * L ** 3 / (0.006838 * L ** 3.07), 2) for L in (20, 40, 60, 80)})


if __name__ == "__main__":
    main()
