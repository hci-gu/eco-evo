"""B_feeding_level.py - realised feeding level of wild demersal fish
(field ration / Cmax, and stomach fill / satiation fill), plus the size of
the Beddington-DeAngelis interference term against saturation.

Inputs (sources in tables/B_mechanisms_evidence.csv):
- Cmax: Hansson et al. 1996 cod function (FB4 row), Kitchell f(T);
  haddock = 1.3 x cod function (central judgement of the earlier
  benthivore bioenergetics note, range 1.0-2.0); whiting 2.45 and saithe
  2.3 %BW/d at 8.4 C from the earlier GER construction
  (Fiskgrupper i rutan/predators/evidence_A_physiology.md).
- Field rations: Daan 1973 (via Durbin 1983), Durbin et al. 1983,
  Greenstreet et al. 1997 (secondary), SMS key run 2023 (ICES WGSAM),
  North Sea whiting 1992-93 (abstract).
- Stomach fill: Ursin et al. 1985 Table 1 (North Sea, ICES data set 9;
  N. North Sea data set 6); satiation fill = the fill that sustains FB4
  Cmax under the WGSAM square-root evacuation (B_evacuation.py), so the
  feeding level from fill is sqrt(S_wild / S_cmax) (square-root law).
- Engine ceiling for benthivores: 138 kJ/kg/d (~1.6 x cod Cmax at 7 C),
  from Bentos atkomlighet/evidence_B_gadoid_foraging.md.

Run: python3 -I B_feeding_level.py > ../tables/B_feeding_level_stdout.txt
"""
import math


def kitchell_f(T, CQ=2.41, CTO=13.7, CTM=21.0):
    V = (CTM - T) / (CTM - CTO)
    Z = math.log(CQ) * (CTM - CTO)
    Y = math.log(CQ) * (CTM - CTO + 2)
    X = Z**2 * (1 + (1 + 40 / Y) ** 0.5) ** 2 / 400
    return V**X * math.exp(X * (1 - V))


def cmax_cod_pct(W, T):
    return 100 * 0.099 * W**-0.195 * kitchell_f(T)


def r_prime_cod(L, T, E=4.2):
    return 0.00224 * math.exp(0.083 * T) * L**1.30 * E**-0.85


def main():
    print("## field ration / Cmax (%BW/d)")
    print("species  W(g)  T   ration(lo-hi)   Cmax   level(lo-hi)   source")
    rows = [
        ("cod 30cm", 270, 7.0, 1.71, 1.71, None, "Daan 1973 via Durbin 1983"),
        ("cod 40cm", 640, 7.0, 1.28, 1.28, None, "Daan 1973"),
        ("cod 50cm", 1250, 7.0, 1.03, 1.03, None, "Daan 1973"),
        ("cod 60cm", 2160, 7.0, 0.85, 0.85, None, "Daan 1973"),
        ("cod >30cm", 1000, 6.8, 0.9, 1.5, None, "Durbin et al. 1983 bounds"),
        ("cod", 500, 7.0, 1.44, 1.44, None, "Greenstreet et al. 1997 [S], W assumed (J)"),
        ("cod", 500, 7.5, 1.14, 1.77, None, "SMS key run 2023 quarters"),
        ("haddock", 500, 7.0, 1.30, 1.30, "had", "Greenstreet et al. 1997 [S]; Cmax 1.3x cod (J)"),
        ("whiting", 200, 8.4, 0.78, 1.24, 2.45, "SMS 2023 quarters; Cmax GER construction"),
        ("whiting", 200, 8.4, 0.41, 1.75, 2.45, "North Sea 1992-93 [A]"),
        ("saithe", 1070, 8.4, 0.43, 1.25, 2.30, "SMS 2023 quarters; Cmax GER construction"),
    ]
    for sp, W, T, lo, hi, cm, src in rows:
        if cm is None:
            cmx = cmax_cod_pct(W, T)
        elif cm == "had":
            cmx = 1.3 * cmax_cod_pct(W, T)
        else:
            cmx = cm
        print(f"{sp:<9s}{W:6d} {T:4.1f}  {lo:5.2f}-{hi:<5.2f}  {cmx:6.2f}   {lo / cmx:4.2f}-{hi / cmx:<4.2f}   {src}")
    print("# haddock with Cmax 1.0x / 2.0x cod: "
          f"{1.30 / cmax_cod_pct(500, 7.0):.2f} / {1.30 / (2 * cmax_cod_pct(500, 7.0)):.2f}")

    print("\n## the same rations against the engine benthivore ceiling (~1.6 x cod Cmax at 7 C)")
    for sp, W, r in (("cod 40cm Daan", 640, 1.28), ("cod Greenstreet", 500, 1.44), ("haddock Greenstreet", 500, 1.30)):
        print(f"{sp:<20s} {r / (1.6 * cmax_cod_pct(W, 7.0)):.2f} of the engine ceiling")

    print("\n## stomach fill of wild North Sea cod (Ursin et al. 1985 Table 1) vs satiation fill")
    T = 7.0
    data = {"9 North Sea ICES": [(429, 6.33), (852, 14.7), (1951, 36.4), (5718, 111)],
            "6 N. North Sea": [(386, 5.22), (829, 10.1), (1575, 25.0), (2797, 35.5)]}
    print("set               W(g)  S(g)  S%BW  S_cmax%BW  level=sqrt(S/S_cmax)")
    for k, pts in data.items():
        for W, S in pts:
            L = (W / 0.01) ** (1 / 3)
            R = r_prime_cod(L, T)
            C = cmax_cod_pct(W, T) / 100 * W
            Sc = (C / 24 / R) ** 2
            print(f"{k:<17s}{W:6d} {S:6.1f} {100 * S / W:5.2f} {100 * Sc / W:9.2f} {math.sqrt(S / Sc):9.2f}")
    print("# mean fills include empty stomachs; pooled multi-season data; GER at 7 C fish prey")

    print("\n## Beddington-DeAngelis interference vs saturation (benthivores, w = 0.45 km2/t)")
    print("B_pred t/km2   wB     x=ahN    intake ratio (1+x)/(1+x+wB)")
    for B in (0.5, 1.53, 3.0, 6.0):
        for x in (1, 10, 100, 500):
            wB = 0.45 * B
            print(f"{B:10.2f} {wB:6.2f} {x:8d} {(1 + x) / (1 + x + wB):12.4f}")


if __name__ == "__main__":
    main()
