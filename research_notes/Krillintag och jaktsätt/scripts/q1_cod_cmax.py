"""Q1: Hansson et al. 1996 cod Cmax (FB4 CEQ 2: a 0.099, b -0.195, CQ 2.41, Topt 13.7, Tmax 21;
g wet prey / g wet fish / d, fish-prey basis) at the sizes/temperatures of the Q1 field rations,
and the ratio ration / Cmax. Same function as research_notes/"Fiskgrupper i rutan"/planktivores/
scripts/cod_cmax_proxy.py. Run: python3 -I q1_cod_cmax.py"""
import math
A, B, Q, TOPT, TMAX = 0.099, -0.195, 2.41, 13.7, 21.0

def f_t(t):
    v = (TMAX - t) / (TMAX - TOPT)
    z = math.log(Q) * (TMAX - TOPT)
    y = math.log(Q) * (TMAX - TOPT + 2)
    x = (z ** 2 * (1 + (1 + 40 / y) ** 0.5) ** 2) / 400
    return v ** x * math.exp(x * (1 - v))

def cmax(w, t):
    return 100 * A * w ** B * f_t(t)   # %BW/d

def cmax_jobling(w, t):
    """Jobling 1988 as used by Neuenfeldt et al. 2020 eq. 2 (repo: research_notes/"Exponering av
    bottenbyten"/scripts/C_cmax_bottom.py): 0.223 e^(0.104T - 0.000112T^3) W^0.802 kJ/d, converted
    to %BW/d at the 4.184 kJ/g convention (the capelin-based function; the unit reading is the repo's)."""
    import math as _m
    kj = 0.223 * _m.exp(0.104 * t - 0.000112 * t ** 3) * w ** 0.802
    return 100 * kj / 4.184 / w

cases = [
    # label, W g, T C, total ration %BW/d (list), krill share of diet by weight
    ("Barents cod Rybachy Bank Jun 1988 (Fortunatova, Bajkov, Santos)", (1000, 2100), 1.5, (2.20, 2.10, 0.64), 0.841),
    ("Barents cod Murman shallow Aug 1989 (3 methods)", (1000, 2100), 2.45, (0.33, 0.28, 0.48), 0.484),
    ("Barents cod Demidov Bank Jul 1988, Themisto 74.1 %W (3 methods)", (1000, 2100), 1.3, (3.76, 3.61, 0.74), 0.741),
    ("Barents cod Goose Bank Jul-Aug 1989, Themisto 64.5 %W (3 methods)", (1000, 2100), 1.75, (0.91, 0.85, 0.99), 0.645),
    ("Pacific hake 500 mm, Washington-Oregon Jul 1967 [cod proxy, J]; krill share 0.72 (Oregon)", (700, 1000), 8.2, (2.5,), 0.72),
    ("Pacific hake 500 mm, Washington-Oregon Jul 1967 [cod proxy, J]; krill share 0.90 (Washington)", (700, 1000), 8.2, (2.5,), 0.90),
]
for lab, ws, t, rations, share in cases:
    print(lab)
    for w in ws:
        c = cmax(w, t)
        cj = cmax_jobling(w, t)
        print("  W %5d g T %.2f C: Hansson F(T) %.3f Cmax %.2f %%BW/d; Jobling Cmax %.2f %%BW/d"
              % (w, t, f_t(t), c, cj))
        for r in rations:
            print("    total ration %.2f -> %.2f (H) %.2f (J) of Cmax; prey part %.2f %%BW/d -> %.2f (H) %.2f (J)"
                  % (r, r / c, r / cj, r * share, r * share / c, r * share / cj))
