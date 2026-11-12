"""Krill rations from Onsrud et al. (2004) Table 3 (inner Oslofjord, 120 m site, Nov 2000 and Mar 2001;
deep water homogeneous below ~30 m) [F: `Vitlingens fiskbyten/data/step2_lit/vertical/onsrud2004.pdf`, p. 1287].

Table 3 values used (day / night): % stomachs with krill; krill per stomach WITH krill.
Krill content per fish (all stomachs) = mean over day and night of (share with krill x krill per
stomach) x the fresh mass of one adult M. norvegica (0.15 / 0.23 / 0.30 g WW; `Krill i Skagerrak`).
State of digestion 1.9-5 (Table 3): counted krill are partly digested, so fresh mass overstates
the content (upper side).

Ration:
  whiting and Norway pout: the ICES gadoid GER (WGSAM 2024 Table 3-7, whiting set; pout proxy),
    24 rho e^(0.081 T) L^1.35 E^-0.85 sqrt(S) on krill-bearing stomachs (S = krill per stomach x mass),
    times the share with krill. E 5.0 kJ/g, T 7 C (Oslofjord deep water 6-8 C [J]).
  herring: no GER parameters; 1-2 stomach turnovers per day [J; the earlier note's assumption].
Fish sizes are not given for the stomach fish in the paper; assumed [J]: whiting 21-28 cm
(one 21 cm fish held 40 krill), pout 14-18 cm, herring 25-30 cm (stomachs taken from fish > 25 cm).
"""
import math

T3 = {  # species: month: ((with_day, with_night), (n_day, n_night))
    "whiting": {"Nov": ((42.31, 35.19), (1.82, 3.47)), "Mar": ((75.51, 79.69), (3.95, 5.92))},
    "Norway pout": {"Nov": ((59.09, 44.07), (1.54, 1.54)), "Mar": ((75.00, 47.62), (2.17, 2.10))},
    "herring": {"Nov": ((31.03, 17.24), (1.90, 2.20)), "Mar": ((50.00, 20.00), (5.13, 1.50))},
}
FULL = {"whiting": {"Nov": (2.67, 3.00), "Mar": (3.28, 3.25)},
        "Norway pout": {"Nov": (3.27, 2.93), "Mar": (4.00, 3.62)},
        "herring": {"Nov": (2.91, 2.29), "Mar": (3.43, 2.70)}}
SIZE = {"whiting": [(21, 0.00603, 3.077), (28, 0.00603, 3.077)],
        "Norway pout": [(14, 0.0065, 3.17), (18, 0.0065, 3.17)],
        "herring": [(25, 0.0075, 3.1), (30, 0.0075, 3.1)]}  # herring L-W [J]: 25 cm ~165 g
RHO, LAM, DEL, MU, E, T = 0.00171, 1.35, 0.081, 0.85, 5.0, 7.0

for sp, months in T3.items():
    for mo, ((wd, wn), (nd, nn)) in months.items():
        P = (wd + wn) / 200.0
        n_with = (wd / 100 * nd + wn / 100 * nn) / (2 * P)
        for L, a, b in SIZE[sp]:
            W = a * L ** b
            out = []
            for wk in (0.15, 0.23, 0.30):
                S_with = n_with * wk
                content = P * S_with / W * 100
                if sp == "herring":
                    rat = (content * 1.0, content * 2.0)
                    out.append(f"{content:.2f}% -> {rat[0]:.2f}-{rat[1]:.2f}")
                else:
                    R = 24 * RHO * math.exp(DEL * T) * L ** LAM * E ** (-MU) * math.sqrt(S_with) * P
                    out.append(f"{content:.2f}% -> {100*R/W:.2f}")
            print(f"{sp:11s} {mo} L {L} cm W {W:4.0f} g: with krill {100*P:.0f} %, {n_with:.2f} krill per "
                  f"krill-bearing stomach, fullness d/n {FULL[sp][mo]}; krill content -> ration %BW/d "
                  f"(0.15/0.23/0.30 g): " + " | ".join(out))
