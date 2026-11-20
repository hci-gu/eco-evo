"""Why a rest payoff lowers T but not fill: the extra foraging a full reserve costs vs the risk it saves.

Usage: python3 -I fill_tradeoff.py > ../tables/fill_tradeoff_stdout.txt

Inputs [S]: feeding level (share of the ceiling) needed to hold fill 0.1 vs 1.0 under the model's growth law
(research_notes/"Mattnad hos rovfisk" summary 1.3, a-umlaut): whiting 0.275 -> 0.388, water column
0.350 -> 0.435; share of the ceiling while foraging S (fix105 yardstick, rest_payoff_stdout.txt B): whiting
0.826, water column 0.779. Rest payoff per unit rest share (candidate_forms_stdout.txt): form V incl. today's
mammal floors whiting 0.0446 /yr, VM central 0.0705; water column 0.0025 (today) / 0.0016 (VM).
Production at stake: somatic_production P (library): whiting 0.61, water column 0.42 /yr; the growth law
pays g (s - u), so at fill 0.1 (u 0.07) production is (0.1 - 0.07)/(1 - 0.07) = 0.03 of that at fill 1.
"""
CASES = {"whiting": (0.275, 0.388, 0.826, {"V": 0.0446, "VM central": 0.0705, "VM low-floor": 0.1947}, 0.61, 0.07),
         "water column": (0.350, 0.435, 0.779, {"today": 0.0025, "VM low-floor": 0.0056}, 0.42, 0.07)}
for g, (f_lo, f_hi, S, pays, P, u) in CASES.items():
    t_lo, t_hi = f_lo / S, f_hi / S
    dT = t_hi - t_lo
    lost_P = P * (1 - (0.1 - u) / (1 - u))
    print(f"{g}: foraging share to hold fill 0.1 {t_lo:.3f}, fill 1.0 {t_hi:.3f}; extra {dT:.3f} of time")
    print(f"  production given up at fill 0.1 vs 1.0: {lost_P:.3f} /yr of biomass")
    for k, v in pays.items():
        print(f"  risk saved by resting that extra time, form {k:12s}: {dT * v:.4f} /yr "
              f"({100 * dT * v:.2f} %/yr) = {dT * v / lost_P:.4f} of the production at stake")
print("-> at any r in the evidence range the fish keeps its reserve full: the payoff lowers T, not fill.")
