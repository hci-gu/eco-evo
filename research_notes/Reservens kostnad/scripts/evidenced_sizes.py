"""Evidenced carrying costs of reserve in model units, against the thresholds of model_reasons.py
(learner) and optimal_fill.py (field fish). Inputs are the literature syntheses in lit_cost.md and
lit_risk.md (tags there); thresholds are copied from tables/model_reasons_stdout.txt and
tables/optimal_fill_stdout.txt. Research only.

Run: python3 -I scripts/evidenced_sizes.py > tables/evidenced_sizes_stdout.txt
"""
# k = extra metabolism at fill 1 as a fraction of RM (negative = saving), from lit_cost.md verdict
K_EVID = {
    "benthivores": (0.0, 0.05, 0.11),     # gadoid liver respiration, PROXY tissue rates [D]
    "whiting": (0.0, 0.05, 0.11),
    "water_column": (0.0, 0.05, 0.11),
    # herring at depth: metabolic 0-0.02 + girth 0.03-0.07 - lift 0.05-0.18
    "planktivores": (0.0 + 0.03 - 0.18, 0.01 + 0.05 - 0.115, 0.02 + 0.07 - 0.05),   # at depth; ~0 at the surface
    # mackerel: metabolic 0-0.03 + girth 0.05-0.10 - lift 0.15-0.50
    "mackerel": (0.0 + 0.05 - 0.50, 0.015 + 0.075 - 0.30, 0.03 + 0.10 - 0.15),
}
K_CRIT = {"planktivores": 1.88, "benthivores": 2.14, "whiting": 1.92, "water_column": 2.51,
          "mackerel": 1.29}
BETA_STAR = {"planktivores": 4.4, "benthivores": 5.9, "whiting": 2.7, "water_column": 61.1,
             "mackerel": 76.9}

print("1. Energy-type carrying cost k (fraction of RM at fill 1): evidence vs the learner's k_crit")
print("   group          k low / central / high      k_crit   central / k_crit")
for g, (lo, c, hi) in K_EVID.items():
    print(f"   {g:13s}  {lo:+.2f} / {c:+.2f} / {hi:+.2f}      {K_CRIT[g]:5.2f}   {c / K_CRIT[g]:+.3f}")
print("   Field fish (optimal_fill B, 20-d gaps): k 0.1 moves the optimum 0.94 -> 0.93, k 1.0 -> 0.87,")
print("   so the evidenced gadoid k (<= 0.11) moves a field optimum by <= ~0.01 [D].")

print("\n2. Hazard-type cost beta (predation hazard at fill 1 / at fill 0, minus 1)")
print("   evidence in target species: none measured within fill 0-1; lean fish swim worse and are")
print("   taken more at the LOW end (cod K 0.54 endurance x0.30 [A]; PROXY smolts RR 1.8-3.0 [F/D];")
print("   PROXY damselfish RR 3.0 [A/D]); in-range cod K 0.68 -> 0.92: Ucrit up, sprint unchanged [A];")
print("   Pacific herring (PROXY) lipid x2: Ucrit unchanged [F]. Sign of beta: <= 0 [J].")
print("   group          beta needed: learner (D1)   field optimum at field mean (D2, 10-20 d gaps)")
for g, b in BETA_STAR.items():
    print(f"   {g:13s}  {b:8.1f}                     ~0.3-1 (whiting dynamic programme only)")

print("\n3. Inertia (the one bird-mechanism leg that transfers to water): a full reserve adds mass")
print("   to accelerate in a fast start. Mass added at fill 1 per lean mass [D, J]: gadoid liver")
print("   lipid ~0.05 (HSI 8 % x 0.6), herring ~0.12-0.14 lipid, mackerel ~0.18 lipid. PROXY gonad")
print("   slope (mosquitofish): -0.37 % Ucrit per % mass [A/D] -> Ucrit -2 / -5 / -7 % if lipid acted")
print("   like gonad mass. Gonad mass is dense and bulges the abdomen; lipid is lighter than water")
print("   and spread in muscle and liver, so this is an upper bracket, not an estimate [J].")
for name, m in (("gadoid", 0.05), ("herring", 0.13), ("mackerel", 0.18)):
    print(f"   {name:9s} mass +{m:.2f} -> Ucrit {-0.37 * m * 100:+.1f} %")
