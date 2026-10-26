"""Energy need of a tonne of benthivores in the engine, before / after the proposal.

Usage: python3 -I engine_check.py

budget_gate.py only tests break-even at the Holling ceiling with the static prey qualities;
the proposal leaves those unchanged (static quality 4500, same h, same cost), so its output
cannot move. This script redoes the accounting of research_notes/"Bentivorernas energibudget"
(summary 1.1, probe year 1 of fix76 gen 10: action mix eat 0.957 / move 0.041 / rest 0.002,
reserve fill 0.722, action costs 10 414 MJ/t/yr, which the proposal does not change) with the
new production charge ec + s ME and the new derived g (build_library.py).
"""
YEAR = 1460.0
COSTS = 10414.0          # action costs at the realised mix (unchanged: feeding_cost is out of scope)
S = 0.722                # realised reserve fill in the probe
U = 0.5
CASES = {"today": dict(ec=4500.0, me=1800.0, g=0.0021816966),
         "proposed": dict(ec=3600.0, me=1800.0, g=0.0020892507356)}
Q_INF = 0.7 * 2600.0     # assimilated MJ per t infauna
Q_DIET_ENGINE = 2709.0   # engine quality of the central redistributed diet (diet_m2.py)
Q_DIET_ENGINE_OLD = (0.511 * 1820 + 0.209 * 3000 + 0.084 * 5700 + 0.019 * 4202) / 0.823
for name, c in CASES.items():
    p = c["g"] * (S - U) * YEAR                 # t produced per t per yr at s = 0.722
    unit = c["ec"] + S * c["me"]                # MJ per t produced (debit + reserve dilution)
    need = COSTS + p * unit
    print(f"{name:9s}: P {p:.3f} /yr x {unit:.0f} MJ/t = {p * unit:6.0f}; need {need:6.0f} MJ/t/yr"
          f" -> infauna-only Q/B {need / Q_INF:.2f}, engine Q/B on the redistributed diet "
          f"{need / Q_DIET_ENGINE:.2f} (old checked-pool mix {need / Q_DIET_ENGINE_OLD:.2f})")
print("Budget Q/B: 3.48 today, 4.2 proposed (energy-equivalent on the redistributed diet 4.46).")
print("Residual gap = the per-eat-tick feeding_cost (2.0 x RM), outside decision 223.9(2).")
