"""Does the booked rest water column -> mackerel group pair pay at a higher mackerel density?

Usage: python3 -I rest_on_mackerel_vs_density.py <data/fg_library_mack.yaml> [B1 B2 ...]

For each mackerel group density B (t/km2, grid mean, in one cell) prints:
  * the booked M2 of the pair: the predators' diet flow 0.01802 t/km2/yr (split_fields_mack.py) / B;
  * the rest's return per unit eat share on mackerel, a f B E / (1 + a h f B) (MJ/t/tick), against
    the rest's other pairs at the library's grid-mean prey densities (planktivores 3.3 t/km2 at the
    box J/B 0.38, zooplankton 35 000 t / 3600 km2, benthivores 1.53 at J/B 0.15, whiting 0.446);
  * the unpaid share of the mackerel group's gross budget if the pair is never eaten: M2 / gross.
The eat share is linear and exclusive per cell (223.31), so a prey is eaten only where it pays best.
Density per cell can exceed the grid mean (schools): the B at which mackerel matches each other pair
is printed too. No model output is read.
"""
import sys

import yaml

lib = yaml.safe_load(open(sys.argv[1], encoding="utf-8-sig"))
I, S = lib["interaction_definitions"], lib["species_definitions"]
Bs = [float(x) for x in sys.argv[2:]] or [0.138, 0.25, 0.5, 1.0, 2.0, 4.0]
FLOW = 0.01802                      # t/km2/yr, the rest's diet flow on mackerel + horse mackerel
M1, R, M2_MAMMALS = 0.159, 0.255, 0.0098 + 0.0063  # field-density group (228)


def ret(pred, prey, B, E, jb=None):
    p = I[f"{pred}_preys_on_{prey}"]
    a, h = float(p["max_intake_rate"]), float(p["handling_time"])
    if "stage_access" in p:
        fj, fa = p["stage_access"]
        f = fj * jb + fa * (1 - jb)
    else:
        f = float(p.get("accessible_fraction", 1.0))
    x = a * f * B
    return x / (1 + x * h) * E


E = {"planktivorous_fish": 7125.0, "zooplankton": 4500.0, "benthivorous_fish": 4500.0, "whiting": 4800.0,
     "mackerel": 7898.5}
alt = {"planktivorous_fish": ret("predatory_fish", "planktivorous_fish", 3.3, E["planktivorous_fish"], 0.38),
       "zooplankton": ret("predatory_fish", "zooplankton", 35000 / 3600, E["zooplankton"]),
       "benthivorous_fish": ret("predatory_fish", "benthivorous_fish", 1.53, E["benthivorous_fish"], 0.15),
       "whiting": ret("predatory_fish", "whiting", 0.446, E["whiting"])}
print("rest water column, return per unit eat share at grid-mean prey (MJ/t/tick): "
      + ", ".join(f"{k} {v:.1f}" for k, v in alt.items()))
print(f"\n{'B t/km2':>8} {'tonnes':>7} {'M2 booked':>9} {'gross /yr':>9} {'unpaid %':>8} {'return':>7}")
for B in Bs:
    m2 = FLOW / B
    gross = R + M1 + M2_MAMMALS + m2
    print(f"{B:8.3f} {B * 3600:7.0f} {m2:9.4f} {gross:9.3f} {100 * m2 / gross:8.1f} "
          f"{ret('predatory_fish', 'mackerel', B, E['mackerel']):7.2f}")
print("\nmackerel density in ONE cell at which the pair matches the rest's other pairs:")
for k, v in alt.items():
    lo, hi = 1e-3, 1e4
    if ret("predatory_fish", "mackerel", hi, E["mackerel"]) < v:
        print(f"  {k:20s} never (the pair's ceiling {E['mackerel'] / float(I['predatory_fish_preys_on_mackerel']['handling_time']):.1f} < {v:.1f})")
        continue
    for _ in range(100):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if ret("predatory_fish", "mackerel", mid, E["mackerel"]) < v else (lo, mid)
    print(f"  {k:20s} {hi:8.2f} t/km2")
