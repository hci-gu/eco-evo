# E1: mobile_epifauna (exposed taxa) and norway_lobster (resume 223.12)

Written by the research subagent, saved by the main agent (2026-10-26). Research only. Scripts in `scripts/`
(`split_e1.py` variants central / massbal / codnns, `build_e1.py`, `energy_e1.py`, `budget_gate_e1.py`),
output in `tables/`, copies in `data/` (`fg_library_e1_central.yaml`, `mareld2_e1_central.yaml`, snippet).
The central copy was built on the live library at sha256 77bcbc947419 (with the infauna refuge edits);
re-run `build_e1.py` on the final live library before applying. J = judgement.

## Groups (central variant)

| | mobile_epifauna (exposed) | norway_lobster |
|---|---|---|
| Members | crabs/hermits/Munida 0.65 (2.00 t/km2), northern shrimp 0.18 (0.56), small shrimps 0.10, mysids 0.07 | Nephrops (4.57 t/km2; UWTV 2024 strata x 30 g on mud; flat to ~250 m, ~0 below 300 m) |
| Grid mean | 3.06 t/km2 (11 021 t) | 4.57 t/km2 (16 458 t) |
| Energy as eaten | 3750 (3100-4700) | 3600 (2800-4800, J; decapod AFDW/WW 16.5 % x 20.74; Calocaris ~2.8, Parslow-Williams 1998; was 4000) |
| r_max | 1.11 (0.9-1.3) stock-weighted (221-style sum-of-K value 0.97, J) | 0.34 (0.25-0.5; Hiddink 2019 longevity, M 0.2-0.3, Ecopath P/B 0.35-0.4) |
| Trawl F, bf100 | 0.209; 0.004747 | 0.0874; 0.001985 |
| M2 | benthivores 0.699 + predators 0.094 | benthivores 0.001 |
| growth_rate | 0.0013034246575 (r_gross 1.903) | 0.00023356164384 (0.341) |
| Equilibrium share | 0.474 | 0.741 |
| capacity_depth_table | 81: 6.13, 123: 5.70, 173: 5.84, 221: 11.11, 272: 8.72, 387: 4.30 | 81: 8.78, 123: 4.53, 173: 6.75, 221: 10.87, 272: 2.87, 300: 0.19 |
| max_carrying_capacity | 6.7 | 6.3 |
| Start (+/-25 %) | 11 000 t (8250-13 750) | 16 400 t (12 300-20 500) |

Pairs (Type II, no dark_ratio): benthivores -> exposed M2 0.699, assim 0.8, h 138.4, a 0.44, f 0.63
(0.45-0.8), half-saturation 0.026 t/km2; predators -> exposed M2 0.094, h 147.1, a 0.33, f 0.33 (0.2-0.45);
benthivores -> Nephrops M2 0.001, h 132.9 (do not round down: at 132.8 it becomes the "best" pair), a 0.44,
f 0.02 (0.07 x size window 0.29, J), x 0.84 of the ceiling; predators -> Nephrops unchecked. The 223.10
redistributed prey (cephalopods, krill/hyperbenthos, small demersal fish, the epifauna slices of
unidentified fish and carrion) all go to the exposed group. Observes: benthivores both, predators exposed
only. No knock-on on any other group's g, s* or gross.

## Independent checks (fish flow / (P/B x B))

| | central | massbal | codnns | bound |
|---|---|---|---|---|
| Crabs | 94.7 % (fails) | 71.8 % | 60.3 % | 60-92 % |
| Northern shrimp | 27 % | 27 % | 66 % | 15-39 % |
| Nephrops | 0.3 % | 0.3 % | 11 % | 65-94 % (North Sea-wide) |
| Exposed r_gross vs P/B 1.04 | x1.83 | x1.63 | x1.78 | - |

Margins are thin: the budgeted fish flow on the exposed group is 96 % (central) / 93 % (massbal) of its
maximum surplus net of trawling; each fish group's demand at its ceiling is 12-17 t/km2/yr; on Nephrops the
benthivores could demand 16.8 t/km2/yr against a surplus of 0.30. E1 protects Nephrops and the group total;
it does not make the fish's epifauna intake sustainable (223.3).

## Open (user's call)

1. Central vs massbal: massbal applies the 221 mass-balance convention to the 223.10 ration (crabs 2.64,
   small shrimps/mysids 0.545, a member "cephalopods and small demersal fish" 0.23 t/km2, matching M&D's
   independent North Sea values); r_max 1.13, M2 0.539 + 0.072, growth_rate 0.0011924658, K 7.17 / 6.68 /
   6.76 / 11.04 / 8.83 / 4.89, start 14 300 t, bf100 0.004142, f 0.62 / 0.29. Passes the crab bound;
   reopens 223.10's "stocks unchanged". Only the central copy is ready to apply.
2. benthivores -> Nephrops: checked tiny saturated pair (as copy) vs unchecked (221's rule for tiny pairs)
   vs checked at the northern North Sea cod-diet M2 0.038; f 0.02 vs 0.07.
3. Exposed r_max 1.11 vs 0.97. 4. Nephrops energy 3600. 5. Small shrimps : mysids 0.3 : 0.2.
6. Depth K cannot represent the mud patches.
