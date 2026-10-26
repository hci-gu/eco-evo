# Benthivore energetics and diet: proposed library values (resume 223.9, decision 2)

Written by the research subagent, saved by the main agent (2026-10-26). Nothing live was changed. Scripts in
`scripts/`, printed output in `tables/`, library copies in `data/` (`fg_library_proposed.yaml` and the
variants `energy_only`, `diet_only`, `qb3.4`, `qb5.0`). J = judgement.

## 1. Lean body and reserve (`scripts/lean_reserve.py`)

Engine convention (from the code): with `prey_includes_reserve` a predator gets assimilation x
(`energy_content` + fill x ME) as the static quality (energy_gain_mat, gates, pair h), corrected at run time
by (actual reserve per tonne - fill x ME); without the flag the prey's reserve is removed with the eaten
biomass and its energy is lost. Growth (`population_change._charge_growth_to_reserve`) debits
`energy_content` per new tonne and the new biomass dilutes the reserve, so a tonne produced costs ec + s ME.

| Species | Weight | Lean kJ/g | Whole kJ/g | ME (MJ/t) | Basis |
|---|---|---|---|---|---|
| Haddock | 0.384 | 3.8 | 4.6 | 2000 | J, cod proxy (fillet 3.29, Matvaretabellen; liver store, Nanton et al. 2003) |
| Cod | 0.312 | 3.8 | 4.75 | 2000 | Ryberg et al. 2020; Rojbek et al. 2012; Dutil 1995; Lambert & Dutil 1997 |
| Long rough dab | 0.125 | 2.6 | 3.2 | 1000 | MacKinnon 1972 / J |
| Dab | 0.045 | 3.6 | 4.5 | 1500 | J |
| Plaice | 0.038 | 3.6 | 4.6 | 1850 | Fonds et al. 1985; Rijnsdorp 1983 |
| Others | small | 3.5-4.0 | 4.0-5.8 | 1500-1850 | mostly J |

Weighted: lean 3628 MJ/t (3400-3850, J), whole body 4486, ME 1802, implied fill 0.476.

Proposal: `energy_content` 3600, `max_energy_reserve` 1800 (unchanged), `prey_includes_reserve: true`,
`reserve_reference_fill` 0.50 (J; keeps the static prey quality at exactly 4500, so h on predators / seals ->
benthivores stays 176.5 / 21.2; porpoises' 28.6 is a mass ceiling). At run time an eaten tonne is worth
3600 + s x 1800 (4500 at s 0.5, 4896 at 0.72, 4140 at 0.3). A tonne produced at s 0.72 costs 4896 MJ
instead of 5796 (-15.5 %); the mass-balance ceiling ME/ec rises 0.40 -> 0.50 per tick. Side effects, as
for the other two fish groups: the reward energy B ec + R weights the reserve more; the extinction /
local-reward floor (threshold x energy_content) falls 20 %. Flag: witch, lemon sole, weever imply a fill
above 1 (October whole-body peaks; 3.4 % weight).

## 2. The 17.7 % of the budget diet without a pair (`scripts/diet_m2.py`, `tables/diet_categories.csv`)

| Category | Share of consumption | Central destination | Reason |
|---|---|---|---|
| Unidentified fish | 3.91 % | planktivores 0.74, predators 0.06, epifauna 0.20 | Skagerrak cod's identified non-within fish prey (43.7 : 3.7 : 11.8) |
| Cephalopods | 3.18 % | epifauna | sepiolids/octopods taken on the bed; 3 items in one winter haddock survey |
| Krill / hyperbenthos | 2.51 % | epifauna | mysids and small shrimps, same near-bottom search |
| Small demersal fish | 2.38 % | epifauna | gobies, dragonets, sculpins taken with shrimps and crabs (J) |
| Carrion (hagfish) | 1.19 % | infauna 0.71 / epifauna 0.29 | hagfish's live prey |
| Sandeel | 0.17 % | planktivores | small schooling zooplanktivore |
| Other invertebrates | 0.10 % | infauna | |
| Within-group fish | 4.33 % | dropped (renormalised) | cannibal production is in neither M1/M2 nor the Q/B 4.2 energy budget |

Zooplankton stays unchecked (the pool is A17 copepods; winter haddock 0.4 % zooplankton).
New central shares: infauna 0.511 -> 0.544, epifauna 0.209 -> 0.314, planktivores 0.084 -> 0.120,
predators 0.019 -> 0.023. Variants: benthos-heavy 0.633 / 0.259 / 0.088 / 0.020; fish-heavy 0.520 / 0.269 /
0.183 / 0.027. Energy check: 4.2 is ~11,000 MJ/t/yr over 2623 MJ/t; on the redistributed diet (2472
MJ/t) the energy-equivalent mass Q/B is 4.46 (3.6-5.3). 4.2 kept as asked.

## 3. Library keys old -> new (Q/B 4.2, central diet; brackets Q/B 3.4-5.0)

M2 = Q/B x 1.53 x share / prey mean (infauna 93.6, epifauna 7.75, planktivores 3.33, predators 1.20); the
method reproduces today's 0.0291 / 0.1434 / 0.1344 / 0.0852. Full list: `tables/library_keys_old_new.csv`.

| Key | Old | New | Range |
|---|---|---|---|
| benthivorous_fish energy_content | 4500 | 3600 | |
| benthivorous_fish prey_includes_reserve | (absent) | true | |
| benthivorous_fish reserve_reference_fill | (absent) | 0.5 | |
| benthivorous_fish growth_rate (s*) | 0.0021816966 (0.7579) | 0.0020892507 (0.7693) | not affected by Q/B |
| ben -> infauna M2 | 0.029 | 0.037 | 0.030-0.044 |
| ben -> epifauna M2 | 0.141 | 0.26 | 0.21-0.31 |
| ben -> planktivores M2 | 0.14 | 0.23 | 0.19-0.28 |
| ben -> predators M2 | 0.085 | 0.12 | 0.098-0.144 |
| benthic_infauna growth_rate | 0.00070479452 | 0.00071027397 | 0.00070548-0.00071507 |
| benthic_infauna capacity_depth_table | 131.0 / 65.5 | 132.02 / 66.01 | |
| benthic_infauna max_carrying_capacity | 109.9 | 110.8 | |
| mobile_epifauna growth_rate | 0.00060068493 | 0.00068219178 | 0.00064829-0.00071610 |
| mobile_epifauna capacity_depth_table | 14.31/10.77/12.27/17.43/9.38/3.75 | 16.25/12.23/13.93/19.80/10.65/4.26 | x1.08-x1.19 |
| mobile_epifauna max_carrying_capacity | 12.2 | 13.9 | |
| planktivorous_fish growth_rate | 0.0032411316 | 0.0034948014 | 0.0033709-0.0036196 |
| predatory_fish growth_rate | 0.0022805371 | 0.0024165833 | 0.0023271-0.0025077 |
| mareld2 start totals | 337 000 / 27 900 | unchanged | equilibrium sums 336 999 / 27 871 t |

The K tables are refitted by least squares to keep today's start map B*(c) (section 209/221 convention);
without the refit the epifauna start would fall to 24 543 t. The benthivores' g falls because a cheaper
tonne raises s* while the gross budget (0.8214 /yr) is unchanged.

Independent check (not a target): fish share of epifauna production M2/(F + M2) 0.57 -> 0.69, M2/r_gross
0.20 -> 0.30 (Mackinson & Daskalov 2007: crabs 0.82, Nephrops 0.71, infauna 0.04; here infauna 0.036).
Flag: the epifauna's implied r_gross becomes 1.0 /yr, high for a pool that is 60 % Nephrops (P/B ~0.3-0.6).

## 4. budget_gate and engine check

`tools/probes/budget_gate.py` output is identical before and after (it reads only static quality, h,
cost and ration). `scripts/engine_check.py` on the fix76 accounting: energy need 14,515 -> 13,732
MJ/t/yr; infauna-only Q/B 7.98 -> 7.54; engine Q/B on the redistributed diet 5.36 -> 5.07 vs the budget's
4.2 (energy-equivalent 4.46). The remaining ~1.15-1.2x is the per-eat-tick feeding_cost, outside decision
223.9.

## 5. Tests

Full pytest on a scratch copy with the proposed library (CPU): 2 failed, 1330 passed, 163 skipped. Fail:
`tests/test_fgconfig_depth_map.py` (the infauna rows 131 / 65.5 and the preview mean 109.9). Pass only if
the live edit re-derives g and refits K: `test_growth_budget::test_stored_growth_rate_is_the_derived_one_for_the_project`,
`test_habitat::test_manifest_total_is_the_equilibrium_sum`.

## Open

Haddock lean energy and liver store are a cod proxy; the cephalopod share rests on 3 items; the epifauna
r_gross ~1.0 /yr; Q/B 4.2 by mass vs 4.46 energy-equivalent (user's call); the feeding_cost gap.
