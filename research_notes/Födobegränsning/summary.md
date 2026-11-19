# Food limitation at today's densities: synthesis (2026-11-07)

Question (user): can the food be made to limit the fish at today's densities from field evidence
(steps 1-2 of the plan after the fix90 growth diagnosis, "Krill i Skagerrak" a_revisit/
fix90_growth/ section 7, option iii)? Research only; no library value changed. Parts: `benthic/`
and `fish_krill/` (research subagents, saved by the main agent), `model/` (saturation over each
predator's whole menu at the start densities, shared handling).

## Model side (model/tables/saturation_stdout.txt)

| predator | X = sum h a B_vis | share of ceiling while foraging | every prey x ... for -10 % intake | ... to halve it |
|---|---|---|---|---|
| benthivores | 193 | 0.995 | 0.044 | 0.005 |
| whiting | 28 | 0.966 | 0.24 | 0.033 |
| water column | 18 | 0.947 | 0.32 | 0.050 |
| planktivores | 10 | 0.909 | 0.45 | 0.083 |
| mackerel | 1.5 | 0.61 | 0.78 | 0.28 |

## Field side

1. **Benthivores** (benthic/): the field total bed intake is saturated too. The feeding level comes
   from foraging time, not a.
   - The Irish Sea trawling gradient (Johnson 2015) gives a lower bound on a 6.5-19x below the
     library's 0.036.
   - The earlier field-implied a 0.002-0.01 assumed tau = 1 and was circular.
   - The library a stands.
2. **Fish prey** (fish_krill/): saturated locally (1-6.5 km), food-limited at stock scale (whiting
   growth vs forage fish, Lauerburg 2018; saithe vs pout, Cormon 2016).
   - The stock-scale limit comes through partial predator-prey overlap: K grows with the averaging
     scale. It does not come from a smaller local a.
   - The model's 1 km cells and movement already carry that scale.
3. **Per-prey-group windows.** Whiting's plateau per prey group (0.06-0.22 of Cmax each) is
   independent of the other groups and of prey density (Rindorf & Gislason 2005). Shared handling
   (242), where all prey compete for one denominator, was "very poorly" supported there.
   - This bears on the krill squeeze of section 248: with additive windows krill would not crowd
     out the fish prey.
   - A structural question of its own, not food limitation.
4. **Krill:** no functional response of any fish on euphausiids exists (gap).
5. **Adult growth.** No significant density-dependent growth in North Sea cod, haddock or Norway
   pout (Lorenzen & Enberg 2002, via "Tathetsberoende och interferens hos torskfiskar", a-umlaut).

## Conclusion

The field does not support making the food limit the box's fish through the pairs' a: the bed
pairs are saturated in the field as well, and the fish-prey limit is a stock-scale overlap effect.
Population regulation of these species sits in recruitment, outside the adults' food, which is
also the fix90 diagnosis (r_max paid through adult feeding at fill 1). Option iii is therefore not
the route. What the evidence points to is option ii, the open-population form:
- r_max out of the adults' derived g (production pays M1 + M2);
- recruitment as a flux into the juvenile compartment from outside the box (an open nursery, the
  recruits spawned elsewhere, 223.37), from survey recruit data;
- today's state as the equilibrium of that flux against the losses.

Separately, per-prey-group windows vs shared handling are worth their own investigation (the krill
squeeze).
