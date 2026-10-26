# Field feeding level and satiation (resume 223.12, decision 2)

Written by the research subagent, saved by the main agent (2026-10-26). Research only. Scripts and tables
in `scripts/` and `tables/` (A_* field feeding levels, B_* mechanisms, unprefixed synthesis). J = judgement.

## Headline

1. The "0.3 of Cmax" premise was against the engine's own ceiling. Against species Cmax at ambient
   temperature: whiting 0.25 (0.10-0.35, best constrained), cod 0.53 (0.43-0.75), haddock 0.22 (0.15-0.31),
   plaice ~0 in Q1 and 0.5-0.9 in the feeding season, saithe no Cmax. The benthivore group ceiling (108
   kJ/kg/d) is 1.3-1.6x cod's literature Cmax (haddock x1.3 assumption, flatfish values, 8.4 C layer mean).
   Engine realised feeding: benthivores 0.54 of their ceiling (eat 0.96 x hunger 0.56), ~0.7-0.85 of cod
   Cmax, 1.5-2.5x the group's field level (J ~0.36); predators 0.43-0.50 vs field 0.25-0.49.
2. The bigger defect is the missing density response: x = a h f N = 44-266 on the benthic pairs, so a
   foraging haddock fills a 6 h tick's ceiling in 1.4 min on infauna. Benthic pairs: the problem is in a
   (1-2+ orders of magnitude above field-consistent rates); a time window cannot explain it because the
   stomach buffers 1-4 days of meals (gut_buffer_window.py). Fish-prey pairs: a about right; the plateau is
   a capture window (Rindorf & Gislason 2005) on the whole response. Type II with h = 1/Cmax is supported
   (Jeschke et al. 2002: the digestion-limited form).
3. Mechanisms ranked: exposure/detectability of benthic prey (in a); capture window on fish prey; seasonal
   feeding window (Q1 empties: whiting 0.45, cod/haddock 0.16-0.18, plaice 98.7 % empty); armour on the
   ceiling (h x 1.5-2, J); satiation (the hunger gate sets the engine's level, no field form); patchiness
   and interference negligible on saturated pairs.

## Recommended (not applied)

- P1 benthic pairs: an exposure factor e inside a (a_eff = a_enc x e), field-consistent e ~0.002-0.01
  (infauna), 0.007-0.03 (benthivores -> epifauna), order of magnitude only; predators -> epifauna a ~0.007
  per whole predator tonne (diet test: encounter a's predict 77 % epifauna vs 8 % in the field; ~2.7x of the
  overstatement is bed access by non-bed members, ~17x exposure). No direct mechanistic measurement of e
  (gap); route: exposed-item density x yield per item (Amphiura arm cropping, siphon nipping). Test:
  Hiddink 2016 intake fall under trawl depletion favours P1.
- P2 fish-prey pairs: a capture window tau on the whole response (a and 1/h), whiting ~0.1-0.2 per prey
  group; must come with or after P1 (208.5: lowering one pair's ceiling pushes an energy maximiser to the
  others).
- P3 a seasonal feeding window per group (monthly multiplier; budget paid over the season as herring).
- P4 ceiling: armour on shelled benthos; revisit the benthivore Cmax (bottom water 7-8 C).
- Once the benthic pairs are unsaturated, growth_budget's s*/ideal intake (paid at the ceiling) must be
  re-expressed at realised intake.

Gaps: no adult haddock, saithe or flatfish Cmax; no offshore flatfish rations; no measured infauna exposure
per contact; no sub-km CV; no armour factor for brittle stars or bivalves.
