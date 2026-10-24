# Why the model's gadoids sit far below the trawl survey

Question: the model's gadoids start at 2500 t = 0.69 t/km2; the GOV survey gave 4.2 t/km2 for
benthivores + predators (2021-25). Where does the factor ~6 come from? [D] unless tagged.

## 1. Origin of 0.69
- 2500 t is the "Codfish" row of the initial-biomass table in `Mareld simulator.pdf` (the project's
  design slides). No source is given there.
- It was questioned before: `reports/Fiskeridödlighet i Skagerrak idag.md` ("Startbiomassan för
  gadoiderna kan vara fyra gånger för låg": IBTS wing 2.7 t/km2 around the grid, 3.3 Skagerrak-wide),
  resume 161.10. The user kept 2500 t (resume 166.3).
- Since then the model is calibrated on it: gadoid K = 0.69 / (1 - 0.12/0.37) = 1.02 t/km2
  (resume 185), the seal -> gadoid pair (predation_mortality 0.12; 0.03 "would match the IBTS
  9700 t", 166.3), the porpoise rates (206), and the prey M2 the gadoids pay.

## 2. Decomposing the factor
| Step | t/km2 | Factor to the model |
|---|---|---|
| GOV, benthivores + predators incl. blue whiting, 2021-25 (the "x6") | 4.2 | 6.1 |
| Gadoids only (the model group; no flatfish, rays, mackerel, spurdog, anglerfish, blue whiting) | 3.1 | 4.5 |
| Gadoids only, mean of 2001-25 (2021-25 is a peak) | 1.7 (median 1.4) | 2.5 (2.0) |
| Gadoids only, 2001-20 | 0.9-2.1 per year, mean 1.4 | 2.0 |

GOV densities are swept area at wing spread, q = 1, depth-stratified to the grid (per-year values
from `long_series.py` kg/h x the 2021-25 kg/h -> t/km2 ratio 0.0068; `tables/long_series_by_year.csv`).
Per year: 1.0-1.9 in 2001-15, 0.9-2.1 in 2016-21, then 2.6 (2022), 3.3, 3.8, 4.5 (2025): haddock
(x40 from 2018-19) and saithe.

**The like-for-like gap over the long run is ~2x, not 6x.** Of the 6x, ~1.35x is the group
definition and ~1.8x the 2021-25 peak.

## 3. The remaining ~2x is inside the catchability uncertainty
- Door spread instead of wing (/3.75): long-run 0.37, 2021-25 0.83 - the model then sits inside the
  survey. But F = catch/B becomes ~0.5/yr, against ICES' 0.12 (the F report).
- Fraser et al. 2007 GOV q (cod 0.21, saithe 0.17-0.43, haddock 0.75, whiting 1.7 on the wing basis;
  `../Torskfiskarnas artsammansättning/gadoid_species_mix.md` section 4): x2-2.7 up, i.e. 2021-25
  6-8.5 t/km2.
- Wing q = 1 sits between them and is the convention used so far.

## 4. Independent anchors
| Anchor | t/km2 | Note |
|---|---|---|
| Skagerrak gadoid catch 2023-25 (13.8 kt/yr incl. discards) / F 0.12 | ~3.5 | over ~32 600 km2 (F report); the peak years |
| Landings in rectangle 45G0 2010-20 (0.18 t/km2/yr, MKB) / F 0.12 | ~1.5 | landings only, Norway partly missing: a floor; pre-peak |
| North Sea Ecopath 1991 / 1981, gadoid groups (Mackinson & Daskalov 2007; Christensen 1995) | 1.96 / 2.71 | North-Sea-wide, MSVPA-based [F, `../Restdödlighet M0 hos torskfiskar/ecopath_models.md`] |
| Density-dependent balance of frozen fix55 + F 0.12 (F report) | ~2.8 | indicative |

They span 1.5-3.5 t/km2, with the pre-peak anchors at 1.5-2 and the 2021-25 ones at ~3-3.5.

## 5. Reading
- The model's 0.69 is low by ~2x against the long-run survey level and by ~4-5x against 2021-25.
  It agrees with the survey only at door spread, which contradicts the ICES F.
- Which level is "today" is a choice: 2021-25 (~3 t/km2, a haddock year-class pulse; ICES
  assessments would tell whether it persists) or the long-run 1.4-2.
- Raising the level means recalibrating what was fitted to 0.69: gadoid K (1.02 -> K = B*/(1 -
  F/r)), the seal -> gadoid and porpoise pairs, the gadoid prey demand (consumption x2-4: benthos M2
  0.0048 is small next to trawling 0.2; the herring pair predation_mortality 0.30/yr cannot scale
  with it - the 2020-25 stomach data put fish at ~7 % of the gadoid diet, and most "other fish" is
  Norway pout, which now has a pool in planktivorous_fish). With the three-group split this is a
  recalibration anyway.

## 6. Decision (user, 2026-10-24): the long-run level, 2 t/km2

The gadoid group is to represent the long-run state, 2 t/km2 (7200 t on the 3600 km2 grid), not
the 2021-25 peak. For the three-group split the same convention (GOV wing, q = 1, depth-stratified,
2001-25) gives: benthivorous_fish mean 1.3 (median 1.2; 2001-20 1.1, 2021-25 2.1), predatory_fish
1.0 (0.9; 0.85, 1.7), planktivorous_fish 1.4 from GOV, which is a floor; the acoustic grid level
sets the planktivores (~4.7 in July, ~2.5-3 as an annual mean;
`../Ekolodsdata planktivorer/acoustic_planktivores.md`).
