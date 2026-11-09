# Whiting's missing predation: selectivity, fine-scale avoidance, total loss, feeding level (2026-11-09)

Research only. Nothing in the library, code, manifest, resume or `results/` was changed. `lib/diagnostics/field_diet.py`
was read to trace one constant. No value here is fitted to model output; model numbers are quoted as context only.
Sources and numbers: `evidence.md`. Scripts: `scripts/` (run with `python3 -I`); output: `tables/`; inputs and
licences: `data/README.md`. The frozen-policy model diagnostics that raised the question are in `diagnostics/`.

Tags: [F] full text read; [P] part of the full text read (sections named in evidence.md); [A] abstract only;
[S] secondary (cited inside a source read, or the earlier note named); [D] derived here from data; [J] judgement or
inference.

## Side finding first: a haul-key collision in two earlier scripts

`Vitlingens predatorer/scripts/gov_overlap.py` and `Krill i Skagerrak/a_revisit/scripts/gov_by_bed.py` key GOV hauls
on (year, quarter, ship, haul number). The catch of a same-numbered haul elsewhere in the North Sea is then added to
the box haul. Example, checked in the raw file: 2016 Q1, ship 26D4, haul 18 at 57.8805 N received the whiting of
haul 18 at 54.4449 N (4 546 /h).

With ShootLat in the key [D] (`scripts/gov_overlap_fixedkey.py`, `scripts/whiting_bed_shares_fixedkey.py`):

| quantity, box, 2000-2020 | stored | corrected |
|---|---|---|
| L cod >= 20 cm x whiting 10-28 cm, Q1 | 1.57 | **1.07 (0.81-1.39)** |
| L cod >= 20 cm x whiting 10-28 cm, Q3 | 1.57 | **1.84 (0.56-3.31)** |
| L saithe x whiting 10-28 cm, Q1 / Q3 | 0.39 / 0.87 | **0.26 / 0.17** |
| whiting bed shares < 100 / 100-150 / 150-250 / > 250 m (2000-2019) | 0.533 / 0.211 / 0.251 / 0.006 | **0.765 / 0.210 / 0.025 / 0.000** |

- The stored bed shares are `FIELD_BED_SHARES['whiting']` in `lib/diagnostics/field_diet.py`, which budget_gate's
  field block uses. They equal the old `gov_by_bed.py` box output exactly.
- Over 150-250 m beds the whiting rate falls from 58.5 to 3.0 kg/h.
- With 2021-25 added, box L is 1.03 in Q1 and 1.59 in Q3.
- The saithe shares are robust to the key, as "Sejens djupläge" found.
- To recheck, not redone here:
  - "Vitlingens predatorer" 3.1 (the x1.6 co-occurrence);
  - "Torskfönster och bottenandel" section 3;
  - "Vattenpelaren på vitling" (whiting x saithe L, whiting field spread);
  - budget_gate's whiting field rows;
  - the other "Krill i Skagerrak" scripts with the same key.

## Conclusions

1. **Do gadoid predators select prey, or take what they meet? Mostly what they meet, and below their ceiling.**
   - Diets track availability:
     - cod: more than 75 % of fish prey come from the least preferred size quartile, so "prey abundance is the
       main determinant" (Floeter & Temming 2003 [A]; Floeter 2005 thesis [P]);
     - northwest Atlantic cod "generally reflects prey availability" (Link & Garrison 2002 [P]);
     - Celtic Sea predators are "opportunistic", with "general independence between predator and prey
       distributions" (Trenkel et al. 2005 [P]).
   - Where preference is measured at the station scale, it exists but is modest (southern North Sea, 150 matched
     hauls, Chesson alpha; Ludwig et al. 2024 [P]):
     - adult cod: whiting 0.18 +- 0.06 and dab 0.13 +- 0.05;
     - adult whiting: sandeel 0.63 +- 0.43 and clupeids 0.80 +- 0.30;
     - juvenile whiting: no significant preference.
   - **Size.** Once size is accounted for, cod, whiting and grey gurnard prefer relatively smaller prey as they grow
     (Floeter & Temming 2005 [A]; thesis [P]). Saithe's ratio shifts with season.
   - **Negative switching.** At 60 x 60 nmi, North Sea cod and whiting show b 0.15-0.35 (Rindorf et al. 2006 [F],
     36 559 stomachs). The authors say the scale "cannot be used to determine the underlying biological
     mechanisms". Kempf et al. 2008 [P] explain it largely by overlap and early saturation.
   - **The sharpest datum is whiting itself** (Rindorf & Gislason 2005 [F], cached postprint):
     - fish intake plateaued at 6-17 % of maximum ration with prey abundant;
     - other-food intake did not fall when fish intake rose;
     - there were no fish or "other food" specialists;
     - fish prey were eaten at dawn and dusk.

     The fish were neither full nor visibly choosing.
   - **Hake** in Biscay and the Celtic Sea selected (Mahé et al. 2007 [A]): its diet "did not generally reflect fish
     prey availability".
   - **None found among the sources read** shows a well-fed gadoid declining suitable prey it has met.
   - Implications, in principle [J]:
     - The field supports **p fixed at 1 against suitable prey that is met**, with the shortfall carried by a, the
       capture window and overlap.
     - A free p has no field counterpart for fish that are not satiated.
     - **Undecided:** whether satiation narrows a gadoid's diet. No field measurement was found.

2. **Do whiting avoid cod at 2-4 km? No sign of it in the box or the Skagerrak, down to the tow.** [D]
   (`scripts/overlap_scale.py`, cod >= 20 cm x whiting 10-28 cm, GOV 2000-2025)
   - **Box L.** Q1 1.03 (0.77-1.30), 256 hauls; Q3 1.59 (0.50-2.81), 180 hauls.
   - **L does not fall from coarse scales to the tow.**
     - Box Q3: 0.25 deg 2.02 -> 0.1 deg 1.69 -> haul 1.59.
     - Skagerrak Q1: about 1.0 at every scale.
     - Skagerrak Q3: 1.07 at 0.5 deg, 1.6-1.7 at 0.25 deg and below.
   - **Within one ~10 km cell and survey (0.1 deg)**, hauls with more cod hold the same or more whiting: L_within
     1.29 / 1.08 in the Skagerrak (Q1 / Q3) and 1.54 in box Q1. The single exception is box Q3 (0.55), on only
     20 hauls.
   - **Where field L is low, it is set at large scale by habitat, not within cells:**
     - northern North Sea Q3: L 0.38, flat from the haul to 0.5 deg, L_within 0.9-1.1;
     - saithe x whiting in the box: 0.03-0.26, by bed depth.
   - **Rose & Leggett 1990 [A].** Cod and capelin are in phase above 4-10 km, out of phase at 2-3.5 km ("scales less
     than aggregation dimensions"), and in phase at 3.5 m while feeding. That this tracks the size of capelin
     aggregations rather than detection of predators is my reading [J].
   - **The only fine-scale whiting record goes the other way.** Whiting removed an aggregation of more than 50
     million juvenile cod in 18 km2 within 5 days (Temming et al. 2007, via Fauchald 2009 [S]).
   - **Scale [J].**
     - A GOV tow is a ~3.7 km x ~20 m line, about two model cells long. It cannot resolve below itself.
     - Gadoid encounters happen at metres.
     - No source measures prey responding to a predator kilometres away.
   - Implications, in principle [J]:
     - Avoidance between cells, down to L ~0.35, has no support at the scale the survey resolves.
     - Avoidance inside a cell (below ~1 km) is untested.
     - The evidence is against **avoidance on** as the model realises it. It does not decide about avoidance
       within a cell.

3. **Whiting's total loss in the box is at least the model's booked L, and above its realised L.** [D]
   (`scripts/catch_curve.py`; NS-IBTS CPUE-at-age 1991-2025, cohort Z Q1 -> Q1)

   | unit | ages 2->3 / 3->4 / 4->5, 2000-2025 | 2010-2025 |
   |---|---|---|
   | box | 1.53 / 1.15 / 1.03 | 1.07 / 1.09 / 0.79 |
   | roundfish area 8 | 1.51 / 1.28 / 1.21 | 1.16 / 1.22 / 0.89 |
   | stock area (RFA 1-9) | 0.75 / 0.75 / 0.58 | 0.71 / 0.62 / 0.41 |

   - **Stock-wide Z** is consistent within about 0.1-0.2 with SMS 2018-22 total M (0.67 / 0.51 / 0.49 at ages 2-4)
     plus F. North Sea Fbar was 0.09-0.18 in 2020-25 and higher earlier, so the 2010-2025 row is the cleaner
     comparison.
   - **The box excess, ~0.4-0.8 /yr, is net emigration**, as "Beståndsreglering" found.
   - **Against the model.** For comparison only:
     - the booked group L 0.98 is at the low end of the box field Z;
     - L at the realised M2 0.09 (~0.70, a model diagnostic) is below every box estimate since 2000.
   - The comparison is approximate both ways: field Z is for adults in numbers, while the model's L is for the
     whole group, juveniles included. An adult-only model L would use e_A 0.3 and the adults' share of M2.
   - **The split.** SMS 2023 attributes adult whiting M2 (ages 2-5) mainly to harbour porpoise (~0.20 of 0.27-0.31),
     with cod 0.03-0.06 and saithe 0.02-0.09 [S, "Restdödlighet M0 hos torskfiskar"]. The model books porpoise
     0.020 and fish 0.34. The field Z constrains the sum, not the split.
   - **Caveats on Z.**
     - Z is everything that lowers a cohort's index: M + F + net emigration + any drop in q at age.
     - Box ages come from DATRAS's area-level age-length keys.
     - Age 6 is a plus group, so 5->6 is not used.
     - Q1 1->2 is about 0 or negative, because 1-groups arrive.

4. **A full reserve all year is not what field gadoids show.**
   - Field feeding level against Cmax at ambient temperature ["Födonivå och mättnad i fält", S]: whiting 0.25 (Q1
     0.23-0.26, Q3 0.33-0.36; 45 % of Q1 stomachs empty), cod 0.53, haddock 0.22, saithe unmeasured.
   - Reserves cycle ["Svältgräns och startreserv", S]:
     - haddock liver lipid moves through ~0.33-0.8 of the range;
     - saithe liver is ~4-7 % of body mass, lowest in Q1-Q2;
     - whiting whole-body energy density "varies little month to month" (Hislop 1991), but Pedersen & Hislop
       2001 [A] report seasonal differences "obvious in mature fish".
   - The library's whiting `start_fill` (0.30-0.73) is a haddock proxy.
   - Implications, in principle [J]:
     - A fish at fill ~1 has no energy reason to choose, so a learned p has no gradient and drifts. That fits the
       run-to-run spread.
     - Field fish at 0.25-0.5 of Cmax with reserves below the maximum would have that reason.
     - The full reserve is a model state, not a field state. It sits upstream of both decisions.

## What would settle it

1. **Selectivity.** Station-matched cod and saithe stomachs against GOV prey catches in the Skagerrak, with Chesson
   alpha per haul and predator condition (HSI) as covariate. The ICES DB has 593 Skagerrak cod stomachs, 4 with
   whiting: too few without new sampling. Lab or field tests of hunger-dependent selectivity in gadoids were not
   found.
2. **Avoidance.** Hauls 1-2 km apart in one area and survey: the GSBTS boxes (~21 hauls per 10 x 10 nmi; North Sea,
   not the Skagerrak). Alternatives: acoustic telemetry of whiting and cod together; night hauls (the box has 1).
3. **M2 split.** Porpoise diet and density in the Skagerrak, against the SMS North Sea split.
4. **Reserve.** A monthly whiting HSI or liver-lipid series (none found). Pedersen & Hislop 2001 (paywalled) has
   energy density by quarter.
