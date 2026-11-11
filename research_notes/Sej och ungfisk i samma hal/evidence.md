# Evidence - saithe and the benthivores' juveniles in the same haul; re-booking WC -> benthivores (2026-11-11)

Grades: [F] field data parsed here or full text read, [L] lab, [A] abstract only, [S] secondary / via another note,
[D] derived here, [J] judgement. Model facts M1-M7 are read from the library, the code and the exec'd derivations.

## Model facts

- M1 Library pair (fgconfig/fg_library.yaml `predatory_fish_preys_on_benthivorous_fish`): predation_mortality 0.0833,
  max_intake_rate 0.0418, stage_access [0.392, 0.0388], depth_access_table 0.0693 / 0.057 / 0.0234 / 0.0128 (mean 1
  on the 60 x 60 map: 1.63 / 1.34 / 0.55 / 0.30), dark_ratio 0.165. The decided corrections (combined capture-p x
  schooling x2.96; hake out, stage p, adults off the 2 m slab: a 0.117, stage_access [0.3818, 0.01653]; recruit
  shape 2.71 / 1.00 / 0.21 / 0.05) are in research_notes/"Vattenpelarens vuxna bottenfisk" gate_adult.py, central
  run `pA+Aprof+hake_out` (gate 1.54 equal cells / 1.41 field spread / 2.13 P2).
- M2 Growth budget (lib/world/growth_budget.py, section 250): an open-population DM with `somatic_production` P has
  g = P / ((s* - u) x 1460); `gross_annual` returns P before any predation term, so the booked M2 on the
  benthivores (seals 0.0373, WC 0.0833, whiting 0.1067, porpoises 0.0068) is NOT in their g. Benthivores P 0.74,
  g 0.000550 /tick; predatory_fish P 0.42. Verified on in-memory copies (`scripts/growth_check.py`).
- M3 Other uses of predation_mortality: tools/probes/budget_gate.py (booked diet shares, prey-side implied M2 =
  booked M2 x encounter share / booked share, total ration row), tools/probes/probe_diets.py and ideal_growth.py
  (booked columns), lib/world/habitat.py `predation_m2` (only for `spawn: {mode: equilibrium}`: infauna, epifauna,
  Nephrops, krill - not the benthivores, whose spawn is `colony`), the FG editor's matrix.
- M4 Tests: none pins this pair's predation_mortality (grep of tests/ for 0.0833 and the pair name: only
  tests/test_half_saturation.py, which pins its dark ratio 0.165 / -5 deg). tests/test_growth_budget.py asserts
  the benthivores' g is the same with and without M1 (section 250); tests/test_open_pool.py's ration assert is
  whiting's and its krill M2 sum excludes this pair; tests/test_pair_shares.py compares the benthivores' g with
  and without the epifauna pair's accessible_fraction. None moves with the booking.
- M5 Juvenile compartment (lib/environments/ecosystem_env/stages.py, section 223.32 / 250): J rides inside B (every
  movement fraction carries its J); recruits are imported into J uniformly over the cells (the decided
  recruit_depth_table would shape the arrival by bed class); one J/B per cell; no separate J movement.
- M6 Decisions: the water column and the benthivores move by learned policies on 1 km cells (60 x 60), with a
  Rest/hide action; edge moves of the fish are masked (section 266.16). Nothing in the engine places a group
  relative to another group except the policies.
- M7 Booking origin [F notes] (research_notes/"Rovfiskens uppdelning" tables/split_fields_223_stdout.txt lines
  49-62; params_split_223.md 3.1-3.2; params_split_mack.md section 2): the pre-228 water column (0.754 t/km2,
  nine members) consumption-weighted share on "Bf" 0.061; M2 = sum_m Q_m Bf_m / 1.53 = 0.0844, x0.9874 (library
  sum) = 0.0833; section 228 conserved the M2 on the rest when mackerel and horse mackerel left (both score < 0.1
  on benthivores, so the mackerel group has no pair).

## Field and literature

- E1 NS-IBTS GOV Q1 + Q3 2000-25, ICES DATRAS extracts on disk (research_notes/"Fiskgrupper i rutan"/data/extract and
  research_notes/"Torskfiskarnas artsammansattning" (a-umlaut)/data/extract; HH + CPUEL, every species, CC BY 4.0),
  parsed here (`scripts/gov_within_class.py`) [F data, D]. Hauls with BySpecRecCode 1 (zeros real); box 436
  hauls (Q1 256, Q3 180), window 57.3-59.2 N, 8-12 E 1533. Median tow 3.43 km x 20 m wing spread. Saithe >= 35 cm
  present in 0.08 / 0.33 / 0.92 of the box Q1 hauls by class (< 100 / 100-150 / 150-250 m), 0.27 / 0.93 / 0.94 / 1.0
  in Q3; J present in 0.89-1.0 of the hauls in every class with hauls. Hauls per (year, class) stratum in the box:
  median 1-5 (150-250 m Q1: 1; Q3: 2).
- E2 Walker, Maxwell, Le Quesne & Jennings 2017, ICES J. Mar. Sci. 74:1448-1457 (GOV catchability at length for
  haddock and cod; the coefficients as in research_notes/"Rekryternas bottendjup" gov_juv_bed.py) [S].
- E3 research_notes/"Rekryternas bottendjup" [F data / D]: J definition (haddock 27.0, cod 27.1, plaice 16.0, witch
  14.8 cm), member shares of J (haddock 0.87, cod 0.13), class shapes, the class-level F_eff of the pair with its
  depth table (J 0.655-0.72, A 0.88), Gap 2: within-class co-occurrence not representable at class level.
- E4 research_notes/"Sejens hojd over botten" [F data / D]: haul-level F 0.086 (box) / 0.296 (wide) for in-gape
  haddock (haul-weighted, Sejens djuplage extract, haddock < 19 cm only, presence-weighted Q3 0.9); saithe box
  presence Q3 0.9 / Q1 0.1; cooccurrence_check.py (J part x F).
- E5 research_notes/"Vattenpelarens vuxna bottenfisk" evidence E1-E19 and scripts/field_vs_model.py [S]: member diet
  shares on the model's benthivore members, Q/B, J/A fractions, with ranges (saithe 0.05 (0.02-0.09), A 0.08
  (0-0.17); spurdog 0.06 (0.02-0.15), A 0.85; anglerfish 0.19 (0.08-0.34), A 0.85; ling 0.10 (0.03-0.20) [J], A 0.5
  [J], Q/B 2.0 [J]; pollack, velvet belly 0 (0-0.02); hake 0.01 (0-0.03)).
- E6 research_notes/"Sejens diet pa unga torskfiskar" (a-ring) summary [S]: saithe 5 % (2-9) of food on benthivores,
  almost all 0-1 group haddock (CRR 164 1980-82 5.5 %, CRR 219 1991 5.0-5.2 %, SMS observed stomachs 4.1 %, Mehl
  2005 5.8 %); haddock share ~0 in Q1 and peaking in Q3; SMS modelled 9.1 % vs observed 4.1 % in 1980-91 (SMS
  overshoots x0.45); SMS share vs haddock density log-log slope 1.11 (structural, not independent).
- E7 SMS North Sea key run 2023 (ICES WGSAM 2024), sms2023_summary.out, sms2023_partial_m2.out on disk
  (research_notes/"Fiskgrupper i rutan"/predators/data/population), parsed here (`scripts/sms_rebook.py`) [F data
  of an ICES model, D]: 2001-22 saithe biomass 366 kt (0.637 t/km2 over 575 000 km2 [J]), Q/B 2.87, haddock 4.1 %
  of saithe food (J 3.5, A 0.6), no cod; saithe partial M2 on haddock < 27 cm 0.432, >= 27 cm 0.052 /yr. Box
  haddock J per saithe tonne 0.42 vs the North Sea 0.25.
- E8 research_notes/"Fiskgrupper i rutan"/predators/evidence_C_diet_consumption.md [S]: the "Bf" pool of the 223
  diets includes rays, gurnards, weevers, gobies, dragonets, poor cod, rocklings, eelpouts ("Other gadoids (small)"
  counted Bf; "Bf is slightly high for spurdog and monkfish"); member Q/B saithe 3.0 (2.87-3.6), spurdog 2.0
  (1.5-2.5), anglerfish 1.8 (1.5-2.2), hake 3.5; horse mackerel Bf 0.07 from Dahl & Kirkegaard 1986 (20-29 cm
  horse mackerel, eastern North Sea July 1985: haddock 16-43 %), while Skagerrak horse mackerel are mostly 8-16 cm
  and the 2023 box sample was 87 % krill.
- E9 Carroll et al. 2019, Global Ecol. Biogeogr. 28:1561-1577, doi:10.1111/geb.12984, "A review of methods for
  quantifying spatial predator-prey overlap" [title only: Unpaywall record, closed access; `papers/`]. Named
  because the factor used here, mean(S Y) / (mean S x mean Y) inside a stratum, is the standard biomass-weighted
  co-occurrence ratio of that literature; its definition here is the script's, not read from the paper.
