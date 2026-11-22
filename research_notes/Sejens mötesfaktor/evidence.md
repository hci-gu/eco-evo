# Saithe's encounter-realisation factor: evidence trail (2026-11-22)

Research subagent with one literature subagent (`lit_evidence.md`). No edits to the library, code, tests, manifest,
AGENTS.md or the resume. Nothing committed. Tags: [F] full text / primary data, [A] abstract, [S] secondary or earlier
note, [D] arithmetic here, [J] judgement.

Rule, as in research_notes/"Mötesfaktorn per grupp" E0: factor = the field whole-product a / the library's chain a
before section 296 (the "Was ..." value in the library comment), under the anchor's own conditions. No model output,
FIELD_QB or potential ration sets the factor. The budget_gate runs (E5) are consistency checks only.

## E1. Data: saithe stomachs and Norway pout in the SAME haul, NS-IBTS Q1 2026 [F]

- **Stomachs.** The ICES stomach DB download held in research_notes/"Födonivå och mättnad i fält"/data/A_ices_stomach/all
  (ICES data licence). It holds 1278 saithe records (959 hauls). The only set that both weighs prey and links to a
  survey haul with a prey catch is the Norwegian 2026 Q1 upload: R/V G.O. Sars (58G2), GOV and JTDS trawls, 11-26 February
  2026, 29 hauls at 57.3-61.2 N, 0.3 W-6.1 E, beds 80-169 m. It has 207 single fish (203 after 4 regurgitated),
  33.5-93 cm (median 46 cm, 732 g), individual weights, and prey weights in mg (`tables/saithe_db_overview_stdout.txt`).
  - The 1980-87 records are counts only in Q1 (no prey weights). They are weighed in Q3 1982/1984/1986, but DATRAS
    holds no Q3 survey before 1991 for those ships. They pair only by rectangle and across mixed gears, so they were not
    used [J].
- **Diet.** Of the prey mass, 0.75 is Norway pout (0.82 of fish prey). Then come unidentified Teleostei and Gadidae,
  krill (0.05) and Pasiphaea. 50 stomachs are empty and 41 hold pout.
- **Prey density in the haul.** DATRAS NS-IBTS Q1 2026 HH/HL (`scripts/fetch_datras.py`, ICES data policy). Pout
  numbers at length are scaled to each catch category's recorded CatCatchWgt. The pout length-weight 0.0065 L^3.17
  overstates these fish 1.56x, so it is used for shape only. The weight is divided by the wing swept area (Distance x
  WingSpread, median 0.070 km2) and by the Walker et al. 2017 GOV q-at-length (0.066 <= 11 cm, 0.19 >= 12 cm). That is
  the convention behind the library's 1.64 t/km2 pout (research_notes/"Bottenfiskens bytesbalans" section 5) [S]. At
  q = 1 the densities are x11.8 lower.
- **Ration.** WGSAM 2024 gadoid evacuation (saithe = whiting parameters rho 0.00171, delta 0.081, lambda 1.35, as in
  research_notes/"Fiskgrupper i rutan"/predators/scripts/calc_ger_cmax.py) [S]:
  - C_pout = 24 rho e^(delta T) L^lambda E^-0.85 S_tot^0.5 x S_pout / S_tot, per fish.
  - T 7 C (6-8.5) [J]: BotTemp is not reported.
  - E pout 4.6 kJ/g (4.2-5.4) [J].

## E2. Chain side [D]

- Inputs: the pre-296 a 0.02348 (`predatory_fish_preys_on_planktivorous_fish`), x 0.92/0.80 (saithe's share of the
  pair's a / its biomass share), x 0.95 pout access (saithe's 19 cm gape, "Makrillens bytesstorlek" section 5 [S]),
  x m_Feb 0.815 (dark_ratio 0.14, -5 deg, 58.15 N, daily mean), x q(T 7 C) 0.887 (ingestion_q10 2.25 over the at_50m
  layer).
- Each haul's bed depth then takes the pair's depth_access factor, renormalised over the box cells as in the engine:
  < 100 m 2.47, 100-150 m 1.08, 150-250 m 0.24.
- The sharing over saithe's other prey types is not used here. The pout comparison concerns the fish pair only.

## E3. Fit (`scripts/saithe_pout_2026.py` -> `tables/saithe_pout_2026_stdout.txt`, `saithe_pout_2026_hauls.csv`) [D]

Factor = the field Type II low-density slope over each haul's chain slope. Pmax is free (Rindorf's form).

| reading | factor | note |
|---|---|---|
| Type II, all 29 hauls | **0.081** | plateau 0.0177 g/g/d = 0.82 of GER Cmax; half-saturation ~2.6 t/km2 pout |
| haul bootstrap, 5-95 % | 0.022-0.19 | median 0.078 |
| T 6 C / E 5.4 ... T 8.5 C / E 4.2 | 0.065-0.098 | |
| GOV hauls only (n 21, D <= 3.5) | 0.056 | Walker q is a GOV catchability |
| JTDS hauls only (n 8, D 0.4-12.9) | 0.31 | the four densest hauls are JTDS |
| ratio sum P / sum chain D | 0.026 | biased low under saturation; not a candidate |
| GOV q = 1 instead of Walker | 1.03 (ratio 0.31) | not the library's pout convention; a corner |
| pout past DigestionStage 0 only | 0.001 | net-feeding bound (E4) |
| content capped at FullStomWgt | 0.065 | E4 |

- Spearman(P, D) is 0.57. It is not cited as support, because net feeding produces the same correlation (E4).

## E4. Net feeding: the pout-rich hauls are probably contaminated (`scripts/net_feeding_check.py`) [D on F]

- **Digestion stage.**
  - 3195 of 3323 g of pout in the stomachs is DigestionStage 0. Of the other prey, 127 of 1110 g is.
  - The stage-0 pout is 8-11 cm, the small pout category of the same catches.
  - The code list was not found (ICES StomachData fact sheet: codes from tblSampleCodes, 9 = missing). Reading 0 as
    fresh is [J].
  - Haul 152 (no pout in the catch) and haul 150 hold only stage-1 pout.
- **Stomach fill.**
  - 21 fish carry > 5 %BW of pout (max 35 %BW), against the sustained fill 2.5-5 %BW behind the GER Cmax.
  - 19 of those 21 fish come from hauls with > 10 kg pout in the catch. These hauls average 3.4 %BW pout per fish,
    against 0.15-0.19 %BW at lower catches.
- **Ration.** The top hauls imply 1.2-3.6 %BW/d of pout alone in February. The field annual Q/B 2.83 is 0.78 %BW/d
  of all food.
- **Stomach weight.** Summed prey exceeds the recorded FullStomWgt (stomach including wall) in 30 of 153 fed fish.
  This holds for 20 of the 21 fish above 5 %BW of pout: prey in the gullet or mouth, or a recording error. Capping each
  fish's content at FullStomWgt gives a Type II factor of 0.065 (ratio 0.020).
- **Opposite bias.** The haul catch is a noisy proxy for the prey the fish met over the previous day. That flattens the
  slope (errors in variables) and partly offsets the upward biases [J].
- **Pool to pout.** The pair's a is a pool average that includes herring, which near-bed saithe barely reach. The
  chain's a per tonne of pout is likely higher, which would lower the factor [J].
- **Reading [J].** Saithe probably fed on the codend pout during the haul, so the field slope is a PROBABLE upper bound (conditional on stage 0 = fresh and on Walker q). All
  hauls were by day, and 24 x R overstates a day feeder's ration, which points the same way. Stage > 0 alone (0.001)
  is a floor that discards true recent meals. The anchor is a bound, not a value.

## E5. budget_gate on library copies (`scripts/make_copy_wc.py`, `scripts/gate_copy.py`) [D]

- **Method.** Each copy scales the predatory_fish pair a's by F/0.16.
- **Live library.** It reproduces 298: potential 3.12 = 1.10 x Q/B 2.83, 5 short months
  (`tables/budget_gate_live.txt`).
- **Check that the copy took effect.** The engine's `max_intake_mat` row is printed at the top of each run, and the
  a's change as intended.

| copy | engine a -> planktivores | potential x Q/B (annual) | monthly range | short months | ceiling share | new OFF |
|---|---|---|---|---|---|---|
| live (0.16) | 0.003757 | 1.10 | 0.85-1.36 | 5 (Dec-Apr) | 0.41 | - |
| all pairs 0.056 | 0.001315 | 0.61 | 0.45-0.75 | 12 | 0.22 | none |
| all pairs 0.08 | 0.001878 | 0.76 | 0.57-0.94 | 12 | 0.28 | none |
| all pairs 0.19 | 0.004461 | 1.20 | 0.93-1.48 | 2 (Jan-Feb) | 0.45 | none |
| all pairs 0.30 | 0.007044 | 1.47 | 1.16-1.80 | 0 | 0.55 | none |
| all pairs 1.0 | 0.02348 | 2.10 | 1.75-2.56 | 0 | 0.79 | none |
| planktivore pair only 0.08 | 0.001878 | 0.96 | 0.71-1.20 | 7 | 0.34 | none |
| planktivore pair only 1.0 | 0.02348 | 1.76 | 1.50-2.16 | 0 | 0.72 | WC -> krill (diet gate, both spreads: enc 0.113 vs booked 0.370) |

- A common factor leaves the diet shares and the prey-side ratios unchanged. The OFF list stays the live one
  everywhere (porpoises' pairs, whiting -> benthivores, whiting -> epifauna at the field spread, WC -> mackerel prey
  side, seals -> WC).
- On a planktivore-only factor, WC -> benthivores moves (enc/booked 2.74 at 0.08, 0.64 at 1.0, both "ok").

## E6. Literature (`lit_evidence.md`, literature subagent; ~40 min cap) [F/A]

- **No source pairs a saithe, hake or spurdog ration with a prey density in t/km2.**
- **Smith & Smith 2020**, Fish Fish 21:891, doi 10.1111/faf.12468 [F]. These are fitted multispecies functional
  responses for pollock (= saithe), spiny dogfish and goosefish from NEFSC stomachs and uncorrected trawl or bongo
  densities.
  - Saithe 21-80 cm on euphausiids is Type IV, peaking at the lowest sampled density.
  - Dogfish > 80 cm on herring is flat.
  - The parameters (Table S3) were not obtained (Wiley 403).
- **López-López et al. 2015**, MEPS 541:165, doi 10.3354/meps11490 [F]. European hake fullness against blue whiting
  is Type III with a threshold of 100-1000 ind/km2. Deviance explained is 2-6 %, so it gives no slope.
- **DeVries et al. 2025**, doi 10.3389/fmars.2025.1621343 [F]. Spurdog consumption against herring is a Kendall tau
  only.
- **Tanasichuk 1999** [A]. Pacific hake's euphausiid selection held at 12-27 % of the 1991 krill. This is plateau
  evidence.
- **Not searched:** ling, pollack, European anglerfish, Bergstad 1991, Du Buit 1991, Icelandic saithe.

## E7. Krill and the other pairs

- **Krill.** No anchor. No krill density exists for any saithe haul, which repeats "Krillintag och jaktsätt" Q1.
  Smith & Smith's Type IV is a shape without parameters.
- **Benthivores, whiting, mackerel prey.** No paired data. The 2026 set holds 230 g of Gadidae across all hauls,
  which is too little to fit.
