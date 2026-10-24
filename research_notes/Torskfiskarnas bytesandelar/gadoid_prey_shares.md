# Gadoid prey shares phi: benthos, pelagic fish and zooplankton from one stage-resolved method

Purpose: the pooled `gadoids` group caps its eat allocation per prey pair with `predator_share` phi
(`eat_j = min(eat_j, phi_j)`; `lib/environments/ecosystem_env/predation.py`). Only the benthos pair
had a phi (0.53). The fish and zooplankton pairs were uncapped, so an energy maximiser put the whole
group on herring and zooplankton. This note scores all three pairs with one method. phi_j is the
share of the pooled gadoid BIOMASS (species x life stage) that can exploit pool j at the full
per-capita rate. Scores come from morphology and habitat only. Diets are used only as a check.

Tags: **[F]** full text read or data downloaded, **[A]** abstract/snippet, **[S]** secondary,
**[D]** computed here. Detail is in:
- `evidence_A_morphology_habitat.md`: gill rakers, jaws and gape, body form, water-column position,
  stage boundaries, prey position [F]/[A].
- `evidence_B_stage_biomass.md`: stage biomass shares from ICES assessments (SAM numbers- and
  weight-at-age), the check of the survey's small-fish shares [F]/[D].
- `evidence_C_diet_check.md`: diet by mass from the ICES stomach database 2020-25 and older
  sources, mapped to the model pools [F]/[S].
- `scripts/` (run with `python3 -I`, paths as arguments; order: `length_biomass.py`,
  `pelagic_prey_sizes.py`, `prey_shares.py`, `joint_caps.py`, `diet_cap_check.py`; `stock_*` belong
  to evidence B), `inputs/` (stages, boundaries, scores, gape, pelagic 0-group, diet check),
  `tables/` (outputs and stdout), `data/` (downloads, untrusted, see `data/README.md`).

## Bottom line

| Pair | phi central | Range | Q1 / Q3 | Basis |
|---|---|---|---|---|
| `gadoids -> benthic_community` | **0.56** | **0.33-0.64** | 0.69 / 0.52 | haddock 1.0 at all stages; cod 0.72, whiting 0.23, saithe 0.05, hake 0.06 (stage-weighted) |
| `gadoids -> pelagic_fish` | **0.37** | **0.22-0.58** | 0.23 / 0.43 | gape class (sprat / + juvenile herring / all clupeids) x mouth, and access; saithe carries 0.23, haddock 0.06, whiting 0.06, cod 0.03 |
| `gadoids -> zooplankton` (krill-sized prey) | **0.52** | **0.41-0.61** | 0.44 / 0.54 | access to the near-bottom/midwater krill layer x capability; saithe 0.22, haddock 0.18, whiting 0.10 |
| zooplankton if the pool is half copepods | 0.34 | 0.25-0.43 | 0.26 / 0.37 | phi = f x 0.52 + (1 - f) x 0.17, f = krill share 0.5 (0.25-0.75); copepod-sized phi 0.17 (0.10-0.28) |
| **Joint cap, fish OR zooplankton** | **0.59** | 0.59-0.73 (variants raw to Fraser; no MC) | 0.47 / 0.63 | sum over stages of w x max(score_F, score_Z); see "Model consequence" |

Central = raw survey mix (depth-stratified NS-IBTS GOV Q1+Q3 2021-25), central scores and boundaries [D].
Range = Monte Carlo 5-95 % (haul bootstrap, triangular scores, boundaries, pelagic 0-group shares;
4000 draws), extended to the centrals of the Fraser-q variants, which are treated as bounds, not as
the central (same convention as `../Torskfiskarnas artsammansättning/gadoid_species_mix.md`). The fish
low end 0.22 is the gape-share sensitivity (section 2.4). With hauls held fixed, the score and
boundary uncertainty alone is +-0.04 (benthos 0.48-0.56, fish 0.31-0.39, zoo 0.44-0.56). The
haul sampling, the season and the catchability convention are wider than any single score.

**Recommendation:** benthos 0.56, pelagic_fish 0.37, zooplankton 0.52.

**Model consequence (the most important finding).** The three caps sum to 1.45. Under
`min(eat_j, phi_j)` an energy maximiser can still put 0.37 + 0.52 = 0.89 of its eat mass on fish and
zooplankton. Benthos is then forced to only 0.11 [D, `tables/joint_caps.csv`]. The caps are
right per pair, but they do not say that the same saithe cannot feed on herring and krill at the full
rate at once. A joint cap on fish + zooplankton of 0.59 forces benthos to >= 0.41 (Q1 0.53, Q3 0.37).
The benthos note's validation band (0.35-0.50) is a diet share by MASS, while phi caps the EAT
allocation. The gadoid ceilings are energy-equalised (h benthos 87, zooplankton 130 ticks, herring
higher), so at the ceiling an eat share of 0.41 on benthos is a larger mass share. The forced
minimum is therefore at or above the band's low end, but an exact comparison needs the h ratios. How
to put a joint cap into the tick is a model design question outside this note.

**Why benthos moved from 0.53 to 0.56.** The species-level scores were haddock 1, cod 1, whiting 0,
saithe 0, hake 0. Stage resolution lowers cod to 0.72 (large cod, more time in midwater) and gives
whiting juveniles, juvenile hake and deep-feeding saithe a hyperbenthos share (mysids, Crangon and
shrimps are in `benthic_community`). Net +0.03. The Fraser low bound (0.33) and the Q1/Q3 split are
almost unchanged from the species-mix note.

## 1. Pool definitions and crustacean mapping

These definitions move phi more than any score.

| Prey | Model pool | Note |
|---|---|---|
| Polychaetes, bivalves, ophiuroids, other infauna | benthic_community | |
| Mysids, gammarid amphipods, cumaceans, isopods | benthic_community | hyperbenthic, on or just above the bed |
| Crangon, Processa, Pontophilus, Pandalus, Nephrops, Calocaris, crabs, galatheids | benthic_community | epibenthic decapods |
| Calanus and other copepods, Paraeuchaeta | zooplankton | 1-3 mm; copepod-sized scores |
| Euphausiids (Meganyctiphanes, Thysanoessa), pelagic hyperiids (Themisto) | zooplankton | 15-40 mm; krill-sized scores |
| Pasiphaea | zooplankton by habit (judgement) | deep, partly epibenthic by night; 300-400 m by day [A]; kept apart in the diet check |
| Herring, sprat, mackerel, horse mackerel | pelagic_fish | the model pool: herring 0.73, mackerel 0.19, sprat 0.06, horse mackerel 0.02 |
| Norway pout, sandeel, blue whiting, gobies, gadoids, flatfish | none | "other fish": in no model pool |
| Cephalopods | none | |

- `pelagic_fish` is scored as "can this stage forage on schooling clupeids or mackerel", not "eats
  fish". Most gadoid fish prey in the area (Norway pout, gadoids) is in no model pool (section 4).
- The library comment says the zooplankton pair stands in for euphausiids. So krill-sized prey is
  the central reading. But the model's zooplankton biomass is calibrated to the A17 totals, which
  are mostly Calanus and Metridia. That mismatch is flagged, not fixed: the row "half copepods" in
  the bottom line shows its size.

## 2. Method (tasks 1-3)

phi_j = sum_i w_i [ (1 - p0_i) sum_k s_ik score_ikj + p0_i score_i,pel0,j ] + sum_m w_m score_mj [D]
- w_i: box biomass share of species i (depth-stratified, as in the species-mix note).
- s_ik: share of species i's biomass in stage k, from the 1-cm survey length biomass.
- p0_i: pelagic 0-group share (not caught by the GOV).
- score = access A x capability C, each min / central / max in [0, 1].
  - A: overlap of the stage with the layer where the pool's prey is (bottom, near-bottom krill
    layer, midwater schools), by day and night.
  - C: mouth position, jaws, barbel, gape vs prey size, gill rakers, dentition, body form.
- Minor species (~1 % raw, ~2-3 % Fraser) keep species-level scores; they move phi < 0.01.

### 2.1 Stages and boundaries (task 1)

Boundaries are morphological where possible: the length at which the gape admits a prey class, or a
body-form marker (evidence A, sections 2c and 5) [F; haddock, whiting and saithe gape thresholds are
the evidence agent's jaw-scaled estimate from the cod gape line, not measured]. Published piscivory
onsets are all diet-derived and only agree with these lengths.

| Species | Stages (central boundary, range) | Boundary basis |
|---|---|---|
| Cod | pelagic 0-group (< ~7 cm); juvenile < 30 (25-35); medium 30-50; large > 50 (45-57) | settlement ~7 cm [S]; gape admits 15-20 cm herring at 30-45 cm and 25 cm herring at 51-57 cm (Scharf et al. 2000 gape line) [F] |
| Haddock | pelagic 0-group (< ~7 cm); < 40 (35-50); >= 40 | pelagic ~3 months [F]; gape admits small herring from 42-63 cm (jaw-scaled) |
| Whiting | pelagic 0-group (< 5-10 cm, with jellyfish); juvenile < 20 (18-23); adult | FAO [F]; gape admits a 10 cm sprat from 20-23 cm (jaw-scaled) |
| Saithe | 30-60 cm; > 60 (55-70) | ages 0-2 in coastal nurseries [F]; gape admits 25 cm herring from 57-76 cm (jaw-scaled) |
| Hake | < 20 (16-25); >= 20 | FAO pectoral-fin marker at 20 cm SL [F] |

**Stage shares of species biomass** [D, `tables/stage_shares.csv`; raw pooled, Fraser A_floor in
brackets]:
- Cod: < 30 cm 13 % (18), 30-50 cm 40 % (49), > 50 cm 47 % (33).
- Haddock: < 40 cm 76 % (68), >= 40 cm 24 % (32).
- Whiting: < 20 cm 22 % (15), >= 20 cm 78 % (85).
- Saithe: 30-60 cm 66 % (71), > 60 cm 34 % (29). Nothing below 30 cm in any variant.
- Hake: < 20 cm 2 % (2).
- Q1 vs Q3 differs most for whiting (< 20 cm: 16 % in Q1, 29 % in Q3) and cod (< 30 cm 6 % / 19 %).

**Check of the small classes against the stock assessments** (evidence B) [F/D]:
- Stock-level biomass share of age 0 in Q3: haddock 1.6 %, whiting 10.9 %, cod (Viking substock,
  which holds the Skagerrak) 1.7 %. The 0-group dominates numbers (32 %, 61 %, 68 %) but not biomass.
- Age 1 in Q1: haddock 9.5 %, whiting 23 %, cod 7.2 % (2021-25; 2019-24 higher, strong year classes).
- The raw box survey shares of small haddock are at or above the stock shares. The Fraser
  up-correction of small haddock (q 0.012 at <= 11 cm) is not confirmed: it would need the box to
  hold 6-13 times the stock-average 0-group per older fish. So the raw shares are the central and
  Fraser is a bound.
- Small whiting in Q1 are fewer in the box than in the stock (geography, not gear).
- Saithe: the assessment starts at age 3. Box saithe < 30 cm are ~0 in both sources.
- Pelagic 0-group (Jun-Jul, 3-7 cm): haddock 0.1-0.4 %, whiting 1.2-3 %, cod 0.2-1.5 % of stock
  biomass. Annual mean used here: haddock 0.05 %, whiting 0.7 %, cod 0.15 %, hake 0.1 %
  (`inputs/pelagic0.csv`); 0 in Q1. They move phi by < 0.005.

### 2.2 Trait evidence used in the scores (evidence A)

| Trait | Cod | Haddock | Whiting | Saithe | Hake |
|---|---|---|---|---|---|
| Mouth, jaws, barbel [F] | terminal, lower jaw shorter, barbel well developed | inferior, "jaw bones greatly shortened", small barbel | terminal, upper jaw projects slightly, sharp teeth, barbel small/absent | lower jaw projects, small barbel | large oblique mouth, lower jaw projects, sharp teeth, no barbel |
| Upper jaw, % of body length [F] | 10.7 | 7.8 | 10.4 | 8.0 | 12.8 |
| Gill rakers, first arch [F] | 21-28 | 24-27 | 21-22 (19-26) | **35-40** | 8-12, short and thick |
| Body form, swimming [F/A] | fusiform, Ucrit 1.6-2.6 BL/s, in situ < 1 BL/s | fusiform, good sustained swimmer | elongate, Uopt 1.0 BL/s | fusiform, forked tail, Uopt 1.4 BL/s, schools | elongate, slender |
| Water column [F/S] | > 55 % of time within 5 m of the bed; night ascents | Barents Sea: up to ~150 m off bottom by day | Oslofjord: in the krill layer by day, upper 50 m at night | pelagic schools; near bottom > 300 m by day, far above it at night | near bottom by day, off bottom at night |

Prey position (evidence A section 6): in Norwegian fjords krill and overwintering Calanus sit at or
near the bottom by day (Onsrud & Kaartvedt 1998 [F]; Kaartvedt et al. 2021 [F]). No Skagerrak-slope
measurement was found. Herring and sprat are near the bottom by day in several systems [F/A]. So
daytime overlap with demersal gadoids is plausible, and gape is the binding limit for fish.

Gill rakers: no inter-raker gap was measured for any gadoid, and no study shows these species
filtering Calanus. Retention only limits 1-3 mm copepods. For krill the limit is encounter. That is
why zooplankton is split into krill-sized and copepod-sized scores.

### 2.3 Scores (task 2) [judgement on the evidence above; `inputs/scores.csv`]

Score = A x C, central (min-max of the product). B = benthos, F = pelagic fish, Z = krill-sized,
Zc = copepod-sized. Evidence A's "Suggested scores" table was a first pass; the values here supersede
it (they add the access x capability split, the stepped gape rule and the copepod/krill split).

| Species, stage | B | F | Z | Zc | Main reasons |
|---|---|---|---|---|---|
| Cod < 30 | 1.0 (0.72-1) | 0.16 (0.04-0.36) | 0.48 (0.24-0.8) | 0.12 | barbel, pelvic taste buds, near bed; gape: sprat only |
| Cod 30-50 | 0.72 (0.42-1) | 0.38 (0.14-0.68) | 0.30 (0.12-0.56) | 0.03 | 55-57 % of time near bed; gape: sprat + juvenile herring |
| Cod > 50 | 0.64 (0.30-1) | 0.54 (0.24-0.8) | 0.12 (0.02-0.32) | 0.01 | gape admits adult herring; krill < 5 % of body length |
| Haddock < 40 | 1.0 (0.72-1) | 0.09 (0.02-0.24) | 0.40 (0.18-0.7) | 0.05 | inferior mouth, shortest jaws: sprat at most |
| Haddock >= 40 | 1.0 (0.72-1) | 0.21 (0.06-0.48) | 0.30 (0.12-0.56) | 0.03 | small herring just within gape |
| Whiting < 20 | 0.40 (0.18-0.7) | 0.04 (0-0.1) | 0.81 (0.49-1) | 0.27 | no barbel: hyperbenthos only; below sprat size; krill layer |
| Whiting >= 20 | 0.18 (0.04-0.4) | 0.40 (0.18-0.75) | 0.54 (0.28-0.8) | 0.05 | sharp teeth; sprat and small herring |
| Saithe 30-60 | 0.06 (0-0.2) | 0.70 (0.4-0.85) | 0.90 (0.56-1) | 0.50 | 35-40 rakers, forked tail, schools, diel migration |
| Saithe > 60 | 0.03 (0-0.1) | 1.0 (0.68-1) | 0.50 (0.24-0.7) | 0.30 | all clupeids within gape |
| Hake < 20 | 0.32 (0.12-0.6) | 0.24 (0.09-0.45) | 0.56 (0.3-0.9) | 0.08 | on mud by day; sprat only |
| Hake >= 20 | 0.06 (0-0.16) | 0.72 (0.42-1) | 0.12 (0.03-0.3) | 0 | largest gape, piscivore arch |
| Any pelagic 0-group | 0 (0-0.14) | 0 | 0.2 | 1.0 | copepod feeders |

Fish C follows one stepped rule: gape class "none" 0.05, "sprat only" 0.3, "sprat + juvenile
herring" 0.7, "all clupeids" 1.0, times a mouth/dentition factor (haddock 0.5, cod 0.9, others 1).
The gape classes come from the boundary lengths above.

### 2.4 Gape-share sensitivity for the fish pair [D]

Instead of the stepped rule, C can be computed as the share of today's pelagic prey that a stage can
swallow: prey length <= k x predator length (k cod 0.47, haddock 0.33, whiting 0.46, saithe 0.38, hake
0.56; `inputs/gape.csv`), applied to the near-bottom GOV catch of herring, sprat, mackerel and horse
mackerel in the region (`tables/pelagic_prey_sizes.csv`; length-weight coefficients for these four
are approximate generic values, not sourced, so the biomass shares are approximate; herring dominates
by a factor of ~20 in Q1, so the 94 % is robust).
- By prey biomass: phi_fish 0.22 (Q1 0.10, Q3 0.25). In Q1 the near-bottom pool is 94 % herring by biomass,
  80 % of it 15-25 cm, which few in-box gadoids can swallow.
- By prey numbers: phi_fish 0.38, the same as the stepped rule.
- This is not used as the central: it mixes a prey-field property (today's herring sizes, which vary
  by year and season) into a predator-side share. It is an accessible-fraction question, like f in
  the benthos note. The GOV catch also has fewer large fish than the model's pool (herring 0.73,
  mackerel 0.19). It sets the low end of the fish range.

### 2.5 What moves the result most (`tables/phi_swings.csv`, raw pooled) [D]

| Input | Benthos | Fish | Zoo |
|---|---|---|---|
| Catchability: Fraser A_floor instead of raw | -0.23 | +0.20 | +0.09 |
| Catchability: Fraser A_sw_floor | -0.08 | +0.10 | -0.01 |
| Season: Q1 / Q3 instead of pooled | +0.12 / -0.05 | -0.14 / +0.05 | -0.07 / +0.03 |
| Gape-share instead of stepped rule (fish) | | -0.15 | |
| Krill share of the zooplankton pool 0.5 instead of 1 | | | -0.17 |
| Haddock access to benthos (A 0.8-1) | -0.07 / 0 | | |
| Haddock access to the krill layer (A 0.3-0.7) | | | -0.06 / +0.06 |
| Saithe capability on fish (C 0.5-0.85) / on krill (0.7-1) | | -0.04 / +0.03 | -0.04 / +0.02 |
| Whiting adult capability on fish / krill | +-0.02 | -0.02 / +0.03 | +-0.02 |
| Any stage boundary | < 0.01 | <= 0.02 (haddock, saithe) | <= 0.02 (saithe) |

The species mix (saithe's weight: season and catchability) is the main lever on all three pairs.
Among the scores, haddock's access to krill and saithe's two capabilities matter most.

## 3. Weighting results by quarter and variant (task 3) [D, `tables/phi_summary.csv`]

| Variant | Quarter | Benthos | Fish | Zoo (krill) | Zoo (copepod) | F or Z joint |
|---|---|---|---|---|---|---|
| raw | Q1 | 0.69 (0.56-0.70) | 0.23 (0.18-0.29) | 0.44 (0.36-0.51) | 0.07 | 0.47 |
| raw | Q3 | 0.52 (0.34-0.63) | 0.43 (0.27-0.50) | 0.54 (0.42-0.61) | 0.21 | 0.63 |
| raw | pooled | **0.56 (0.41-0.64)** | **0.37 (0.26-0.44)** | **0.52 (0.41-0.58)** | 0.17 | **0.59** |
| A_sw_floor | pooled | 0.49 (0.35-0.60) | 0.47 (0.33-0.52) | 0.50 (0.39-0.57) | 0.20 | 0.63 |
| A_floor | pooled | 0.33 (0.22-0.50) | 0.58 (0.40-0.62) | 0.61 (0.45-0.66) | 0.28 | 0.73 |

Brackets: MC 5-95 %. Corner bounds (every factor at its min or max at once) are wider, e.g. raw
pooled benthos 0.37-0.67, fish 0.21-0.50, zoo 0.25-0.79. Saithe is 2 % of the Q1 mix and 39 % of
Q3 (raw), so phi_fish roughly doubles from Q1 to Q3 and phi_benthos falls.

## 4. Diet check (task 4) - a cap test, not a match [D, `tables/diet_cap_check.csv`]

phi_j caps the group's eat allocation to pool j. So the test is: the group's diet share by mass in
pool j should be <= phi_j. Where phi_j is far above the diet share, the cap would not bind. Scores
were not changed after this check. Group values weight species diets by the box mix and assume equal
ration per unit biomass (not measured). Diet shares are by mass; phi is an eat share. At the
energy-equalised ceilings benthos (h 87) gives more tonnes per unit eat share than zooplankton (h 130)
or herring, so for benthos the test is conservative: a mass share of 0.43 is a smaller eat share.

| Pool | phi | Group diet share by mass, central (range) | Verdict |
|---|---|---|---|
| Benthos | 0.56 | 0.43 (0.32-0.44) | agrees; near binding (but see the mass vs eat-share note below) |
| Pelagic fish | 0.37 | 0.07 (0.02-0.27) | agrees as a cap; far above the diet |
| Zooplankton | 0.52 | 0.22 (0.06-0.41) | agrees as a cap; well above the winter diet |
| Other fish (no pool) | - | 0.21 (0.15-0.58) | demand the model has no pool for |

By species (score vs diet; post-2000 ICES stomach database unless marked):
- **Haddock**: B 1.0 vs 0.81 (NNS + western Trench, Jan-Feb 2025, 485 fed) [F]; 0.63 in summer
  (Albert 1994, PRE-2000) [S]. Agrees. Z 0.38 vs 0.004 in winter and 0.02-0.22 euphausiids in summer
  (PRE-2000). **Disagrees**: the score is well above any diet. The access factor rests on Barents
  Sea diel ascents and fjord krill depths. It is the least-supported input and was not moved.
- **Cod**: B 0.72 vs 0.29 (Skagerrak 2023-25, 56 fed); B falls from 0.71-0.80 below 30 cm to 0.15
  at 50-69 cm, as the stage scores do. F 0.43 vs 0.06 (Kattegat 0.26). Other fish (Norway pout,
  gadoids) is 0.65 of the diet. The scores do not know Norway pout exists; that is the pool
  definition, not a score error.
- **Whiting**: B 0.23 vs 0.20, Z 0.60 vs 0.49 (Skagerrak, April 1991, PRE-2000, 147 stomachs) [F].
  Agrees. Post-2000 data are too thin (32 fed, mostly whiting eaten).
- **Saithe**: Z 0.77 vs 0.21-0.76 euphausiids at 40-50 cm (total North Sea 1981, PRE-2000) [S].
  Agrees. B 0.05 vs <= 0.01. The fish part (24-78 %) is not split by species, so F cannot be checked.
- **Hake**: F 0.71 vs 0.82 (Skagerrak Aug 2024, 48 fed; box subset 0.81) [F]. **The diet exceeds the
  score slightly.** Hake is 0.9 % of the mix; the effect on phi is < 0.002.
- Caveat on Z: every post-2000 row has Z ~ 0, but the large samples are Jan-Feb and daytime bottom
  trawls. That is no evidence that gadoids skip krill in summer.
- The 1981/1991 data describe another ecosystem state and are context only.

## 5. Gaps: what cannot be measured and must be judged

- **Access factors (A)**: no vertical-distribution data for haddock, whiting or saithe on the
  Skagerrak slope. Haddock's krill access rests on the Barents Sea; krill, Calanus and herring depths
  rest on Oslofjord and other fjords. This is the largest judged input.
- **Gape** of adult haddock, whiting, saithe and European hake: not measured. The thresholds are the
  cod gape line scaled by jaw length.
- **Raker spacing**: not measured for any gadoid, so the copepod-sized scores are judgement.
- **Krill share of the zooplankton pool**: not known for the box. It changes phi_zoo by up to
  -0.26 (krill share 0.25) and is a pool-definition question for the model, not a gadoid score.
- **Saithe weight**: one-haul leverage, Q1 2 % vs Q3 39 %, and its catchability convention. It
  shifts benthos and fish in opposite directions.
- **Other fish**: about 0.2 of the group's diet (Norway pout, gadoids, blue whiting) is in no model
  pool. At the energy ceiling the group must take that demand from the three pools.
- **Model structure, not data**:
  - Separate caps allow fish + zooplankton = 0.89 of the eat mass. Only a joint cap (0.59) brings the
    forced benthos share into the validation band (section "Bottom line").
  - One static cap cannot carry Q1 vs Q3 (fish 0.23 vs 0.43). A year-round 0.37 lets the Q1 group
    eat more herring than its composition supports.
  - phi caps eat share, not intake: a stage that can take only small herring is treated as a full
    fish eater once its cap share is allocated (section 2.4).
- **Not obtained**: Du Buit 1991, Bergstad 1991, Huwer et al. 2014, DAPSTOM, the restricted part of
  the ICES stomach database (2013, most 1981/1991), WGSAM SMS key run, Walker et al. 2017 q supplement.

## Source status (main sources; full lists in the evidence files)

| Source | Use | Status |
|---|---|---|
| DATRAS NS-IBTS Q1/Q3 2021-25 (extracts of the species-mix note) | length biomass, species mix, pelagic prey sizes | data [F/D] |
| Fraser, Greenstreet & Piet 2007 | length-dependent GOV q (bounds) | via the species-mix note [F] |
| ICES WGNSSK 2026 / SAM runs, SAG API (haddock, whiting, cod Viking, saithe, hake) | stage biomass check | data [F/D], evidence B |
| Svetovidov 1962; FAO Cohen et al. 1990; FishBase morphology pages | rakers, jaws, mouth, habitat | [F], evidence A |
| Scharf, Juanes & Rountree 2000 MEPS 208:229 | cod gape line | [F] |
| Hobson et al. 2007/2009; Engås et al. 1988; Onsrud & Kaartvedt 1998; Onsrud et al. 2004 | water-column position, krill layer | [F] |
| ICES Stomach Content database (stomachdata.ices.dk), public part 2020-25 | diet by mass | data [F/D], evidence C |
| Albert 1994; Dorman et al. 1992; Daan 1989 (FishBase); Bergstad 1991 | older diets (context) | [S]/[A] |
