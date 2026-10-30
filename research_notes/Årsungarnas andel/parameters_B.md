# Parameters for the juvenile compartment (plan B, work package 1) - 2026-10-29

Derived from field data only; no model output enters. Sources in this folder: `summary.md` (stage
shares), `evidence_selectivity.md` (fishing), `evidence_prey_size.md` (predator windows), scripts and
`tables/` as named below. Grades: D computed from data, F stated in a source, A analogy, J judgement.

## 0. Stage definition (refines decision 2 of plan_B.md)

J = fish below the species' mean length at age 1 in Q3 (the size at their second summer), from the
2021-25 CA records in the Skagerrak rectangles: herring 18.2 cm, sprat 10.9, Norway pout 14.9,
haddock 27.0, cod 27.1, whiting 20.5, plaice 16.0, witch 14.8 [D]. The user's "age 0 + 1 until the
second summer" as a size limit: a graduation RATE then fits it, a fixed age-quarter cut does not (it
needs a calendar graduation date, and the field cycle under it cannot be reproduced by a rate).

Box J/B (q-corrected, planktivores pool-weighted, benthivores member-weighted;
`tables/size_stage_share_by_group.csv`) [D]:

| group | 2001-25 Q1 | Q3 | 1991-96 Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|---|
| planktivores | 0.45 | 0.30 | 0.78 | 0.45 | 0.56 | 0.78 |
| benthivores | 0.15 | 0.16 | 0.42 | 0.39 | 0.49 | 0.38 |

## 1. Growth split kappa and graduation gamma [D]

SMS 2023 key run, 2013-22, J as age 0 + 1 (`scripts/stage_box_model.py`,
`tables/stage_box_model_stdout.txt`):

| | planktivores | benthivores |
|---|---|---|
| per-tonne growth J / adults (/yr) | 1.07 / 0.20 | 2.06 / 0.50 |
| total mortality Z, J / adults (/yr) | 0.84 / 0.68 | 1.55 / 0.81 |
| **kappa = g_J / g_A** | **5.3** | **4.1** |
| z = mass at entry / at graduation | 0.13 | 0.06 |
| **gamma** (de Roos et al. 2008) | **0.66 /yr** | **1.02 /yr** |

gamma = (nu - mu) / (1 - z^(1 - mu/nu)) is the maturation rate of a stage-structured biomass model
(de Roos, Schellekens, van Kooten, van de Wolfshaar, Claessen & Persson 2008, Theor. Popul. Biol. 73:
47-62). A constant guess below nu - mu (0.23 / 0.51) makes J self-sustaining; 1 /yr was close to
that edge for the planktivores. Herring and pout carry the planktivore values (sprat's SMS weights
are flat). Later option: gamma from the model's own per-cell J growth and loss (state-dependent).

## 2. Recruitment schedule rho(t) [D, J]

Arrival window (J): Jun 0.15, Jul 0.35, Aug 0.35, Sep 0.15 of the annual recruitment; 0 in Oct-May.
Evidence: no age-0 fish in the May survey, all in by August; haddock settle mid-Jul to mid-Aug
(`../Sejens diet pa unga torskfiskar/evidence_settlement.md`), herring 0-group from Jun-Jul.

Magnitude: the one number set to the box survey (field data): the annual-mean J/B of the 2001-25 box
(planktivores 0.38, benthivores 0.15) under the field rates of section 1. As allocation weights (J
tonnes take kappa x an adult tonne's share of the group's new tonnes, recruitment adds rho(t) per adult
tonne, in adult-growth units):

| month | Jun | Jul | Aug | Sep | other |
|---|---|---|---|---|---|
| rho planktivores | 1.7 | 4.1 | 4.2 | 1.8 | 0 |
| rho benthivores | 0.2 | 0.5 | 0.5 | 0.2 | 0 |

Share of the group's new tonnes going to J: planktivores 0.74-0.89, benthivores 0.39-0.58 by month.
Field cycle J/B Feb / May / Aug / Nov: planktivores 0.37 / 0.35 / 0.40 / 0.39 (box survey Q1 0.45, Q3
0.30); benthivores 0.15 / 0.14 / 0.17 / 0.16 (0.15, 0.16). The box planktivores peak in Q1 because
their adults leave in winter (WBSS to the Sound, NSAS to spawn): a resident group cannot reproduce
that phase, and its J/B peaks in August instead.

Caveat: the box is a nursery. Its J/B is 2-5x the stock's (SMS North Sea 0.05-0.26), because the fish
emigrate as they grow; in the model that emigration becomes graduation inside the box.

## 3. Fishing selectivity s_J (rate on J tonnes / group mean) [D, J]

`evidence_selectivity.md` (SMS F at age, cross-checked against the 2015-25 ICES assessments); the
"age 1 all year" reading matches the size-defined stage:

| impact x group | s_J | range | grade |
|---|---|---|---|
| pelagic_trawling x planktivores (98 % of their F) | **0.7** | 0.35-0.85 | D |
| bottom_trawling x benthivores | **0.15** | 0.08-0.35 | D (high end J: Skagerrak shrimp and Nephrops trawls retain 10-20 cm gadoids) |
| bottom_trawling x planktivores | 1.5 | 1.0-2.2 | J |
| pelagic_trawling x benthivores (F 0.0004) | 4 | 2-5.3 | J |

Keeping the group's mean F, adults get s_A = 1.23 (planktivores) and 1.20 (benthivores).

## 4. Pair stage access [f_J, f_A] [D, J]

Member windows from `evidence_prey_size.md` (p95 prey/predator length over the box predator sizes:
saithe 0.30, spurdog 0.34, hake 0.50, anglerfish 0.57, mackerel 0.33, horse mackerel 0.31, whiting
0.39, cod 0.30; velvet belly 11 cm, J; porpoise 0-40 cm; harbour seal 7-35 cm), weighted by each
member's share of the pair's a, applied to the stage length distributions (`scripts/stage_access.py`,
`tables/stage_access_stdout.txt`):

| pair | f_J | f_A | f_J Q1-Q2 / Q3-Q4 | effective f at box J/B | today |
|---|---|---|---|---|---|
| water column -> benthivores | **0.43** | **0.027** | 0.30-0.34 / 0.52-0.56 | 0.088 | f 0.075 |
| whiting -> benthivores | **0.049** | **0** | 0.01-0.02 / 0.06-0.11 | 0.007 | f 0.05 |
| water column -> planktivores | **0.32** | **0.11** | 0.26-0.28 / 0.37 | 0.19 | f 1 |
| benthivores -> planktivores (cod) | **0.49** | **0.14** | 0.42-0.45 / 0.54-0.56 | 0.27 | f 1 |
| porpoises -> planktivores | 1 | 1 | | 1 | 1 |
| seals -> benthivores | **1** | **0.34** | | 0.44 | 1 |
| seals -> planktivores | 1 | 1 | | 1 | 1 |

The water column -> benthivores effective f (0.088) reproduces the independently derived 0.075 of
223.31; the planktivore pair's 0.19 is the "~0.2" estimated in "Makrillens bytesstorlek". The
whiting -> benthivores pair falls from 0.05 to ~0.007: whiting only reaches the small end of age 0.

**Validation against SMS stomachs (field, `tables/sms_juvenile_diet_stdout.txt`)**, juvenile share
of the take implied by the stage access at the box J/B vs SMS (North Sea, 2013-22):

| pair | implied | SMS |
|---|---|---|
| water column -> benthivores | 0.74 | saithe on haddock 0.77 |
| whiting -> benthivores | 1.0 | whiting on haddock 1.00 |
| water column -> planktivores | 0.64 | saithe on herring 0.15, mackerel / horse mackerel 1.00 |
| cod -> planktivores | 0.69 | cod on herring 0.19 (stock J share 0.14 against the box's 0.38) |
| porpoises -> planktivores | 0.38 (= box J/B) | porpoise on herring 0.14 (= the stock's 0.14) |
| seals -> benthivores | 0.34 | grey seal on haddock 0.27, cod 0.30 (stock 0.35, 0.19) |

Read the planktivore rows through the nursery caveat: the box J/B is 2.7x the stock's, so juvenile
shares of the take are higher there by construction. As selectivity (odds ratio of the juvenile share
eaten over the stock's): saithe on herring 1.1, cod on herring 1.4, porpoise 1.0, against the
derived f_J / f_A 2.9 (water column, mackerel-weighted) and 3.6 (cod): the derived cod window may be
too small for the North Sea cod sizes behind SMS (box cod are small). Harbour seals in the box are
smaller predators than SMS's grey seals; the seal window is the weakest row (otoliths not corrected
for digestion). Kappa et al. 2026 (Hvaler, Sep 2020; evidence_prey_size.md section 6) puts the juvenile share of the
harbour seals' benthivore take at 0.74-0.86 (implied 0.34); f_A from it would be 0.03-0.06. Kept as the
window (capability) with the Kappa share as the validation target, the porpoise convention.

## 5. Open points

- The seasonal f_J (low in Q1-Q2, high in Q3-Q4: the age-1 fish grow out of the windows in spring)
  is folded into the annual mean; the compartment's own dynamics carry part of it.
- gamma and kappa from the North Sea SMS run, not the box.
- Whiting juveniles: later (decision 3).
- Kappa et al. 2026 (Mar. Mamm. Sci., doi 10.1111/mms.70069) probably holds the local harbour seal
  prey lengths (1977-78 vs 2020); not fetched (download blocked), worth a manual fetch.

## 7. Whiting (resume 230, user 2026-10-31) [D, J]

The third staged group, with the same methods; the scripts gained a whiting entry and the outputs for
the two older groups are unchanged (`tables/*_230*`).

- **Stage:** whiting below its length at age 1 in Q3, 20.5 cm (section 0). Box J/B 2001-25 Q1 0.14, Q3
  0.19 (1991-96 Q1-Q4 0.50 / 0.48 / 0.52 / 0.75), target annual mean 0.165.
- **kappa, gamma** (`scripts/stage_box_model.py`, SMS species 17, 2013-22;
  `tables/stage_box_model_stdout_230.txt`): g_J 1.73, g_A 0.39 /yr -> **kappa 4.5**; Z_J 1.90, mass ratio
  0.09 -> **gamma 0.65 /yr** (sensitivity 0.52-0.81 moves the cycle by <= 0.01).
- **Recruitment** (window Jun-Sep 0.15/0.35/0.35/0.15, J): rho **0.6 / 1.3 / 1.4 / 0.6**; field cycle J/B
  Feb / May / Aug / Nov 0.16 / 0.13 / 0.19 / 0.18; start share by month 0.138-0.195.
- **Fishing** (`scripts/box_weighted_sj.py`, `tables/box_weighted_sj_stdout_230.txt`): bottom_trawling
  **s_J 0.29** (variant A, 2001-25 shares; 0.19-0.29 over the variants; whiting age 0 F ~0, age 1 0.026
  against the stock F 0.072); pelagic_trawling **4** (J, as the benthivores: industrial small-mesh
  bycatch is small fish; F 0.005; cap 1/p_J ~6).
- **Stage access** (`scripts/stage_access.py`, `tables/stage_access_stdout_230.txt`), effective at J/B
  0.165:

| pair | f_J | f_A | effective | was |
|---|---|---|---|---|
| water column -> whiting (saithe 0.77, spurdog 0.10, hake 0.07, velvet belly 0.04, anglerfish 0.01 of a) | 0.57 | 0.15 | 0.22 | f 0.47 |
| benthivores (cod > 20 cm) -> whiting | 0.38 | 0.042 | 0.10 | 1 |
| seals (7-35 cm) -> whiting | 1.0 | 0.84 | 0.87 | 1 |
| porpoises (0-40 cm) -> whiting | 1.0 | 0.95 | 0.96 | 1 |

The water column's access halves: the 223 value 0.47 used one gape length per member (saithe 19 cm),
the windows use each member's length mix at p95 0.30. The cod pair gets a prey-side window for the first
time; its a keeps the predator-side capable share 0.10, as benthivores -> planktivores (a x 0.32, stage
access [0.49, 0.14]).

Mechanical check (untrained 1-update policies, frozen_years 1 yr, seed 20260530): whiting J/B 0.07-0.14
by month, filling at recruitment and graduating after; juvenile share of the water column's whiting take
0.35, seals 0.15, porpoises 0.14. The policies validate nothing; a trained run does.
