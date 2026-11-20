# Whiting's protection over shallow beds (2026-11-20)

Written by the research subagent; saved by the main agent. This is research only:
- Nothing in the library, manifest, code, tests, AGENTS.md or `mareld_resume.txt` was changed, and nothing was committed.
- No value is chosen from model output. Model numbers are context only.

Per-source detail is in `evidence.md`, all quantities in `tables/factors.csv`, and inputs and licences in `data/README.md`.

Tags: [F] full text, [A] abstract, [S] secondary, [D] derived here, [J] judgement.

## Question

In the field, whiting stand over shallow beds: 0.77 / 0.20 / 0.03 of the biomass over < 100 / 100-150 / 150-250 m beds.

In the model, both fish predators' encounter with whiting per predator is higher over < 100 m than over 150-250 m beds: 5.25x for the water column (WC) and 4.43x for the benthivores. This is because vision rule A follows the light at the bed.

Does whiting have a protection over shallow beds that the model lacks, large enough to cut that gradient by a factor of 2 or more?

## Answer

**No whiting-side protection cuts the gradient by 2.** Light-dependent escape, diel position and shelter together come to about x0.85 (0.6-1.0) [J].

The gradient is still overstated, for two reasons outside whiting's own behaviour:
- **The shallow beds are dark, not bright.** The box's < 100 m beds lie in dark coastal water. On the Swedish side, gliders measure the near-bed daytime light at 0.002-0.08 lx, around the vision half-saturation Ke (0.02 lx). The light convention puts 0.5-3 lx there. This cuts the per-predator class a by x0.55 (0.42-0.82) on the Swedish side, and by x0.64 (0.49-0.88) box-weighted with an estimated Danish side [D/J].
- **The field predators stand elsewhere.** Saithe, the main water-column member, is almost absent over < 100 m beds, and cod peak at 100-150 m. This placement cuts the hazard ratio on whiting by x0.36 (0.23-0.58) [D]. It is a placement effect, not a protection.

**Per-predator gradient.** With measured light and judged escape, it falls by x0.54 (0.24-1.06): WC 5.25 -> 2.86, both pairs 4.80 -> 2.61. That is roughly the factor of 2 asked about, but only x0.64 of it is measured, so a factor of 2 is neither established nor excluded.

**Field hazard.** With field predator placement as well, the hazard is about flat: 0.93. Varying only the placement corners gives 0.60-1.52; every factor at its corners gives 0.26-2.95. Whiting's shallow placement is then consistent with ordinary predation.

**The step inside the field range.** Between < 100 and 100-150 m beds, which hold 97 % of field whiting, the per-predator ratio steepens from 1.55 to about 1.95 under the measured light. Only cod's 100-150 m peak offsets it.

## Per mechanism

Factors apply to the shallow (< 100 m) : 150-250 m ratio (model value 4.8, both fish pairs).

| # | mechanism | verdict | factor (range) | in the model? |
|---|---|---|---|---|
| 1 | Light-dependent escape and vigilance | Direction supported; the measured bed light is already crepuscular, so little is left | x0.85 (0.6-1.0) [J on A/F] | No (p is light-independent; q 0.8 is a bright-lab value) |
| 2 | Turbidity and near-bed darkness | Supported on the Swedish side (61 % of shallow whiting); Danish side estimated | x0.64 (0.49-0.88) [D/J]; Swedish side 0.55 (0.42-0.82) [D] | No (one light profile for every bed class) |
| 3 | Diel behaviour and vertical position | No bed-class-specific pattern | x1.0 (0.8-1.2) [D/S] | Yes, as a class-independent 0.7 day / 0.5 night band |
| 4 | Predator density by bed class | Strong; placement, not protection | x0.36 (0.23-0.58); WC x0.10, cod x1.12 [D] | No (the trained WC stands shallow) |
| 5 | Other | Nothing usable | - | - |

### 1. Light-dependent escape [A/F, J]

**What the sources show.**
- Largemouth bass on groups of trout attacked most often and succeeded least at 20 lx. Their success per attack peaked at 0.5 lx, and predation frequency did not differ across 0.005-20 lx (Frey et al. 2026 [F]). This acts through prey group cohesion, which here belongs to the schooling factor.
- Pikeminnow took about 5x more chinook in relative darkness than at 160 lx (Petersen & Gadomski 1994 [A]).
- Juvenile cod escaped a fast model attack 73 % of the time in clear water and 21 % in highly turbid water (Meager et al. 2006 [A]).
- Large cod were most effective on 0-group gadoids at dusk, night and dawn (Strand et al. 2020 [A]).
- Field piscivory peaks at crepuscular light (cited in Frey et al.).

**Reading [J].** The prey's visual advantage needs tens of lux. Bed light over shallow box beds is 0.002-0.08 lx measured, or 0.5-3 lx under the convention. Both are crepuscular, where predation success peaks. No gadoid experiment measures capture per encounter on fish prey across light.

**Double counting.** Claim only the light-dependent excess over q 0.8. Schooling cohesion stays in the schooling factor.

**Night/day gate.** A flatter light response also raises the pair's night/day ratio (0.172 under the rule). The field does not contradict this: whiting stomachs show no diel pattern for larger fish prey (digestion-limited), and whiting eat fish prey at dawn and dusk.

### 2. Near-bed light over shallow beds [D]

**Coverage.** The VOTO gliders took 717 daytime casts over < 100 m beds, all on the Swedish side (57.77-58.27 N, 11.05-11.33 E), in months 1, 2, 5, 8, 9, 11 and 12.

**What they measure.**
- K_PAR at 20-80 m is 0.15-0.18 over < 100 m beds, against 0.09-0.12 over 150-250 m beds.
- On the same mission, daytime PAR at 50 m over < 100 m beds is 0.18 of that over 150-250 m beds (median; range 0.007-0.54). At 65 m it is 0.07 (0-0.43).
- Near the bed (bed - 5 m, on beds of about 76-92 m), monthly median daytime PAR is 4e-5 to 1.5e-3 umol, i.e. 0.002-0.08 lx. Between 0 and 87 % of casts fall below the sensor threshold (3e-3 lx).
- Mean daytime g at Ke 0.02 is 0.14-0.74 by month, against 0.62-0.97 under the convention.

**Rule-A light term, glider over convention, on matched months.** Central value 0.55 (Ke 0.02, r2 0.06).

| variant | F (range) |
|---|---|
| Central (Ke 0.02, r2 0.06) | 0.55 |
| Ke 0.1 to 0.002, r2 0.035-0.09 | 0.42-0.82 |
| Casts west of 11.2 E only | 0.58 |
| West of 11.15 E, beds >= 70 m | 0.50 |
| 2.5x glider level gap treated as an artefact | 0.68 (0.50-0.91) |
| 100-150 m beds | 0.48 (0.26-0.80) |

The darkness is not confined to the innermost coastal casts.

**Validation.** The simplified light term gives 0.91 for K 0.16 against 0.09 below 65 m on < 100 m beds; the chain gives 0.93.

**Danish side** (39 % of the shallow whiting biomass in GOV):
- MODIS surface Kd490 (February-October mean) is 0.148 over Danish shallow beds, 0.235 over Swedish shallow beds and 0.148 over 150-250 m beds.
- Placing the Swedish near-bed column under the clearer Danish surface layer (light x4-6) gives F 0.78 (0.59-0.97) [D/J].
- Weighting the two sides by shallow whiting biomass gives 0.64 (0.49-0.88).

**Beam attenuation does not bind.** Contrast loss would need c > 1.7-3.7 /m to pull R0 below the 2 L cap.

### 3. Diel position [D/S]

The box has 435 day hauls and 1 night haul. North of 56 N, night/day ratios for whiting 10-30 cm, matched within rectangle, year and quarter:

| survey | < 100 m | 100-150 m | 150-250 m |
|---|---|---|---|
| Q1+Q3 2000-25 | 1.50 (0.93-2.51) | 0.70 | 0.94 |
| Q2+Q4 1991-97 | 0.77 | 0.59 | 0.60 |

There is no shallow-specific day refuge. This agrees with Adlerstein & Ehrich 2000 and Rindorf 2003 [S].

### 4. Predator placement [D]

Box GOV shapes, as density over the box's equal-cell mean:

| predator | < 100 m | 100-150 m | 150-250 m |
|---|---|---|---|
| saithe >= 35 cm | 0.02 | 0.26 | 2.63 |
| cod >= 35 cm | 0.77 | 1.86 | 0.69 |
| hake >= 30 cm | 2.60 (Q3 3.08, Q1 0.12) | - | - |
| whiting >= 30 cm | 1.47 | 1.41 | 0.77 |
| water column, member-weighted | 0.23 | 0.43 | 2.30 |

- **Combined hazard profile under field placement** (both pairs, relative to 150-250 m, < 100 / 100-150 / 150-250 m): 1.72 / 2.56 / 1.00. The 100-150 m class, cod's peak, is the most dangerous.
- **Mammals.** Seals and porpoises push the other way, but their booked M2 on whiting is small (0.034 of 0.49).
- **Hake.** Hake is the shallow-standing water-column member in summer, and it eats whiting.
- **Model side.** This is measured by section 284's "WC at GOV" arms: whiting M2 falls from 1.6-1.7 to 0.70-0.84 /yr.

### 5. Other [D]

**Saithe stomachs.** Juvenile-gadoid frequency of occurrence in adult saithe is flat (0.14-0.21) while prey density falls 15-20x with bed depth, giving a clearance index of 0.04-0.05. This is not diagnostic: under ration saturation a light multiplier common to all prey cancels out of diet shares, and the data mix eras.

**Cod stomachs.** Too thin to test: 0 whiting records in 56 stomachs over 150-250 m beds.

**Already covered elsewhere.** Shelter, schooling and avoidance between cells were covered by earlier notes.

## Most load-bearing sources

1. VOTO glider PAR (CC BY 4.0) with EMODnet beds, via research_notes/"Ljus under 65 m".
2. NS-IBTS GOV from DATRAS.
3. MODIS Aqua Kd490 2015-22 (NOAA CoastWatch ERDDAP).
4. Frey et al. 2026 [F]; Petersen & Gadomski 1994 [A]; Meager et al. 2006 [A].

## What would settle it

Daytime PAR in the bottom 10 m over 60-100 m beds on the Danish side (Hirtshals-Skagen, Jutland Coastal Current), and in June-July on the Swedish side. One glider or CTD-PAR transect to the bed in summer would do it.

## Caveats

- **Glider coverage.** No glider casts over < 100 m beds in March, April, June, July or October; matched months only.
- **Below-threshold casts.** Bracketed by setting them at the threshold or at 0.
- **Lux conversion.** 50 lx per umol is a deep clear-water value; no OCR casts exist over shallow beds. This sits inside the level-variant bracket.
- **Vision parameters.** Ke and r2 are not whiting-specific.
- **Water-column split.** The member split beyond saithe 0.80 is judged.
- **Knock-on effects.** Mechanism 2 would also move the other shallow-bed visual pairs and the equal-cell a's; this was not examined.

## Files

| file | content |
|---|---|
| `scripts/glider_shallow_beds.py <repo>` | Coverage, matched-depth light, near-bed light |
| `scripts/shallow_light_factor.py <repo> [level] [lonmax] [bedmin]` | F for < 100 and 100-150 m; stdout variants in `tables/` |
| `scripts/kd490_sides.py <etopo1> <kd490 csv>` | Surface Kd490 by side and class |
| `scripts/gov_predators_by_class.py <gov dir>` | Predator and prey shapes by class |
| `scripts/gov_day_night.py <gov dir> <gov q2q4 dir>` | Day/night catch ratios |
| `scripts/saithe_clearance_index.py <gov dir>` | Saithe stomach clearance index |
| `scripts/combine.py` | Combined factors |
| `tables/factors.csv` | All quantities: mechanism, quantity, low, central, high, unit, source, tag |

Run every script from `scripts/` with `python3 -I`; the `-W ignore` flag is optional.
