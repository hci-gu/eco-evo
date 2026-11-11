# The planktivores' energy budget against its sources (2026-11-11)

Research subagent (saved by the main agent). Every term of `planktivorous_fish`'s energy budget checked
against its sources and against member field metabolism and field consumption, following the method of
`Bottenfiskens budget` and `Fältmetabolism kolja och plattfisk`. Nothing in the library, code, tests,
manifest, resume or AGENTS.md was changed; no value was chosen to match model output.

- Evidence with quotes: `evidence.md` (E1-E28, search log G1-G4).
- Arithmetic: `scripts/budget_terms.py` -> `tables/budget_terms_stdout.txt`, `scripts/otolith_bw.py` ->
  `tables/otolith_bw_stdout.txt` (`python3 -I`).
- OpenAlex refused every request (HTTP 429); all sources are local full texts held by sibling notes.
  Nothing was downloaded.

Tags: [M] measured; [S] secondary (model output, second-hand, abstract only); [D] derived here; [J]
judgement. Units MJ per tonne per year unless stated; 1 MJ/t/tick = 4 kJ/kg/d.

## Verdict

1. **The library's cost terms match their sources and each other.** [D from M/S]
   - `resting_metabolism` 8.0 is a resting value (the Rose/Rudstam herring term at 129 g and 14 C): 0.78 x
     the measured routine of adult herring at low activity (Johnstone et al. 1993, 9.3 C) and 1.45 x the
     fasting rate of starved juveniles (Bernreuther 2007).
   - An eat tick costs 1.65 x RM, the Rudstam herring respiration at 1 BL/s; it multiplies RM, so swimming
     is not counted twice within an eat tick. SDA left feeding_cost in section 244: no double count.
   - Rebuilt member by member, the annual resting term is 1.06 x the library's; two source errors cancel
     (herring taken at 129 g while the box mix costs 1.33 x more per kg; the effective Q10 fitted on pout's
     Hansson cod term, 1.8-1.9 x a cod-SMR proxy).
2. **The group's field metabolism brackets what the trained fish spend.** [D] Biomass-weighted, without
   SDA: low 11 160; measured-routine 14 620; seasonal 14 150; Wisconsin 16 730; high 19 760. The library
   charges 9 741 resting all the time, 13 350 at the field feeding-time share 0.57, 15 673 at the trained
   split, 16 073 eating all the time. In library terms the field fish spend like a fish at the eat cost for
   0.70-1.10 of its time (central readings); the trained fish do 0.81.
3. **The term that does not fit is the comparison Q/B 5.0.** [D]
   - With the model's losses (assimilation 0.8 minus SDA 0.11) and P 0.66, Q/B 5 leaves a respiration of
     7 700 / 9 700 / 11 150 at 3.6 / 4.18 / 4.6 kJ/g prey: at most the "low" reading, nothing for
     continuous swimming. With the Rudstam (FB4) herring losses only 6 250-9 300, below every measured
     herring rate (11 200-15 600) and below the library's pure resting cost.
   - Its sources are stomach contents with evacuation models (Greenstreet 1997, 12 h gastric emptying where
     6 h has been observed) and Ecopath stock values: adult-weighted, biased low.
   - It is weighted at the section 209 shares (herring 0.79); at today's shares 4.87.
   - It counts wet tonnes of field prey at 3.6-4.2 kJ/g, while the closure reads them at the model's
     4.5 / 5.0: the same Q/B carries 12-25 % more energy in the model.
   - The bioenergetic lineage of the cost terms implies 34-66 GJ/t/yr ingested; the trained fish need
     30-32. That lineage was validated against in-situ rations only for 0.04-3 g herring in Jul-Oct.
4. **The closure's e* was overstated by a bug.** `closure.py` read `natural_mortality` (per tick) as per
   year, so M1 (0.273 /yr) was missing from the refill. Corrected e* 0.28 at the field fill, 0.08 at fill 1
   (was 0.43 / 0.32). Neither is a field eating share: the field fish swim all the time. (Main agent:
   confirmed and fixed in closure.py, levers.py and budget_fmr.py on 2026-11-11.)
5. **The source-consistent Q/B is 1.16-1.32 x the 5.0.** Field metabolism plus P x (ec + s ME) gives
   5.8-6.6 model tonnes per tonne-year at the field fill (4.9-7.6 over the full range), 6.2-7.0 at fill 1.
   The trained fish eat 6.6-7.4, so most of their excess over 5.0 is the comparison. What is left is fill 1
   instead of the field fill (+0.37) and the krill-heavy diet share (1.19-1.27 x booked).

## 1. The terms, their sources, and internal consistency

| term | library | source | finding |
|---|---|---|---|
| `resting_metabolism` | 8.0 MJ/t/tick at 14 C; annual mean 6.67 (x0.834) = 26.7 kJ/kg/d = 9 741 | Rose/Rudstam 0.0033 W^-0.227 e^0.0548T g O2/g/d at 129 g (E3, E4) | resting, not routine |
| `metabolism_q10` | 1.5 effective, mean_0_50m | herring's own Rose RQ 0.0548 = Q10 1.73; 1.5 a fit to a member series (E4) | absorbs pout at the Hansson term |
| `feeding_cost` | 1.65 x RM per eat tick | Rudstam e^(0.03 VEL) at 1 BL/s for herring, 1.5 [J] gadoids (E23) | herring 1.73 at box lengths; day / 0.7-night 1.60; Rudstam's own VEL equation 2.8; no double count |
| `resting_cost` | 1.0 | engine default | a rest tick costs less than any measured routine (herring low-activity routine 1.29 x RM); field fish do not rest (E11, E12) |
| `movement_cost` | 1.75 | judgement (243.3 blend) | a fish at 1 BL/s covers 4-5 km per tick; a move tick should cost about an eat tick [J]; 1 % of costs |
| `sda_fraction` | 0.11 of ingested | sprat YOY 0.10, cod 0.12 (E23) | FB4 herring 0.147 of ingested; library at the low end |
| assimilation | 0.8 | library | FB4 herring 0.756 of ingested; ~5 % on q |
| field Q/B | 5.0 (`FIELD_QB`) | members: herring 4.7, pout 4.5, blue whiting 7.5, sprat 6.0 (E22) | section 3 |
| `somatic_production` | 0.66 /yr | SMS weight at age at the box J/B (E25) | section 5 |

RM is resting: ratios to the Rose term at equal mass and temperature are 1.29 (Johnstone 1993, adults
25.5-31 cm at low activity, 0.093 mg O2/g/h at 9.3 C) and 0.69 (0.50-0.87) (Bernreuther 2007, starved
6.6 g juveniles at 9.7 C, "intermediate between standard and routine") [M via S]. Box herring weigh 10-160 g,
biomass-weighted ~45 g; at W^-0.227 the mix costs 1.33 x the 129 g term (12 155 vs 9 166) [D]. Pout: no
respirometry; a cod-SMR proxy gives 19.6-21.6 kJ/kg/d at 12-25 g on the bottom layer; the Hansson cod
"resting" term is 1.8-1.9 x that, i.e. routine, so 1.5 activity on top counts pout's routine twice [M via
S, D]. Rebuilt per member (herring Rose at box sizes, pout at the cod-SMR proxy, blue whiting 0.8 x Hansson
[J], sprat Meskendahl standard): 10 313 = 1.06 x 9 741 [D].

The library's implied annual metabolism [D] (Q/B needed in model tonnes, field fill / fill 1; the refill
here includes M1):

| case | MJ/t/yr | kJ/kg/d | x RM | Q/B needed |
|---|---|---|---|---|
| all rest | 9 741 | 26.7 | 1.00 | 4.46 / 4.83 |
| e = 0.5 | 12 907 | 35.4 | 1.32 | 5.43 / 5.81 |
| e = 0.57 (field feeding-time share) | 13 350 | 36.6 | 1.37 | 5.57 / 5.95 |
| trained split 0.08 / 0.81 / 0.11 | 15 673 | 42.9 | 1.61 | 6.29 / 6.66 |
| e = 1 | 16 073 | 44.0 | 1.65 | 6.41 / 6.79 |

The engine's single "eat" action covers two field quantities: the feeding-time share (0.57 -> 13 350) and
the cost-side share (e_eq 0.70-1.10 -> 14 150-16 730); the trained fish sit at 0.81 (15 673).

## 2. Field metabolism per member

Annual means at the box temperatures and sizes, without SDA, MJ/t/yr ("x RM" against 9 741):

| member (weight) | reading | MJ/t/yr | x RM | tag / basis |
|---|---|---|---|---|
| herring (0.554) | a. adult routine at low activity (Johnstone 1993), no mass scaling | 11 224 | 1.15 | [M via S] |
| | b. a, mass-scaled to the box mix | 15 630 | 1.60 | [D] |
| | c. seasonal: Wisconsin diel Apr-Oct, field fasting loss 15 kJ/kg/d Nov-Mar | 14 780 | 1.52 | [D] |
| | d. Wisconsin diel: Rose at box sizes x activity 1 BL/s day, 0.7 BL/s night | 19 436 | 2.00 | [S] |
| | e. Wisconsin, 1 BL/s all day | 20 980 | 2.15 | [S] |
| | f. Rudstam's own VEL (upper) | 34 104 | 3.50 | [S] |
| | juvenile fasting (Bernreuther), floor | 12 421 | 1.28 | [M] |
| Norway pout (0.350) | low: cod SMR proxy (25 g) x 1.5 | 10 725 | | [S/J] |
| | central: blue-whiting otolith FMR as proxy (b 0.03-0.06) | 12 700 | | [D/J] |
| | high: max(cod SMR at 12 g x 2.2, Hansson at 25 g x 1.25) | 17 305 | | [S/J] |
| | swimming 14-16 cm/s by day (~1 BL/s), 10-11 cm/s at night (Onsrud 2005) | | | [A] |
| blue whiting (0.075) | otolith, cod-anchored (Alewijnse Barents site), x1.10 to Jul-Sep at 50 m | 13 994 (10 173-19 543) | | [M]/[D] |
| | blue whiting / cod at equal mass and T | 1.21-1.51 (CI 0.91-1.82) | | [D] |
| sprat (0.021) | Meskendahl routine / standard, 8 g | 22 430 / 20 327 | | [M] |
| | otolith C_resp 0.34-0.42 above the calibration's asymptote 0.243: no FMR | | | |

Herring and Norway pout are not in the Alewijnse dataset.

| group case | MJ/t/yr | kJ/kg/d | x RM | e_eq |
|---|---|---|---|---|
| low | 11 162 | 30.6 | 1.15 | 0.22 |
| measured routine (b) | 14 624 | 40.1 | 1.50 | 0.77 |
| seasonal (c) | 14 154 | 38.8 | 1.45 | 0.70 |
| Wisconsin diel (d) | 16 733 | 45.8 | 1.72 | 1.10 |
| high | 19 758 | 54.1 | 2.03 | 1.58 |

e_eq = (FMR / RM - 1) / (1.65 - 1): the share of time a library fish must spend at the eat cost to spend the
field FMR.

## 3. Field consumption, and the respiration it implies

| source | value | kind | tag |
|---|---|---|---|
| Greenstreet et al. 1997, North Sea pelagic planktivore guild 1983-85 | 1.09 / 1.66 / 1.67 / 1.17 % BW/d by quarter; Q/B 5.11; Q1 / annual 0.78 | stomach + evacuation, herring with 12 h gastric emptying ("as fast as 6 h observed"); sandeel in the guild | [M], [D] |
| member Q/B behind 5.0 | herring 4.7 (juvenile 5.63 / adult 4.34; EcoBase juvenile 12.5), pout 4.5, blue whiting 7.5, sprat 6.0; 4.87 at today's weights; 5.24 with the box's juvenile-heavy herring | stock-level, adult-weighted | [S], [D] |
| Albert 1995, pout 11-19 cm, 140 m, July, 7 C | 1.1 % BW/d (4.0 /yr if constant) | stomach + evacuation | [M] |
| Bernreuther 2007, Bornholm herring | 2.9 % BW/d June; <= 0.5 % autumn and winter, "partly lower than maintenance" | stomach + evacuation | [M] |
| Arrhenius & Hansson 1994, YOY Baltic herring (0.04-3 g, Jul-Oct, 9-19 C) | 2-17 % BW/d; agrees with the Rudstam model | in situ | [M] |
| Bachiller et al. 2018, NSS herring / blue whiting | C/B 9.5-12 / 4.7-6 in a 168 / 183 d season = 34-48 / 17-24 GJ/t/yr ingested | Rudstam / Hansson model output at 1 BL/s | [S] |
| Arrhenius & Hansson 1993, Baltic herring age 1-2 | Q/B ~12-18 wet at 2.85-3.72 kJ/g = 34-66 GJ/t/yr | model output | [S/D] |

Prey energy [D]: copepods 3.6 and euphausiids 4.0 kJ/g (Bachiller); Greenstreet / Steele 4.18; Baltic
zooplankton 2.85; the model's zooplankton 4.5 and krill 5.0 MJ/t. A wet Q/B of 5.0 is worth 4.0 model
tonnes at Bachiller's energies, 4.45 at Greenstreet's, 5.0 only at the model's.

Respiration implied by Q/B 5 [D], R = A C - P (ec + s ME), P 0.66, fill 0.63:

| losses | 3.6 kJ/g | 4.18 kJ/g | 4.6 kJ/g |
|---|---|---|---|
| model (0.69 of ingested) | 7 700 | 9 720 | 11 160 |
| FB4 herring (0.609 of ingested) | 6 250 | 8 030 | 9 300 |

The largest metabolism Q/B 5.0 can pay at the model's q is 11 500 (field fill) / 10 300 (fill 1), the
"low" reading; Q/B 6 gives 10 200-14 300, Q/B 7 12 700-17 500. Pout at 1.1 % BW/d all year with P 0.66-1.4
gives R 9-19 kJ/kg/d against a cod-SMR proxy of 19.6-21.6, so a July ration as the annual mean cannot pay
pout's SMR plus its growth [J]. Reading [J]: stomach-based values leave too little energy for any swimming
fish at the box's growth; the bioenergetic lineage gives 1.1-2 x the library's need; the library's need
(29.5-31.9 GJ/t/yr ingested) lies between them, against 18-21 for Q/B 5 at field prey energies.

## 4. Feeding time

| member | feeding pattern | feeding-time share [J] |
|---|---|---|
| herring | visual, by day; "did not feed in darkness" (YOY); fewer adult stomachs with krill at night (17-20 % vs 31-50 %); feeds in winter at 0.4-1 % BW/d | 0.46 (daylight at 58.2 N; winter months at half) |
| Norway pout | day and night (small pout stomach weight by day 2 x night; large pout no diel pattern); all year | 0.76 |
| blue whiting | mainly at night; Jul-Sep only in the box | 0.50 |
| sprat | by day | 0.51 |
| group | | 0.57 |

On the cost side the field fish swim continuously (pout ~1 BL/s by day, 0.7 BL/s at night; overwintering NSS
herring 0.25-0.42 m/s), so the cost-side e_eq is 0.70-1.10 and the trained 0.81 is a field cost level. The
corrected e* 0.28 would ask them to sit at resting cost 72 % of the time; no member does. Seasonal: NSS
adults hardly feed Nov-Mar; the box's juvenile herring and pout feed in winter; Greenstreet's Q1 ration is
0.78 of the annual mean.

## 5. Production

P 0.66 reproduces (0.417 x 1.150 + 0.583 x 0.307 = 0.659, SMS weight-at-age growth at the box J/B); the
trained run realised 0.62. The SMS annual per-tonne growth is net of the negative quarters (adult herring Q3
-0.161, Q4 -0.086). Production energy P (ec + s ME) is 4 715 at the field fill and 5 933 at fill 1, against
a field production at whole-body energy of 4.4-11 kJ/g; pricing at the Q1 fill (~0.4) would lower it by ~650
(Q/B -0.2). Gonads are not in a somatic P [J]: ~570 MJ/t/yr (Q/B +0.17); the members spawn outside the box.
P is not where the excess comes from.

## Implications

(a) Field metabolism 14 150-16 730 (central; 11 160-19 760 full range) against the library's 15 673
(trained), 16 073 (e = 1), 13 350 (e = 0.57), 9 741 (resting).

(b) Inconsistencies:

| term | inconsistency | size |
|---|---|---|
| comparison Q/B 5.0 | pays only the "low" metabolism at the model's losses; stock / adult-weighted; 12 h evacuation; old weights; wet prey read at model energy | needs x1.16-1.32 (5.8-6.6 at the field fill); energy convention alone x0.80-0.89 |
| `closure.py` | M1 per tick read as per year | e* 0.43 / 0.32 -> 0.28 / 0.08; Bottenfiskens budget's refill 727 -> ~1 250 (fill 1) |
| herring part of RM | at 129 g; box mix ~45 g | x1.33 on the herring part |
| pout part of RM | Hansson routine term, then x1.5 activity | ~x1.8 on the pout part; cancels the size error at group level (x1.06) |
| `resting_cost` 1.0 | below routine | a rest tick 0.78 x herring low-activity routine; trained fish rest 8 % |
| `movement_cost` 1.75 > 1.65 | a cruising fish moves 4-5 km per tick at eat cost | 1 % |
| `sda_fraction` 0.11 | Rudstam herring 0.147 of ingested | -0.04 of ingested |
| winter metabolism | field fasting loss Nov-Mar 11-17 kJ/kg/d; the library charges 23 at rest and 37-38 on an eat tick; a Q10 cannot give this | reading c: -2 600 on the herring part |
| gonads | not in P | +570 |

(c) Options (nothing applied; Q/B in model tonnes, q = 3 242; e* at Q/B 5.0, corrected):

| option | costs / FMR | Q/B needed (field fill / fill 1) | e* at Q/B 5 | comment |
|---|---|---|---|---|
| library as is | 16 073 at e = 1; 15 673 trained | 6.41 / 6.79; trained 6.29 / 6.66 | 0.28 / 0.08 | the trained ration (6.6-7.4) is the library's own need at fill 1 |
| A. keep costs, replace the comparison Q/B by the field closure | FMR 14 150-16 730 | 5.82-6.61 / 6.20-6.99 | - | as Bottenfiskens budget's 4.94; the booked M2 on zooplankton (1.3) and krill (1.67), built from Q/B 5, would be ~1.2-1.3 x low: the prey groups' call (krill's g is derived from booked M2) |
| B. keep Q/B 5.0, lower costs | FMR <= 11 500 / <= 10 300 | 5.0 | 0.28 | needs the "low" reading, below the juvenile fasting rate at box temperatures; not source-consistent |
| C. seasonal herring metabolism (fasting rate Nov-Mar) | 14 154 | 5.82 / 6.20 | - | a form change |
| D. `resting_cost` = routine (1.29) | all rest 12 566 | 5.33 resting all the time | < 0 | matters only if fish rest |
| E. RM rebuilt per member | x1.06 | 6.70 / 7.08 | 0.18 | |
| F. field Q/B read at field prey energy | - | 5.0 wet = 4.0-4.45 model tonnes | -0.23 | the convention only |
| G. gonads | +566 | 6.59 / 6.96 | 0.19 | |
| Q/B at today's weights (4.87) / box juvenile herring (5.24) | - | - | 0.21 / 0.40 | |

Krill consequence [D/J]: the planktivores' krill M2 excess of 1.73-1.86 is ration 1.35-1.49 x diet share
1.19-1.27. Against option A's 6.29-6.66 the ration part shrinks to ~1.0-1.18; the diet-share part stays and
is the main parameter-side item left in the planktivores' share of the krill excess.

(d) Measured: herring low-activity routine (one study, second-hand) and juvenile fasting rates; sprat
routine and standard rates; blue whiting otolith C_resp (18 fish at the cod site); pout and herring swimming
speeds (abstracts); Greenstreet's quarterly guild rations, Albert's pout ration, Bernreuther's Bornholm
rations; Arrhenius & Hansson's YOY in-situ rations; SMS weights at age. Derived: all scalings, the blue
whiting / cod ratio and FMR, the weightings, R = A C - P, the corrected e*, every Q/B needed. Model outputs:
Rudstam / Rose terms, Bachiller's C/B, Arrhenius & Hansson 1993's consumption. Judged: pout's central FMR and
range, herring night speed 0.7 BL/s, the winter fasting rate in reading c, the feeding-time shares, the
gonad term, blue whiting at 0.8 x Hansson.

## Where this confirms or contradicts earlier notes

- `Krillens flykt och energibudget` check 4a: contradicted by the M1 bug (also in every closure.json row and
  Bottenfiskens budget's refill, ~+0.2 on their Q/B needed). Check 3 confirmed (C uses 0.95 of the
  field-implied intake; the field FMR does the same). Check 4b confirmed. "The parameters do not close with
  the field Q/B even in principle" is true, but the term at fault is the Q/B, not the costs.
- `Planktivorernas vinterkostnad`, partly contradicted: it took herring routine from Bernreuther's fasting
  rate (17 kJ/kg/d); Johnstone's adult low-activity routine at Jan-Feb temperatures is 25.1, so the
  library's rest tick (23.0) is ~0.9 x measured routine, not 1.35 x. Confirmed: winter eat ticks (37-38) are
  far above the winter fasting loss (11-17).
- `Bioenergetik för alla fiskgrupper`, confirmed: feeding_cost 1.65 is activity only (Rudstam at 1 BL/s,
  1.73 at box lengths); no SDA double count.
- `Kolmulens biomassa`, confirmed at group level (x1.06), hiding a herring size error (x1.33) and a pout
  routine double count (~x1.8).
- params_planktivorous_fish.md, contradicted: "field rations of adult NSS herring (Bachiller 2018) sit at
  the Wisconsin value" is circular (Bachiller's rations are Wisconsin model output); the 5.0 uses the 209
  weights (4.87 at today's).

## Gaps

1. Johnstone et al. 1993 not obtained ("low activity" undefined); no herring routine at 4-8 C; no herring
   otolith FMR.
2. Norway pout: no respirometry, no otolith data; its ration from one July station.
3. Bachiller's swimming-speed unit in e^(0.03 SW) unstated (the arithmetic favours cm/s).
4. Sprat needs a clupeid otolith calibration.
5. The prey energy behind the Ecopath and Greenstreet Q/B is unstated.
6. No year-round, evacuation-based ration for juvenile Skagerrak herring and pout at box temperatures.
7. Gonad production of the box's adults, and where they spawn.
