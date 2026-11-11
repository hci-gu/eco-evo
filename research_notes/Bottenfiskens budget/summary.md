# The benthivores' energy budget against its sources (2026-11-10)

Main agent. Question: the policy-free closure (`Krillens flykt och energibudget`, check 4a) shows the
benthivores' budget cannot close at the field Q/B 4.46 even resting all the time. Which term is off,
judged against the sources each term came from? Nothing in the library or code was changed. No model
output sets or selects any value here. Arithmetic: `scripts/levers.py` -> `tables_levers_stdout.txt`.
Tags: [S] what the cited note or resume section says, [D] arithmetic here, [J] judgement.

## The terms and where they come from

| term | library | source | what the source built it on |
|---|---|---|---|
| field Q/B | 4.46 | 223.10; `Bentivorernas energibudget` tables/literature_evidence.md | an energy-budget closure, not a measurement: FMR incl. SDA 8 200 (6 100-10 200) + P 2 900 (Z 0.64 x 4 500 MJ/t whole body) = 11 100 MJ/t/yr assimilated -> 4.2 on the mixed diet, 4.46 on the redistributed diet (2 472 MJ/t). Published Q/B 3.48 (2.5-5.3); SMS evacuation rations cod 4.6, haddock 2.8 [S] |
| resting_metabolism | 3.33 MJ/t/tick | 223.26 | routine metabolism at the bottom layer (SMR 2.8 x routine ~1.2) [S] |
| feeding_cost | 1.18 | 240.2; `Tillväxt mot ranson` | (weighted field metabolism 4.97 - SDA 1.05) / RM 3.33. Field metabolism 7 260 MJ/t/yr = 1.85 x SMR, SDA included; cod 2.2 x (Chung 2019), the haddock and flatfish multipliers (68 % of the biomass) judged [S] |
| sda_fraction | 0.10 | 240.2 | cod 0.12, plaice 0.10-0.16 [S] |
| growth_cost c | 0.69 (0.46-1.01) | 241.2; `Tillväxt mot ranson/slope_decomposition` | top-down residual of the lab growth-ration slopes beyond F + U and SDA: cod 0.37 (own losses) or 0.49 (on the engine's 0.8 and SDA 0.10), haddock 0.84, plaice 0.75; weighted 0.69 [S] |
| somatic_production P | 0.74 /yr | 250 | SMS weight-at-age growth at the box's J/B [S] |

## Findings

1. **The excess sits on the production side, not in the metabolism.** At the trained fish's eating
   share 0.95 and fill 1, per tonne-year [D]:
   - action costs 5 693 (the field budget's FMR is 8 200 incl. SDA; the engine nets SDA from intake);
   - growth 5 421 = P 0.74 x the tonne price 7 326 (ec 3 600 + c 0.69 x (3 600 + 1 800)); of it,
     **2 757 is the synthesis cost c** and 2 664 the deposited structure;
   - refill of the reserve that leaves with losses 720;
   - need 11 835 against the field-implied 9 799 (Q/B 4.46). The 223.9 budget allots 2 900 to
     production; the engine spends 6 141 on it.

2. **c is the weighted lab residual; how much of it is synthesis is known for cod only.** Engine
   route per species: cod 0.49 (0.30-0.75), haddock 0.84 (0.52-1.33), plaice (flatfish proxy) 0.75
   (0.61-0.95); weighted 0.69 (0.46-1.01) (slope_decomposition tables/decomposition_stdout.txt) [S].
   The source reads the haddock and plaice residuals as "more likely an unmeasured feeding cost" and
   notes that a cost proportional to the surplus ration "cannot be told apart from a cost proportional
   to growth" (summary.md items 5, 59) [S]. The engine's feeding_cost comes from FIELD metabolism,
   which holds whatever the fish spend on feeding and on synthesising their field growth, and the
   223.9 budget behind Q/B 4.46 has no separate synthesis term for that reason. For cod (31 %) the
   field metabolism is measured (2.2 x SMR) and its c has an own-species basis; for haddock and the
   flatfish (69 %) both the field multipliers (1.4-2.0 x SMR) and the split of the residual are
   judgements. So a double count is possible for most of the biomass but not demonstrable [J].
   CORRECTION: a first version proposed cod's c for the whole group; a group value must be
   biomass-weighted over its members.

3. **The field metabolism used for feeding_cost is too low to hold the box's growth.** If the
   synthesis of field growth at P 0.74 is taken out of the 240.2 FMR (4.97 MJ/t/tick), the activity
   cost left is (4.97 - SDA 1.05 - synthesis 1.65) / 3.33 = **0.68 x RM, below resting** (1.0) [D]. Even
   at cod's c the remainder is 0.83-0.91 x RM. So 1.85 x SMR cannot contain routine metabolism, SDA and
   the synthesis of a 0.74 /yr growth together. The 1.85 rests on judged multipliers for 68 % of the
   biomass; the 223.9 note's central was 2.0 x SMR (8 200), cod alone 2.2 [S].

4. **Q/B 4.46 is itself a closure with the stock's P, not the box's.** It was built on P = Z 0.64
   (stock steady state). Section 250 set the box's P to 0.74 (younger fish). The same closure with
   0.74 gives 8 200 + 0.74 x 4 500 = 11 530 -> Q/B 4.4 mixed, **4.66 on the redistributed diet** [D].

5. **Fill 1 adds to it.** The trained fish sit at fill 0.99 all year (check 3); the field mean fill
   is 0.62 (start_fill). At fill 1 each tonne carries a full liver into the c price and every lost
   tonne takes a full liver with it (refill 720 against 446 at 0.62) [D]. This is the food surplus,
   not a parameter.

## What each lever does (Q/B needed at eating share 0.95; field 4.46, re-derived with the box's P 4.66)

| case | need MJ/t/yr | Q/B needed |
|---|---|---|
| library, fill 1 | 11 835 | 5.39 |
| c 0.49 (cod) | 11 035 | 5.02 |
| c 0 | 9 077 | 4.13 |
| P 0.64 (the budget's Z) | 11 102 | 5.05 |
| fill 0.62 | 11 212 | 5.10 |
| eating share 0.5 | 11 441 | 5.21 |
| c 0.49 + fill 0.62 | 10 514 | 4.79 |
| c 0.49 + fill 0.62 + P 0.64 | 9 923 | 4.52 |

No single term is far outside its source range; the budget fails because several sit at their
upper side at once (weighted c 0.69 with haddock and plaice residuals the source reads as feeding
cost, the box's P 0.74, fill 1), on top of a feeding_cost taken from total field metabolism, while
the Q/B it is judged against was closed with the stock's P and without a synthesis term [J].

## Source-consistent options (the user's decision; nothing applied)

Biomass-weighted c variants (haddock 38.6, cod 31.4, flatfish ~23 %, renormalised; the engine
route reproduces the source's 0.69 as 0.70) and the Q/B the engine needs at fill 1, eating share
0.95 [D] (the cod-only rows in the lever table above are arithmetic, not candidates):

| c per species | weighted c | Q/B needed |
|---|---|---|
| each species' own residual (the library) | 0.70 | 5.40 |
| cod own 0.49; haddock and flatfish at the bottom-up synthesis 0.33 (Wieser) | 0.38 | 4.83 |
| low end: cod 0.30, the others 0.2 | 0.23 | 4.56 |
| high end: cod 0.75, the others 0.6 | 0.65 | 5.32 |

- **A. Weighted c with the haddock and flatfish residuals split** into synthesis (the
  species-general bottom-up 0.2-0.6, central 0.33) and a feeding cost taken to sit inside the field
  metabolism behind feeding_cost: c ~0.38 (0.23-0.65), Q/B needed 4.8 (4.6-5.3) against 4.66 (the
  223.9 closure with the box's P). The split is a judgement for 69 % of the biomass, and the bottom-up
  0.33 is gross and partly inside measured SDA (the source's double-count warning), so if anything
  high. It only removes a double count if the judged haddock and flatfish field multipliers really
  hold that feeding cost.
- **B. Keep c 0.69 and take the synthesis out of feeding_cost.** Not possible with the 240.2 FMR
  (the activity cost would fall below resting); it would need a higher weighted FMR than the judged
  haddock and flatfish multipliers give.
- **C. Keep the parameters, raise the comparison Q/B** to what they imply (5.4). Above every
  published and budget value; the box's younger size mix argues for a Q/B above the stock's, but
  the SMS rations at the box's size mix (cod 4.6, haddock 2.8) do not reach 5.4.

The weakest evidence in the whole budget is the same for A and B: field metabolism and the
growth-ration split for haddock and the flatfish (69 % of the biomass), all judged. The fill-1
part (finding 5) belongs to the food question (the krill pool), not to these parameters. The
growth-ration slope test of 241.2 moves with A: at c 0.69 the energy slope equals the weighted
lab deposition 0.41 by construction; at a lower c it would sit above it.

## Update 2026-11-11: field metabolism of haddock and the flatfish

`Fältmetabolism kolja och plattfisk` (a-umlaut) found own-species field metabolism (otolith carbon,
Alewijnse 2022; North Sea plaice, Jones et al. 2023) for 0.90 of the biomass (was 0.32). Group FMR
8 890 (7 920-9 980) MJ/t/yr = 2.27 x SMR, against the 7 260 behind feeding_cost (240.2). Consequences
for this note (that folder's tables/budget_fmr_stdout.txt):
- Finding 3 is reversed at the central FMR: with the synthesis at c 0.69 taken out, activity is
  1.02 x RM (low 0.82, high 1.24), so option B is feasible: feeding_cost = (FMR - SDA - synthesis) / RM.
- Protein synthesis sits largely inside measured SDA and a wild fish pays the rest inside its FMR, so
  feeding_cost from total FMR plus a separate c is a double count by construction (finding 2, now
  supported by evidence rather than only possible).
- The field-closure Q/B with the new FMR and the box's P is 4.94 (4.55-5.38). With option B the engine
  needs 4.76 at the field fill 0.62 and 5.05 at fill 1 (c 0.69): the parameter side closes; the
  remaining excess is fill 1, the food surplus.
- Group c bound at the central FMR: 0.58 (SDA 13 %) - 0.72 (SDA 10 %); the library's 0.69 sits at it.
  Option A of this note (splitting c with a judged synthesis share) is superseded by option B.

## Correction 2026-11-11: M1 in the refill

The refill used M1 ~0.0002 /yr (a per-tick value read as per year in closure.py, copied as LOSS 0.40);
the library's M1 is 0.29 /yr, so LOSS is 0.694 and the refill 1 249 at fill 1 / 775 at fill 0.62 (was
720 / 446). `scripts/levers.py` and `Fältmetabolism kolja och plattfisk/scripts/budget_fmr.py` are fixed.
Every "Q/B needed" above rises by ~0.24 (library at fill 1: 5.39 -> 5.63). With option B (section 267,
feeding_cost 1.02) the engine needs 4.91 at the field fill (was 4.76) and 5.29 at fill 1 (was 5.05),
against the field closure 4.94 (unchanged: it has no refill term): the parameter side still closes, now
with no margin; at the low FMR (FC 0.82) 4.49 / 4.87.
