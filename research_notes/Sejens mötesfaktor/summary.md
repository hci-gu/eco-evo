# Saithe's encounter-realisation factor (2026-11-22)

Written by the research subagent; saved by the main agent. Nothing in the library, code, tests, manifest, AGENTS.md or
the resume was changed, and nothing was committed.

- **Trail:** `evidence.md` (E1-E7) and `lit_evidence.md` (literature subagent).
- **Scripts:** `scripts/`. Their stdout is in `tables/`.
- **Downloads:** `data/README.md` lists them with their licences.

## Verdict

**No clean water-column anchor exists. One field dataset gives a probable upper bound for the fish pair. That bound
does not support raising section 296's 0.16, and it leaves the potential ration short of the field Q/B.**

**Data.** Saithe on Norway pout, NS-IBTS Q1 2026 (R/V G.O. Sars, February, 57-61 N, beds 80-169 m). There are 203
saithe stomachs with prey weights (median 46 cm), with the pout catch of the same 29 hauls on the library's own pout
convention (Walker GOV q). This is Rindorf & Gislason's design applied to saithe.

**Fit.** The Type II low-density slope over the pre-296 chain (a 0.02348 x saithe 1.15 x pout access 0.95 x Feb light
0.815 x q(T) 0.887 x bed-class factor) gives:
- **0.081**, haul bootstrap 0.022-0.19 [D on F];
- T/E corners 0.065-0.098;
- content capped at FullStomWgt 0.065;
- GOV hauls only 0.056;
- JTDS hauls only 0.31.

The field plateau is 0.82 of the GER Cmax.

**Net feeding [D on F, J].**
- 96 % of the stomach pout is DigestionStage 0 (other prey 11 %), and it is 8-11 cm, the codend's small pout.
- 21 fish hold > 5 %BW of pout (max 35 %BW). 19 of them come from hauls with > 10 kg pout caught, and 20 of them
  carry more prey than their recorded FullStomWgt.
- Hauls 152 (no pout caught) and 150 hold only stage-1 pout.
- The top hauls imply 1.2-3.6 %BW/d of pout alone, against the field Q/B of 0.78 %BW/d of all food.
- The slope is therefore a PROBABLE upper bound, conditional on stage 0 = fresh [J] and on Walker q. Daytime-only
  sampling points the same way.
- Errors in the haul-density proxy flatten the slope and partly offset this. The pool-to-pout step in the chain
  (herring barely reached by near-bed saithe) would lower the factor [J].

**Floor.** Pout past stage 0 alone gives 0.001, which is not a value either.

**Catchability corner.** At GOV q = 1 for pout (not the library's convention) the bound moves to ~1.0. The bound is
as much about pout q as about encounter.

**By pair type:**
- **fish (pout):** probable upper bound ~0.08 (<= 0.19), as above;
- **krill:** no anchor (no krill density at any saithe haul; Smith & Smith 2020 find a parameter-less Type IV for
  saithe on euphausiids);
- **benthivores, whiting, mackerel prey:** no data;
- **hake, spurdog:** no paired density in t/km2 found. Lopez-Lopez 2015 (hake) gives a Type III threshold only;
- **ling, pollack, anglerfish:** not searched.

## budget_gate on copies (consistency only; 298 block, predatory_fish, field Q/B 2.83)

| copy | potential x Q/B | short months | new OFF |
|---|---|---|---|
| live 0.16 | 1.10 | 5 (0.85-0.97 Dec-Apr) | - |
| all WC pairs 0.056 / 0.08 | 0.61 / 0.76 | 12 / 12 | none |
| all WC pairs 0.19 | 1.20 | 2 (Jan-Feb 0.93-0.96) | none |
| all WC pairs 0.30 | 1.47 | 0 | none |
| all WC pairs 1.0 (q = 1 corner) | 2.10 | 0 | none |
| planktivore pair only 0.08 / 1.0 | 0.96 / 1.76 | 7 / 0 | 1.0: WC -> krill OFF in both diet gates (0.113 vs booked 0.370) |

The engine's a's were printed per run, and the live run reproduces 1.10. A common factor leaves the diet shares and
the prey-side ratios unchanged.

## What this means [J]

- On the library's pout convention, the field saithe-pout data put the fish-pair factor at or below ~0.1-0.2. A
  factor in that range leaves the water column at or below its field ration even with T = 1, every prey attacked and
  no depletion.
- So the 296 level is not the only candidate for the 298 shortfall. Two other candidates were not tested here and
  stay open:
  - the depth/vision term (research_notes/"Djupgradientens bärare");
  - the pout q convention, a library-wide choice that sets every predator's pout encounter at once.
- A cleaner test needs stomachs free of codend feeding, night or dawn hauls, or the 1980s Q3 weighed stomachs paired
  with a Q3 survey that DATRAS does not hold.

## Caveats

- One survey, one month (February), 29 hauls.
- The digestion-stage code meaning is [J].
- The bed-class factors and saithe's share of the a come from the library's chain.
- The pout access 0.95 is [S].
- The sharing over saithe's prey types (for the all-pairs copies) is [J].
- Smith & Smith 2020 is public domain in the USA only: not to be committed.
