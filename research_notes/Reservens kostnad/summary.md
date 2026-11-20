# The cost of carrying reserve (2026-11-20)

Research only: a main research agent and three literature subagents; saved by the main agent.
- Nothing in the library, manifest, code, tests, AGENTS.md or the resume was changed. No model was run and nothing
  was committed.
- No value was chosen from model output. The frozen fix105 fills and the "potential ~2x need" are yardsticks only.

Files:
- **Trail:** `evidence.md` (M model facts, D arithmetic, L literature, synthesis).
- **Subagent trails:** `lit_risk.md`, `lit_cost.md`, `lit_excess.md`.
- **Tables:** `tables/mechanisms.csv` (sized summary); one row per sourced number in `tables/risk_raw.csv`,
  `cost_raw.csv` and `excess_raw.csv`.
- **Scripts:** `scripts/model_reasons.py`, `optimal_fill.py`, `evidenced_sizes.py`, `derive.py`, with output in
  `tables/*_stdout.txt` and `derived_out.txt`.
- **`data/`:** untrusted downloads, not to be committed.

Tags: [F] full text, [A] abstract, [S] secondary, [D] derived, [J] judgement. PROXY = not a model member species.

## Question

Frozen fix105 fish sit at fill ~1 all year. Field fish are lean in spring:

| group | spring | autumn |
|---|---|---|
| benthivores | 0.33 (May-Jul) | 0.81 (Sep-Nov) |
| whiting | 0.30 (May-Jul) | 0.73 (Sep-Nov) |
| planktivores | 0.20 (Apr) | 0.96 (Aug) |
| mackerel | 0.13 (May) | 1.0 (Sep-Oct) |

"Vilans avkastning" showed that a rest payoff does not lower fill. It named a second route: a cost of carrying
reserve. Is there field evidence for such a cost in these fish, of what size and by which mechanism? Could it
produce lean spring fish without a season switch?

## Per mechanism

| mechanism | species | size per unit fill | sign | tag |
|---|---|---|---|---|
| predation risk rising with reserve (bird theory: weight in flight) | all fish | not measured within fill 0-1 in any fish | - | J |
| lean fish swim worse, take more risk, are eaten more | cod; PROXY smolts, damselfish, sea bass | starved cod endurance x0.30 at K 0.54; PROXY relative risk 1.8-3.0 | opposite to a cost (low end only) | A, F/D |
| swimming vs condition within the natural range | cod K 0.68 -> 0.92 (wild spring -> autumn); PROXY Pacific herring with lipid x2 | Ucrit up or unchanged, sprint unchanged, cost of transport down | no cost | A, F |
| meal mass | PROXY trout, silversides | Ucrit -9 to -15 %, gone within hours to 3 d | gut, not reserve | A |
| gonad mass | PROXY mosquitofish | -0.37 % Ucrit per % mass. As an upper bracket for the inertia of lipid: -2 / -5 / -7 % at a full gadoid / herring / mackerel reserve | reproduction | A/D, J |
| reserve-tissue metabolism | gadoids (liver) | k ~ +0.05 of resting metabolism (RM) at fill 1 (range 0-0.11); cod SMR per kg unchanged with condition | small cost | D on PROXY F, A |
| reserve-tissue metabolism | herring, mackerel (lipid) | 0-0.03 RM | ~0 | S/D |
| buoyancy and drag | gadoids | < 0.005 RM; gas secretion <= 1 % of SMR | ~0 | F/D |
| buoyancy and drag | herring, sprat | 0 at the surface; -0.15 to +0.04 at depth (central -0.06) | saving | S/A/D |
| buoyancy and drag | mackerel (no swim bladder) | -0.45 to -0.02 (central -0.21); order of magnitude only | saving | A/F/D |
| storage cycling | all, PROXY | deposition loses 9-24 % of lipid energy, retrieval ~2 %; charged per MJ cycled, not per tick held | favours a steady reserve | A/D |
| reproduction | mature fish (section 279) | gametes and spawning fast | real drain, already in the model | S |

## Reconciliation with the farm and fattening evidence [J]

None of the following needs a carrying cost:
- saithe at fish farms with HSI x1.4-1.8 (Dempster 2011);
- farmed cod livers above 12 %, and cod at HSI 9-17 % eating 1.0-1.4 Cmax;
- saithe taking pellet meals in spring (Uglem 2020);
- immature cod on satiation feeding at HSI 9.3 % in April, against 4.4 % in spawners (Davie 2005 [F]).

On the library's convention (HSI 8 % at 60 % liver lipid = fill 1), cod fed in excess sit at fill ~1.0-1.26 all year
[D].

Optimal-reserve theory predicts a reserve below the maximum only when carrying it costs something. In birds that
cost is weight in flight. Fish are near-neutrally buoyant, and lipid is lighter than water: for herring and mackerel
it saves swimming cost, and gadoids pay only a small liver cost. For fish the theory therefore reduces to "store what
you can get".

The only regulation found is:
- a defended lower level, the salmonid lipostat (catch-up hyperphagia; Ali et al. 2003 [A], earlier notes [S]);
- a photoperiod-set seasonal level in herring (Berg 2024 [S]), already in the model as the section 271 set point.

## Why spring fish are lean [J on F/A]

The carrying cost is excluded as the reason. Acquisition and spawning explain part of the cycle; a residual remains
open.

- **Cod**
  - Immature fish on excess food show no spring low (Davie 2005 [F/D]; Karlsen 2006 [F via a fetch summary]). Under
    excess food the spring drop appears only in maturing fish, at spawning.
  - Wild cod eat almost nothing from December to April and lose 76-84 % of liver dry weight. 81-96 % of the winter
    loss is metabolism (Schwalme & Chouinard 1999 [A]).
- **Whiting**
  - HSI follows forage fish across years in fish of 30 cm and over (Lauerburg 2018 [F]); the 25-29.9 cm class goes
    the other way.
  - About 0.34-0.43 of the biomass spawns in the box ("Lek i rutan" [S]).
  - The SMS ration is flat at 0.42-0.49 of Cmax ([S]), so there is no winter intake fall to point to.
- **Haddock, saithe, Norway pout, sprat:** energy density cycles are "obvious in mature fish" (Pedersen & Hislop
  2001 [A]). Immature fish are covered only by analogy with cod [J].
- **Herring:** the one choice-like case, a light-set cycle under excess food. Adults also fast in winter and spawn.
- **Mackerel:** winter non-feeding and spawning. Choice against acquisition has not been tested.

**Open residual (whiting and immature gadoids)**
- "Lek i rutan" item 5: the May-Jul trough is unresolved. Somatic condition recovers within about a month, the liver
  within 3-6 months.
- The dynamic programme below gives a swing of 0.20-0.25 against the field 0.43. Field growth timing moves it by at
  most 0.04.
- This looks like an allocation order: grow first, refill later. That is choice-like but not a carrying cost [J].

## Model forms and their sizes [D]

Sources: `model_reasons_stdout.txt`, `optimal_fill_stdout.txt`, `evidenced_sizes_stdout.txt`.

**1. What a full reserve is worth to the model's learner.**
- The reward is the one-year mean of log((B ec + R)/E0): ticks 1460, integral reward, random start day.
- The growth ceiling is B g (s - u) p, with g derived at s* = 1.
- Holding fill 1 instead of the field annual mean is worth +0.16 to +0.36 log units, the equivalent of 4.5-9 months
  of somatic production. About half comes from the currency and half from the growth ceiling.
- A wild fish has neither term in this form.
- Existing costs of a full reserve in the model:
  - Losses take the reserve pro rata, so there is no cost in log E.
  - Benthivores pay a tonne price x1.13 at fill 1 (growth_cost c). This raises the energy per new tonne, but the
    tonnes grown are unchanged.

**2. Hazard form: predation M2 x (1 + beta s), or an extra loss mu s.**
- Mechanistic and payoff-based, so learnable: the fish lowers its fill by eating less or resting, with no season key.
- Beta the learner needs before it leaves fill 1 (to the field annual mean):

| group | beta | hazard of a full fish relative to an empty one |
|---|---|---|
| whiting | 2.7-3.0 | 3.7-4x |
| planktivores | 4.4-5.0 | about 5.4-6x |
| benthivores | 5.9-6.3 | about 6.9-7.3x |
| water column, mackerel (M2 ~0.02) | 61-91 | 62-92x |

- As an extra loss independent of M2 the learner needs mu = 1.2-2.1 /yr at fill 1.
- No positive beta is evidenced.

**3. Energy form: extra metabolism k RM s.**
- Paid out of the spare intake, so it binds only above k_crit = 1.3-2.5.
- The evidenced gadoid k (0.05) is 0.02-0.03 of k_crit; herring and mackerel have k < 0.
- Even for a field fish with 20-d food gaps, k = 0.1 moves the optimum by 0.01.

**4. A field fish's optimum.** Whiting dynamic programme on survival x mass, with an activity risk of 1.1.
- Without food gaps, a reserve has no insurance value.
- At beta 0 with gaps, the optimum is 0.85 (5-d spells) to 0.94 (20-d), flat when the potential is 2x need.
- With a seasonal potential equal to need on the annual mean (the field closure) at beta 0, the optimum runs from
  about 0.80 (June) to 0.93 (October).
- Beta 0.3-1 gives field-like means of 0.5-0.67, but a swing of only 0.20-0.25.
- At a potential of 2x need, any beta gives a flat fill.
- The phase of the cycle comes from the seasonal potential, not from the cost.

**5. A season switch.** A season-keyed cap, set point or beta would reproduce the field by construction. The project
rules exclude that form. The 271 set point is the one accepted exception, for herring only, where the cycle persists
under excess food.

## Verdict [J]

1. **No field-sized carrying cost of reserve exists for these fish.**
   - Gadoids pay about 5 % of RM at a full liver.
   - Herring and mackerel save on lift.
   - No predation cost of being fat has been measured; lean fish are the ones at risk.
   - The near-full autumn fills of herring (0.96) and mackerel (1.0) argue directly against a large cost. Gadoid
     autumn fills (0.71-0.81) would allow a beta up to about 0.3, so for gadoids the case rests on the absence of
     evidence, not on the autumn level.
2. **A cost cannot produce lean spring fish.**
   - A season-constant cost lowers the level and flattens the cycle.
   - The field shape fits seasonal acquisition plus spawning drawdown, with an open allocation-order residual for
     whiting and the immature gadoids.
3. **The model's fill ~1 does not come from a missing cost.** It comes from three causes, not ranked:
   - potential intake above need in every month (the acquisition gap of "Mättnad hos rovfisk");
   - the reward currency, which credits a stored MJ like lean tissue;
   - the growth ceiling's (s - u), which makes a full reserve the condition for growth, while field fish recover
     somatic condition before the liver.
   A field-sized cost would not lower the fill: the learner needs beta 3-7 against an evidenced beta of 0 or below.
4. **Options (not decided here).**
   - Acquisition stays the supported route, but not a closed one: per-prey plateaus, krill acquisition and a winter
     supply gap all come from earlier notes.
   - Two engine conventions could be examined instead of adding a cost, both outside this brief: whether the reward
     should count R one for one, and whether growth should depend on fill.
   - Either change would affect every trained policy. Neither is a cost of reserve.

## Settling datum

1. Monthly HSI or liver lipid of immature whiting, haddock and saithe, held one year on excess food at natural light
   and constant temperature.
   - A spring low there means choice or a clock; its absence closes the question as acquisition.
   - Cod already has this at low n (Davie 2005), and herring has it for condition (Berg 2024).
2. The Pedersen & Hislop 2001 full text: energy density of immature and mature fish by month for six North Sea
   target species.
3. For any risk form: escape performance or capture rate of wild gadoids or clupeids across the natural fill range,
   with liver or lipid measured per fish, against a fish predator.

## Load-bearing sources

| source | doi or handle | tag |
|---|---|---|
| Lapointe, Guderley & Dutil 2006 | 10.1086/498290 | A |
| Martinez et al. 2003 | 10.1242/jeb.00098 | A |
| Martinez et al. 2004 | 10.1242/jeb.01142 | A |
| Sewall et al. 2021 | 10.1139/cjfas-2020-0293 | F, PROXY |
| Tucker et al. 2016 | 10.1890/15-0564.1 | F, PROXY |
| Strand, Jørgensen & Huse 2005 | 10.1016/j.ecolmodel.2004.12.013 | F |
| Oikawa & Itazawa 2003 | 10.1046/j.1444-2906.2003.00675.x | F, PROXY |
| Davie 2005, PhD thesis, Stirling | hdl:1893/109 | F |
| Karlsen et al. 2006 | 10.1016/j.icesjms.2005.10.013 | F via a fetch summary |
| Schwalme & Chouinard 1999 | 10.1006/jmsc.1999.0458 | A |
| Lauerburg et al. 2018 | 10.3354/meps12533 | F |
| Pedersen & Hislop 2001 | 10.1111/j.1095-8649.2001.tb00137.x | A |
| He & Wardle 1986 | 10.1111/j.1095-8649.1986.tb05013.x | A |
| Huse & Ona 1996 | 10.1006/jmsc.1996.9999 | A |
| Plaut 2002 | 10.1046/j.1365-2435.2002.00638.x | A, PROXY |
| Lupatsch et al. 2003 | 10.1016/S0044-8486(03)00288-6 | A, PROXY |
| Dempster et al. 2011 | 10.1371/journal.pone.0015646 | S |
| Berg, Seljestad & Folkvord 2024 (in "Ljusstyrd aptit") | - | S |

## Gaps

- **No full text:** Lapointe 2006, Martinez 2003/2004 (no magnitudes, no HSI per group), Pedersen & Hislop 2001,
  Huse & Ona 1996, Bernard 1999.
- **Theory papers not opened, dois unverified:** Lima 1986, McNamara & Houston 1990, Houston et al. 1993, Witter &
  Cuthill 1993.
- **No data found:**
  - reserve vs escape or predation in whiting, haddock, saithe, pout, hake, sprat, mackerel or Atlantic herring;
  - SMR vs lipid in herring or mackerel;
  - Scomber scombrus density vs fat;
  - cod liver respiration per g;
  - a captive excess-food series for any gadoid other than cod.
- **Judgement inputs:**
  - liver respiration taken from a PROXY porgy;
  - the lift-to-drag ratio and efficiency behind the lift savings (uncertain by a factor of ~3);
  - the length and share of food gaps in the dynamic programme;
  - the quarterly growth timing of whiting;
  - mass as the currency of fitness.
- **Scope of the dynamic programme:** it covers whiting only. The beta values for the other groups come from the
  learner calculation alone.
