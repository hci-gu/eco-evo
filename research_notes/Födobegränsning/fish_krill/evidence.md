# Food limitation of fish on fish prey, krill and zooplankton: evidence trail (2026-11-07)

Research only. Nothing in code, library, tests, manifest, AGENTS.md or the resume was changed. No model
output set any value; model numbers (library as of 2026-11-07, from the task brief) are used only for
comparison.

Tags: [F] stated in the source (full text read), [A] abstract or search snippet only, [D] derived here
(arithmetic in `scripts/field_K.py`, output in `tables/field_K_stdout.txt`), [J] judgement.

Units: densities t/km2 wet, box mean unless stated. a in km2 per t of predator per 6 h tick. h in ticks
(E_prey / ceiling). Single-pair half-saturation K = 1/(a h). x = a h B_vis = B_vis/K. Saturation x/(1+x).

Earlier work reused, not redone:
- `../../Födonivå och mättnad i fält/` (summary.md, tables/B_mechanisms_notes.txt): field feeding levels,
  gut buffer, Jensen/patchiness arithmetic, the P2 "capture window" proposal.
- `../../Attackrater ur mötesteori/addendum_validation_behaviour.md`: the first conversion of Rindorf &
  Gislason 2005 to engine units (central vulnerable-clupeid K 0.062 t/km2).
- `../../Krill i Skagerrak/pairs/` and `mackerel_filter/`: the krill a's and the mackerel copepod a (already
  from Macy et al. 1998 clearance).
- `../../Täthetsberoendets styrka per grupp/pelagic_fish.md`: herring and mackerel density-dependent growth.

---------------------------------------------------------------------------------------------------------

## 1. Gadoids on fish prey: local functional responses

### 1.1 Rindorf & Gislason 2005, whiting (J Exp Mar Biol Ecol 324:1-19)

Source: DTU Orbit postprint read in place,
`research_notes/Attackrater ur mötesteori/data/gap_validation/rindorf2005/postprint.txt`.

What was measured [F]:
- 5 North Sea locations, early September; bottom 44-137 m, 8.2-16.7 C.
- Hauls every 4 h for 48-72 h. Intake from 48-h stomach series and an evacuation model, as P = intake/Wmax.
- Prey density = trawl catch of prey-sized fish per trawl hour (catchability unknown).
- Gear (Rindorf 2002, J Fish Biol 61:579, search snippet [A]): demersal EXPO trawl, 1 h hauls at 3.5 kn.

Results [F], Table 2 (Type II unless marked):
- Clupeids: plateau K_R 0.10 (25-30 cm whiting) and 0.17 (30-35 cm) of Wmax; half-saturation 59 and 95
  per trawl hour (Type III 20-24).
- Gadoid prey (juvenile whiting, haddock, pout): plateau 0.12 and 0.063; half-saturation 2907 and 965
  per hour (Type III 514).
- "a saturation level was reached for all types of fish prey, none of the levels corresponded to the
  maximum digestive capacity". No single prey group exceeded 20 % of Wmax.
- Saturation held from 20 herring per hour (lowest saturated catch) to more than 9000.
- Intake of other food: 8-22 % of Wmax, about constant across sites (varied < 3x while fish intake varied
  17-36x), uncorrelated with fish intake.
- The multispecies Type II (shared handling) fitted "very poorly" (r2 0.27-0.45). Example: at the
  herring-rich site whiting still ate many gadoids.
- Fish eaten "almost exclusively during dawn and dusk"; midday and night intake low.
- No aggregative response; predator and prey catches slightly negatively correlated.
- The authors' explanations for the sub-Cmax plateau: twilight-limited vulnerability of schooling prey,
  and partial spatial overlap inside a haul ("predators which overlapped with the prey became saturated
  while predators in other areas fed on other prey").

Conversion [D] (`field_K.py` section 1). K = q_rho x w / (A x c):
- A: 1 h x 3.5 kn = 6.48 km x wing 15-25 m [J] = 0.097-0.162 km2 (central 0.13).
- Clupeids: q_rho 59-95, w 10-20 g (12-14 cm herring/sprat, survey length-weight), c 0.05-1 (central 0.2,
  bottom trawl in a 44-137 m column) [J].
  K = **0.043 t/km2 of vulnerable clupeids (0.0036-0.39)**.
- Gadoid prey: q_rho 965-2907, w 3-8 g (7-10 cm), c 0.3-1 (central 0.5) [J].
  K = **0.13 t/km2 (0.018-0.80)**.
- The tow is 6.5 km long, so this is a km-scale (cell-scale) response.
- Catch efficiency c is the dominant uncertainty (x20). The addendum's earlier 0.062 used A 0.09 km2.

### 1.2 Kempf, Floeter & Temming 2008 (MEPS 367:295)

Read in place (`Attackrater ur mötesteori/data/validation/kempf2008.pdf`).
- Cod and whiting, ICES Year-of-the-Stomach data, ICES-rectangle scale (~55 x 55 km) [F].
- The response is relative stomach content vs RELATIVE prey abundance and the Schoener overlap index. No
  absolute density, so no K in t/km2.
- Shape [F]:
  - a population-level Type III "virtual refuge" below relative abundance ~0.1 and overlap ~0.1;
  - a rapid rise above that, then a plateau from relative abundance and overlap ~0.2 upwards;
  - the authors call this "an early saturation effect for abundant and well available prey types".
- The Type III arises largely from overlap falling as prey become scarce (passive), not from individual
  switching. Overlap is therefore a multiplier on the whole response at large scale.

### 1.3 Temming et al. 2004 (J Fish Biol 64:1351), whiting and haddock on sandeel

Read in place (`temming2004_exhibit.pdf`).
- A 15 x 20 nmi area, GOV 30 min at 4 kn. Whiting aggregated where sandeel catches were highest (an
  aggregative response) [F].
- Stomach contents rose fast at night (buried sandeel taken) [F].
- No functional-response fit. An aggregative response puts local prey density above the area mean. This
  contrasts with Rindorf 2005 (negative correlation) [F].

### 1.4 Fall et al. 2021 (Ecography 44:802), Barents cod on capelin at 2 km

Read in place (`fall2021_ecog.pdf`, CC BY). Fig. 4a and 5 read from the rendered page.
- ~2000 cod stomachs; capelin density from 1.9-km acoustic segments (whole column, ind/km2) [F].
- Consumption "increased sharply with capelin density before levelling off". "The main increase ... occurred
  below median density" [F].
- Median capelin density 0.018e7 = 1.8e5 ind/km2 (Fig. 5, dashed line) [F, read from figure].
- Cod density had no effect on consumption, i.e. no interference or local competition [F].
- Capelin depth (distance above the bed) was a stronger driver than capelin density [F]. "The slow digestion
  rate also implies saturation of the functional response at capelin densities similar to those observed"
  [F, citing Fall & Fiksen 2020].
- Median capelin ration among cod that ate capelin: 0.73 %BW/d (0.04-2.6) [F].
- Digitised partial effects (log link) [D, by eye]:
  - -0.6 at 5e3 ind/km2, -0.4 at 1e4, -0.12 at 5e4, 0 at 1.8e5;
  - +0.30 at 1e7, +0.45 at 5.5e7.
- The curve is log-like, not Holling. Intake at the median is exp(-0.45) = 0.64 of the intake at 300x the
  median.
- A Holling II least-squares fit gives K = 8.9e3 ind/km2 with a poor fit [D].
- Capelin 8-20 g [J]:
  - Holling K 0.07-0.18 t/km2;
  - end of the main rise 0.4-1.0 t/km2;
  - median 1.4-3.6 t/km2.
- Reading [J]: a log-like curve is what a mixture of saturated and unsaturated individuals gives (vertical
  overlap varies). A box-mean K therefore depends on the averaging scale.

### 1.5 Barents cod on capelin at stock scale

- MULTSPEC (Bogstad et al. 1997, via the earlier note): K 1.57 t/km2 of suitable food on ~100 km areas.
- ICES WKCAPELIN 2023 (read in place) [F]:
  - Type III consumption of maturing capelin by immature cod, Jan-Mar, chosen over Type II;
  - the old Type II was "essentially a type I model ... no saturation within the observed range";
  - the new curve "levels out at around 500 000 tonnes of capelin biomass";
  - maximum capelin intake of a 1 kg cod is ~0.15 kg/month, "on the low side" of total cod ration.
- [D, J] Over an overlap area of 0.5-2 x 10^5 km2, levelling occurs at 2.5-10 t/km2.
- **Scale dependence [D]:** K (or levelling) is 0.1 t/km2 (2 km Holling fit) < 0.4-1.0 (end of main rise,
  2 km) < 1.6 (100 km) < 2.5-10 (stock levelling). Coarser averaging dilutes saturation with non-overlap.

---------------------------------------------------------------------------------------------------------

## 2. Stock-scale food limitation of the predators (growth/condition vs prey)

| predator | evidence | tag |
|---|---|---|
| whiting, North Sea | Lauerburg et al. 2018 (MEPS 594:213, read in place): forage fish down 21 % (IBTS) / 42 % (SMS) in 2000-07; length-at-age down in parallel and back up after 2007; observed stomach content ~60 % lower; asymptotic length vs prey ("if prey abundance reaches a certain level, the growth rate ... does not increase any further"). Growth-implied stomach contents 2008-14 vs 2000-07: +12.9-16.8 %, +3.9-7.6 % after temperature correction | F |
| whiting, elasticity | dlnC/dlnN = 0.19-0.73 uncorrected, 0.06-0.34 T-corrected -> Holling x = N/K 0.4-16, saturation 0.27-0.94 at stock scale | D (`field_K.py` 3) |
| saithe (0.80 of the water column) | Cormon et al. 2016 (MEPS 542:141, Archimer postprint, `data/saithe_growth/`): growth coefficient + with Norway pout TSB (r = 0.41; food 17 % deviance) and - with saithe abundance (density dependence 30 %); no temperature effect | F |
| cod, Barents | weight-at-age fell after capelin collapses (Mehl & Sunnana, via Lauerburg 2018); Fall 2021: no local competition | S / F |
| cod, Baltic | Neuenfeldt et al. 2020: feeding level fell to growth-limiting levels with loss of benthic and pelagic prey (via earlier note) | A |
| plaice, dab (contrast) | Johnson et al. 2015: fullness flat across a trawl gradient (earlier note) | F |
| NEA mackerel | Olafsdottir et al. 2016: weight-at-length and cohort growth fall with mackerel and NSS herring stock size (r2 0.89), latest cohort 34 % of max growth (pelagic_fish.md) | A |
| NSS herring | Homrum et al. 2022 (Front Mar Sci): length-at-age fell 2003-10 in all ages, "density dependent limiting factors"; condition tied to zooplankton timing | A (search snippet) |
| North Sea herring | Brunel & Dickey-Collas 2010: no significant within-stock density effect on growth across 15 stocks; Downs/Banks components had inverse weight-at-age vs stock (pelagic_fish.md) | F |
| herring, mackerel at station scale | Bachiller et al. 2016 (read in place, CC BY): stomach fullness models retained temperature, length, season, NOT zooplankton biomass. Debes et al. 2012 (ICES CM M:07): herring fullness highest in the water mass with the lowest zooplankton | F |

Reading [J]: predators on fish prey and on plankton show food limitation at stock scale (growth tracks prey),
while their local (haul/station) responses are flat or saturated. The two are reconciled by
- plateaus well below Cmax (time/overlap windows, 1.1), and
- the scale dependence of K (1.4-1.5): saturation where predator and prey overlap, none where they do not.

---------------------------------------------------------------------------------------------------------

## 3. Zooplankton and krill pairs

### 3.1 Mackerel on Calanus (Macy et al. 1998, MEPS 172:89; read in place)

- Large Calanus (C5-C6, 300 ug DW): Ivlev curve, maximum 1970 copepods fish-1 per 5 min, 90 % reached at
  37.3 /L [F]. Ivlev half-rate 11.5 /L [D].
- Particulate clearance peaked at 19.8 L/min at 3.7 /L [F].
- Feeding started at 4.4-14.7 /L and stopped below 0.4 /L [F].
- The filter switch was at 24-64 /L; herring switch at 18 /L (10-34; Gibson & Ezzi 1990, cited by Macy) [F].
- These are instantaneous (5-min) rates. 1970 x 300 ug per 5 min is ~3.9 %BW DW per hour for a 184 g
  fish, far above any daily ration. The lab plateau is a capture/filter limit, not the daily ceiling the
  engine uses [D].
- The box pool, 9.7 t/km2 over 0-25 m, is 0.39 g WW/m3. As large Calanus that is 0.19-0.32 ind/L, below
  Macy's 0.4 /L stop threshold. At box-mean concentrations a mackerel is deep in the linear part of its
  instantaneous response [D].
- The model's a (0.00084) already comes from this clearance (`Krill i Skagerrak/mackerel_filter/`).
- Holling with the daily ceiling then gives K 17.5 t/km2, x 0.55: food-limited at the box mean.
- Feeding happens only in patches (> 0.4-4 /L, i.e. > 0.6-6 g WW/m3, 2-20x the box mean) [D, J].

### 3.2 Herring/planktivores on zooplankton

- No field functional response with an absolute half-saturation was found.
- Lab: biting is rate-limited at high concentration and herring switch to filtering at ~18 /L (Gibson & Ezzi
  1990, 1992) [A]. That concentration is ~50x the box mean.
- Field: no local fullness-zooplankton relation (Bachiller 2016; Debes 2012) [F]. Stock-scale density
  dependence exists in NSS herring [A] and partly in North Sea herring [F].
- The model's x = 6.2 (a from encounter theory) cannot be tested against a field K (gap).

### 3.3 Fish on krill

- No field or lab functional response of any fish on euphausiids was found (gap; one search pass plus the
  krill notes).
- Capture probability [F, review]: Juanes, Buckel & Scharf 2002 (`data/juanes2002/`) say "Usually about
  half of piscivore attacks on average are successful compared with 70-80% for planktivorous fishes". The
  krill pairs already carry p = 0.5 (0.3-0.7) inside a (`Krill i Skagerrak/pairs/evidence.md`).
- Krill avoidance [A]: an ICES JMS 2024 study ("Extensive avoidance behaviour of krill from predators and
  nets", doi 10.1093/icesjms/fsae123) reports coherent diving "spanning tens of meters at speeds of 10-27 cm
  s-1" when fish schools pass by day. This acts on the encounter (a) and on local density, not on the daily
  ceiling.
- Diel window [F via krill notes]:
  - Oslofjord whiting and pout take krill day and night with no difference in occurrence or digestion
    state; the predators follow the layer (Onsrud et al. 2004).
  - So the twilight window that limits whiting on clupeids (Rindorf 2005) does not apply to krill [J].
- Saithe under krill swarms did not eat the krill (Kaartvedt et al. 2005) [F via krill notes]. Encounter is
  not consumption for the water column, which is a preference/diet question.
- Layer concentration [D]: 5 t/km2 in a ~30 m day layer is 0.17 g/m3, ~1.7 ind/m3 at ~0.1 g. Oslofjord
  80-100 m: 10.9 ind/m3 (krill notes).

---------------------------------------------------------------------------------------------------------

## 4. Capture window: on a, or on a and 1/h together?

Algebra [D]:
- Field response: I = K_R Wmax N/(K + N).
- Engine single-prey response at eat share T: I = T a N/(1 + a h N), h = 1/Cmax.

| placement | engine form | plateau | half-saturation | implied a from field K |
|---|---|---|---|---|
| on a only (capture success p) | p a N/(1 + p a h N) | Cmax | 1/(p a h) | - |
| on a and 1/h together (window tau = time/overlap share) | tau a N/(1 + a h N) | tau Cmax | 1/(a h), unchanged | a = 1/(K h) |
| on 1/h only (capture-limited ceiling) | a N/(1 + a (h/tau) N) | tau Cmax | tau/(a h) | a = tau/(K h) |

Evidence:
1. **A plateau at 0.06-0.17 of Wmax rules out "on a only".** A multiplier on a moves only the knee; the
   plateau stays at Cmax (Rindorf 2005, flat from 20 to 9000 herring per hour) [D].
2. **Capture success per attack (~0.5 piscivores, 0.7-0.8 planktivores; Juanes 2002) belongs in a.**
   Handling and pursuit are seconds against digestion of 1-4 days (b/c ~1/400; earlier note, Jeschke 2002),
   so it cannot set a plateau [D].
3. **What sets the plateau is a window on the whole response** [F, J]:
   - fish prey are taken in twilight only (Rindorf 2005);
   - fish and other-food intakes are independent;
   - the authors' overlap explanation is a within-haul share of predators in contact with schools.
   Kempf 2008's overlap term multiplies the whole stomach-content response. The gut buffer means a window
   costs little unless captures inside it are rare (earlier note B_evacuation). So tau is "few discrete
   captures of schooling prey per twilight, and partial overlap", not "hours open".
4. In the engine, tau on the whole response is the eat share T itself (shared handling:
   demand_j = B T p_j lambda_j / (1 + sum p h lambda)). A field tau per prey group therefore maps to a cap
   on T for that prey group, not to a or h [D].
   - Rindorf's poor multispecies fit and the independence of fish and other-food intake argue against ONE
     shared handling budget across fish and non-fish prey: per-prey-group windows add, they do not
     compete [F, J].
   - Stated as evidence, not as a code proposal.
5. Under "on 1/h only" the field K implies a = tau/(K h) [D]:
   - whiting -> planktivores: a 0.011-0.019, model/field 1.1-1.9;
   - under "whole response": a 0.11, model/field 0.19;
   - the evidence (4.3) favours "whole response".

---------------------------------------------------------------------------------------------------------

## 5. Per-pair translation (`tables/field_K_stdout.txt`)

Visible prey [D]:
- Planktivores 4.72 t/km2 x stage_access at J/B 0.417 (library start_share):
  - whiting 0.079 -> 0.37 t/km2;
  - water column 0.377 -> 1.77;
  - benthivores 0.365 -> 1.72;
  - mackerel 0.010 -> 0.05.
- Krill 5 t/km2 (no stage access). Zooplankton 9.7 t/km2.
- Hiding (rest) and visibility floors are ignored, so these are upper bounds on visible prey.

Pairs:
- whiting -> planktivores:
  - model K 0.23 visible t/km2, x 1.6 (sat 0.62);
  - field K 0.043 (0.004-0.39) on 12-14 cm clupeids, x_field 8.6 (1.0-100);
  - implied a (whole-response reading) 0.11 (0.012-1.3), model/field 0.19 (0.02-1.7);
  - stock-scale x 0.4-16 (Lauerburg);
  - tau 0.10-0.17 (clupeid plateau; Rindorf).
- whiting -> benthivores (Rindorf gadoid prey; pout is in the planktivore pool and juvenile whiting is
  cannibalism, so this is mixed):
  - field K 0.13 (0.018-0.80), implied a 0.054 (0.009-0.39), model 0.21, model/field 3.9 (0.5-24);
  - tau 0.06-0.12.
- water column -> planktivores:
  - no local fit;
  - stock-scale growth tracks Norway pout biomass (Cormon 2016), so the pair is not far into saturation at
    stock scale [J];
  - model x 2.3.
- benthivores -> planktivores, using the Barents cod-capelin proxy (different prey habit; flagged):
  - field K 0.6 (0.11-1.57) -> a 0.0070 (0.0027-0.038), model 0.0198, model/field 2.8 (0.5-7.4);
  - x_field 2.9 (1.1-16) vs model 8.1.
- mackerel -> planktivores: no field data; model x 0.12 (stage access 0.01).
- planktivores -> zooplankton: no field K; model x 6.2.
- mackerel -> zooplankton: a from lab clearance (already); x 0.55.
- krill pairs: no field K; model x 0.9 (mackerel) - 34 (whiting).

Patchiness [D, J]:
- Holling's K is a local (km-scale) quantity. Box-mean K grows with averaging scale (1.4-1.5).
- For the 1 km cell, the right K is the km-scale one (Rindorf, Fall).
- The within-cell share of predators in contact with schools (Rindorf's overlap) belongs in tau, not K.
- Krill and Calanus are layered and swarming at sub-km scale (2-20x the box mean). Where the predator
  follows the layer (whiting/pout on krill, Onsrud), saturation is more likely than the box-mean x
  suggests; where it does not (mackerel needs > 0.4 /L), the box mean is a poor guide.
- Jensen loss at x ~ 1 is 2-15 % for CV 0.3-1 (earlier note, B_jensen).

---------------------------------------------------------------------------------------------------------

## 6. Per-predator saturation, and what the per-pair a comparison can and cannot say

**Saturation is per predator, not per pair [D].**
- Under shared handling the denominator is 1 + sum_k p_k h_k lambda_k, so a predator's total intake is
  saturated by its most saturating prey.
- At today's densities that prey is:
  - krill for whiting (x 34), the water column (x 20) and the benthivores (x 12);
  - zooplankton for the planktivores (x 6.2).
- So in the engine these predators sit on their ceiling whenever they forage, whatever the fish-prey x
  (1.6-8). Realised intake is then about T x ceiling (reserve_room appetite, h = 1).
- The field level for whiting is 0.25 (0.10-0.35) of Cmax (earlier note). Rindorf's per-group plateaus
  give fish 0.10-0.17 plus other food 0.08-0.22, i.e. 0.18-0.39.
- That is a statement about the whole-response share T per prey group, not about a or K.

**The per-pair a ratio is not discriminating [D].**
- Under reading A (tau on the whole response) the library a 0.0212 matches Rindorf only if the bottom
  trawl's clupeid catch efficiency c is ~0.04, other inputs central. Required K = 1/(a h) = 0.228; field
  K = 0.043 x 0.2/c.
- Under reading B (tau on 1/h) it matches at c ~0.2-0.4. Required K = tau/(a h) = 0.023-0.039.
- c is unknown within 0.05-1. The flat plateau from 20 to 9000 herring per hour fits both readings.
- The gut-buffer argument (a window costs little unless captures inside it are rare) leans towards a
  capture-rate cap inside the window, i.e. towards B.
- The earlier addendum's "passes" (central a 0.0063 vs 0.0056) and this note's 0.19 differ because the
  library moved, not the field:
  - a is now per visible tonne under stage_access, against a pool share then;
  - h went from 279.4 to 207;
  - pool 3.33 -> 4.72.
- Verdict on a for whiting -> planktivores: consistent within the c span, low confidence.

**Mackerel shape caveat [D, J].**
- Macy's initiation threshold (4.4-14.7 /L of large Calanus) is 15-75x the box-mean concentration (0.19-0.32
  /L if the whole pool were Calanus).
- In the lab, mackerel feed only in patches. A Type II on the box mean cannot represent that: it is
  within-cell patch structure, not a or K. Recorded as a caveat, not a proposal.
