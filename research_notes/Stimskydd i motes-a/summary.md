# Schooling protection in the encounter a - summary (2026-11-11)

Research subagent; the main agent saved this text. Nothing changed in the library,
code, tests, manifest, AGENTS.md or resume; no training; nothing set from model
output (temporary library copies only in the session scratchpad). Per-source
numbers: `evidence.md` (E1-E34, model facts M1-M4). Grades: [F] field, [L] lab,
[A] abstract or search summary only, [S] secondary, [D] derived here,
[J] judgement.

Files (run from `scripts/`; stdout saved next to each):
- `pair_terms.py` -> `pair_terms_stdout.txt`, `pair_terms.json`: today's a of
  every in-layer-derived fish pair split by (prey species, day/night), with each
  term's light-weighted rule-A radius R_i. Execs research_notes/"Overlapp i
  fiskparen"/scripts/band_pairs.py up to its run block and reproduces A_band
  exactly (e.g. WC -> planktivores 0.017273).
- `school_factor.py` -> `school_factor_stdout.txt`, `school_factor.json`: S per
  term, the pair factor F and the new night/day for five readings; the Andersen
  2017 and Rindorf & Gislason 2005 cross-checks.
- `gate_school.py` -> `gate_school_stdout.txt`, `../tables/gates/
  budget_gate_<variant>.txt`, `../tables/gates/gate_school.json`: runs
  tools/probes/budget_gate.py with `library_path=<copy>`; its `library` run is
  identical to today's gate.
- `papers/`: Handegard et al. 2012 and Andersen et al. 2017 PDFs (Unpaywall
  helper), Unpaywall records for the DOIs without an open copy. Not committed.

```
S=<session scratchpad>
python3 -I pair_terms.py <repo> $S/tmp pair_terms.json          # ~30 s
python3 -I school_factor.py pair_terms.json school_factor.json
python3 gate_school.py <repo> school_factor.json $S/libs ../tables/gates   # ~5 s
```

## Question

The mackerel note (research_notes/"Makrillens antipredatorbeteende") found that
schooling protection (dilution, confusion, collective escape) is always on and not
tied to stopping feeding. Can it move out of the prey's hide action / visibility
floor into the encounter a of the fish-predator pairs? How does it enter the
encounter model, how large is it per prey group, does it double count with rule A
and the band overlap, and what would hide mean afterwards?

## Model (as given and checked)

- Sections 242 and 265: multi-prey Holling II with shared handling, demand_j =
  B T p_j lambda_j / (1 + sum_k p_k h_k lambda_k + w B), lambda = a B_vis; fish
  attack every prey they meet (p = 1).
- Section 269: every visual fish pair's a is built per prey ITEM: scan pi R^2
  v_rel p_capture N/t over the band overlap; R by rule A (R_sat = min(R0,
  2 L_pred), |C0| 0.7, R_nv 0.3 m); capture p on fish 0.2 (M2).
- Section 220.3(2): schooling was left to the prey's hide action and floor; the
  dispersed variant 0.012 was used instead of the school variant 0.0056 (M1).
- Hide: visible prey = B (1 - rest (1 - floor)); the fish-predator floors on
  planktivores are pair overrides (0.25), species floors 0.35 / 0.55 / 0.5 / 0.7
  / 0.7. A pair floor acts on predation AND on that predator's observation of the
  prey (`vis_floor_mat`, `_obs_vis_floor`).
- budget_gate applies stage access and the exposed share, no hide floor, and no
  dark_ratio (M4).
- depth_access_table is renormalised by the loader to an equal-cell mean 1
  (interactions.build_pair_depth_maps), so a factor on max_intake_rate suffices;
  a class-resolved schooling factor would need class-resolved term shares (not
  computed).

## Findings

### 1. Theory: how schooling enters a Holling encounter

Per-capita capture rate of one prey individual [D, following E6-E11]:
- Dispersed (the library form): rate_i = scan pi R_i^2 v_rel,i p_i n, n the prey
  number density.
- Schooled in groups of N: the predator meets SCHOOLS at pi R_s^2 v_rel,s n/N and
  takes k prey per school met: rate_s = pi R_s^2 v_rel,s k n / N.
- Ratio: S = rate_s / rate_i = (R_s^2 / (scan R_i^2)) (v_rel,s / v_rel,i) k /
  (N p_i).

Three components:
1. Detection and dilution -> a. Group detection grows far less than group size:
   reaction distance logarithmic in group size in sticklebacks (E8), detection
   distance doubling for a 100-fold increase in fish (E9); search time rises when
   the same prey are packed into fewer groups (E8). Attack abatement = avoidance x
   dilution (E6); group-dependent responses lower intake (E10, E11). R_s = school
   half-length + detection distance; day herring schools are 21-27 m long (E22).
2. Confusion -> capture success per attack -> k (inside a). Perch catch 0.20 per
   attack on single prey vs 0.03 on groups of two or more (E1), the strongest
   single datum; the library's p 0.2 IS the single-prey figure, so confusion is
   absent from today's a. Bass and cichlid success falls with shoal size (E4, E5).
   Confusion can change the magnitude of the functional response without changing
   its shape (E12).
3. Satiation per encounter -> k <= one meal; handling h unchanged. With k prey per
   school met and h per prey, intake = lambda_s k / (1 + lambda_s k h) (exact
   renewal-reward Holling form), so schooling is a pure factor on a as long as k
   and N do not depend on local prey density. Herring school size is set by
   location and time of day, not by local school density (E23) [F], which supports
   a constant factor. k scales with predator size: one meal is ~1 juvenile clupeid
   for a ~0.7 kg saithe and ~0.5 for a 0.15 kg whiting (inside the 0.3-3 bracket).

Per-capita risk (M2 per prey) is consumption / prey biomass, so below saturation it
scales with a: the factor on a IS the per-capita risk factor.

Two regimes break the pure a-factor [D/J]:
- School following: predators accompany a school for hours (E15, E16; offshore,
  Vestfjord). k then grows with residence time and intake is handling-limited, so
  a stops mattering. Not supported on the shelf: Baltic cod stayed near the bed
  outside the twilight windows (E17); North Sea whiting showed no aggregative
  response (E18).
- Twilight windows (E17, E18): clupeids are catchable mainly while schools
  dissolve or re-form. Andersen 2017: 1 of 194 sprat eaten by day although the
  schools sat at the cod's depth; per hour day / twilight = 0.0011 and day / night
  = 0.042 [D, confounded by satiation and by night overlap]. Rindorf & Gislason's
  clupeid plateau (K 0.10-0.17 of Wmax) equals a window of 2.4-4.1 h per day at
  the digestive rate [D reading]. A window caps the saturated intake on that prey
  (K < 1), which a factor on a cannot express; the 6 h tick cannot express a
  twilight peak, and dark_ratio >= 1 means "no light dependence" in the engine.

Sign of S: the geometry alone does not fix it over the parameter span. Small groups
with a large detection radius give S >> 1 (herring at N 2000, R_s 30 m, k 3: S ~
20). Field and lab data (E1, E4, E5, E13, E17, E18) put day-schooled clupeids on
the protected side, so S is capped at 1 [J]: an aggregation whose geometry would
raise risk is one the predator does not meet as a unit.

### 2. Separable from rule A and the band? Yes, if built per term

- S contains R_i, the rule-A radius of the dispersed item. It is not a post-hoc
  multiplier on rule A: it replaces the item-encounter term with the school term,
  computed with the same R_i. `pair_terms.py` gives each term's light-weighted R_i:
  by day on herring WC 0.84 m, whiting 0.47 m, mackerel 0.52 m; at night 0.30-0.39
  m.
- The band overlap is unchanged: schools and dispersed fish share the prey's
  biomass profile phi_j, and the field profiles already hold the day schools'
  depth.
- Light: at night clupeid schools dissolve (E22, E25, E26), so night terms keep
  F = 1 and rule A's night R. The school detection distance is not a rule-A
  quantity: a 20 m target is limited by water clarity and other senses, not by
  the 2 L_pred cap [J, 10-30 m].
- Double counts to avoid: (a) the hide floor must stop standing for schooling
  (finding 5); (b) stomach-derived night/day ratios (e.g. mackerel -> planktivores
  0.16) already contain prey schooling as realised: scale a and keep those ratios
  (user decision, section 269); (c) k replaces p_i for the schooled share - do not
  also lower p.
- Schooling hits the dominant term: today's a are day-weighted by rule A's light
  (pair night/day 0.06-0.21), and day is when clupeids school. Day terms carry
  0.84-0.94 of the WC and whiting planktivore pairs (M3).

### 3. Field numbers per prey group

The pair factor weights each prey species by its share of today's a (encounter-
weighted, `pair_terms.json`), the right weight for a; biomass weighting enters
through the members' N/t and pool shares inside the 269 derivation.

| prey (pool share) | sigma day / night (share in schools) | school N, R_s | S central (range) | grade |
|---|---|---|---|---|
| herring (0.55) | 0.9 (0.6-1.0) / 0 (0-0.2) | 25 000 (2 000-100 000), 15 m (10-30) | 0.14-0.46 by predator (0.005 to capped 1) | sigma [F] E22, E25, E26; N [F] E21; R_s, k [J] |
| sprat (small) | as herring | as herring | as herring | [F] E17, E25 |
| Norway pout (0.35) | 0.5 (0-0.9) / 0.25 (0-0.9) | 5 000 (500-50 000), 8 m | 0.28-0.81 | school echoes [A] E27; size and diel cycle [J] |
| blue whiting (small) | 0.2 (0-0.5) / 0.1 | 2 000, 8 m | 0.6-1 | [J] (layers, E30) |
| juvenile benthivores (prey stage) | 0 (0-0.3) / 0 (0-0.15) | 300, 4 m | capped 1 (0.04-1) | [J]; cod solitary or loose, haddock nearer the bed (E30) |
| whiting as prey | 0.1 (0-0.5) / 0.05 (0-0.25) | 300, 5 m | capped 1 (0.03-1) | [J] (E30) |
| mackerel (prey of WC) | 1.0 (0.8-1.0) day and night | 300 000 (3e3-3e6), 25 m | 0.24 (0.003-1) | schools 1-7000 t [A] E29; always schooled [L/F] |
| saithe (WC as prey) | - | - | not derived | mammal pairs only, out of scope (E34) |

k = 1 prey per school met (0.3-3) [J]: one meal, reached after several attacks at
0.03-0.24 success per attack (E1, E2); the section 220 value. The field bound on S
for day-schooled clupeids is stronger than the theory's central value: day / night
0.04, day / twilight 0.001 (E17) [D].

### 4. Do today's a or bookings already contain schooling?

- a: no. The section 220 school variant was rejected (M1); since 269 every fish
  pair is built per item with capture p 0.2, the single-prey figure (E1) (M2). The
  only field tests of the level do not discriminate: Rindorf & Gislason's
  half-saturation gives a 0.0063 (0.0005-0.10) against the school variant 0.0056
  and the dispersed 0.012 (E18); Engelhard 2008 matched the dispersed variant on
  sandeel (E19).
- Booked M2: yes, implicitly. The pair M2 are SMS partial M2 and stomach-based
  diet flows, realised predation in a world with schooling, so an a WITH schooling
  is the one that should reproduce the booking. Today's prey-side ratio for the
  planktivores (implied / booked 0.69) is reached without schooling, so some other
  factor may already absorb what schooling removes (p 0.2 vs the review value
  0.45, R, overlap; each uncertain by ~0.3-1 decade). That is a statement about
  the uncertainty of the a level, not a reason to drop or tune S.
- Stomach night/day ratios: yes, realised values with schooling included
  (mackerel -> planktivores 0.16, the krill pairs).

### 5. Consequence for the hide floors

If schooling moves into a, hide keeps only refuges that are not already in the
band profiles and rule A's light (depth, bed and dark are now in a) and that cost
feeding:

| group | real refuge with a feeding cost | evidence | implication for the floor vs fish predators |
|---|---|---|---|
| planktivores | depth / dark by day (vertical migration), already in the profiles and rule A; hypoxic deep water (fjord, not the box) | E20 [A], E17 [F] | no residual documented: pair floors (0.25) toward 1 [J] |
| mackerel | none; schooling always on in the open 0-50 m layer | "Makrillens antipredatorbeteende" | toward 1 (WC -> mackerel is OFF anyway) |
| benthivores (juveniles) | cobble / emergent epifauna (0-group cod); burial in sand (0-group plaice, partial); foraging suppressed under predators (plaice) | E31, E32 [A, analog, shallow nurseries] | a residual exists for the juvenile stage; size unmeasured on the box's 100-300 m mud [J] |
| whiting | 0-group under Cyanea medusae, Jun-Aug | E33 [A] | seasonal, not feeding-exclusive; weak support [J] |
| water column | none found | - | mammal pairs only |

Pair floors also set what that predator OBSERVES of the prey: raising the four
fish -> planktivore pair floors to 1 would also end the hide-observation coupling
for those observers, which the user kept (research_notes/"Gom och sinnen"). The
species floor (mammals and other observers) can stay. A user decision.

### 6. Effect on today's numbers (budget_gate on temp copies; a x F only)

Readings of F: central; low / high protection; clupeids only (pout and blue
whiting not schooling); field S (herring and sprat at S 0.01).

| pair | today a | F central | low / high | clupeids only | field S | night/day today -> central |
|---|---|---|---|---|---|---|
| WC -> planktivores | 0.0174 | 0.57 | 1.0 / 0.13 | 0.77 | 0.54 | 0.059 -> 0.109 |
| whiting -> planktivores | 0.0329 | 0.83 | 1.0 / 0.12 | 0.88 | 0.74 | 0.212 -> 0.265 |
| mackerel -> planktivores | 0.0248 | 0.71 | 1.0 / 0.40 | 0.71 | 0.55 | keep 0.16 (stomachs); the rule gives > 1 |
| benthivores -> planktivores | 0.0214 | 0.69 | 1.0 / 0.11 | 0.91 | 0.66 | 0.141 -> 0.219 |
| WC -> mackerel | 0.000137 | 0.24 | 1.0 / 0.003 | 0.24 | 0.24 | 0.92 -> 0.92 |
| WC -> benthivores, WC -> whiting, whiting -> benthivores, benthivores -> whiting | - | 1.00 | 1.0 / 0.55-0.77 | 1.00 | 1.00 | unchanged |

Lower bound if only herring and sprat school and are fully protected by day [D]:
F >= 1 - (their day share of a) = 0.71 (WC), 0.76 (whiting), 0.49 (mackerel), 0.87
(benthivores).

Gate, prey side (implied / booked M2; equal cells; * = OFF):

| prey / pair | library | central | high | clupeids only | field S |
|---|---|---|---|---|---|
| planktivores (total) | 0.69 | 0.50 | 0.15 | 0.60 | 0.47 |
| WC -> planktivores | 0.78 | 0.54 | 0.16* | 0.67 | 0.51 |
| whiting -> planktivores | 0.54 | 0.45 | 0.07* | 0.48 | 0.40 |
| benthivores -> planktivores | 0.61 | 0.42 | 0.07* | 0.56 | 0.40 |
| mackerel -> planktivores | 0.78 | 0.56 | 0.32* | 0.56 | 0.43 |
| WC -> krill (predator side) | 1.39 | 1.67 | 2.22 | 1.53 | 1.70 |
| benthivores / whiting / mackerel (totals) | 0.62 / 0.60 / 0.02 | 0.69 / 0.65 / 0.01 | 0.66 / 0.50 / 0.00 | 0.65 / 0.62 / 0.01 | 0.69 / 0.66 / 0.01 |
| OFF pairs | 4 (whiting -> benthivores, WC -> mackerel, benthivores -> whiting, seals -> WC) | same 4 | + 4 planktivore pairs | same 4 | same 4 |

Under shared handling the lost planktivore encounters shift the predators' diet
toward krill (WC krill 1.39 -> 1.67 of booked, still inside the band) and toward
juvenile benthivores and whiting, whose implied M2 rises slightly. No central
reading adds an OFF pair; the high-protection reading puts every fish ->
planktivore pair OFF.

## Proposal (not applied)

1. Formula, per fish-predator pair on fish prey, built inside the existing
   derivation:

       a'_pair = a_pair x sum_terms s_t [1 - sigma_t (1 - S_t)],
       S_t = min(1, (R_s / R_i,t)^2 (v_rel,s / (scan v_rel,i)) k / (N p_i)),
       dark_ratio' = dark_ratio x F_night / F_day, except stomach-derived ratios (kept).

   s_t = the term's share of today's a (prey species x period, `pair_terms.json`),
   R_i,t the term's rule-A radius, sigma_t the share of that prey in schools in that
   period, N, R_s, k per prey species (finding 3). The depth_access_table needs no
   change (renormalised to mean 1).
2. Pairs touched: WC, whiting, mackerel and benthivores -> planktivores, and WC ->
   mackerel. The four pairs on juvenile benthivores and whiting stay at F = 1.
3. Values: central F 0.57 / 0.83 / 0.71 / 0.69 / 0.24. Defensible span while the
   Norway pout question is open: from clupeids-only (0.77 / 0.88 / 0.71 / 0.91 on
   the planktivore pairs) down to central. The high-protection reading (0.11-0.40)
   needs pout schooling at sigma 0.9 with S < 0.01, which no source supports.
4. Floors: the four fish -> planktivore pair floors (0.25) and mackerel's floor
   lose their schooling meaning. Either pair floors 1 for fish predators on
   planktivores and mackerel (species floor kept for observation and mammals), or
   a documented residual refuge per group (finding 5). User decision.
5. Judges: budget_gate diet gate and prey-side M2 (no new OFF at central;
   planktivore implied / booked 0.69 -> 0.50-0.60); after training probe_diets
   (realised planktivore M2 vs booking; share of fish-prey intake taken in day
   ticks - field: almost none by day, E17, E18); tests/test_half_saturation.py
   pins change with any new a or dark_ratio.

## Gaps

- Norway pout: school size, packing and diel school cycle (E27 abstract only).
  Pout carries 0.52-0.54 of the WC and whiting planktivore pairs, so it decides
  between the clupeids-only and central readings.
- k and R_s are [J]: no gadoid capture success per attack on a clupeid school, no
  count of attacks per school encounter, no school detection distance, no
  gadoid-on-herring-school data with time of day.
- sigma for herring: the share of herring biomass in schools by day in the
  Skagerrak; the size of juvenile (0-1 group) herring schools in the box.
- Full texts not obtained (no open copy): Neill & Cullen 1974, Major 1978, Misund
  1993, Turner & Pitcher 1986, Fryxell 2007.
- No engine form for the twilight-window cap (the Rindorf plateau); a per-pair
  window cap would be a separate design.
- Krill swarms (Meganyctiphanes) follow the same algebra but were not treated.
- Mammal pairs (seals, porpoises on fish) not re-derived.

## Verdict

Schooling protection belongs in the encounter a, not in the hide action [D/J on
F/L: E1, E6-E13, E17, E23]. The theory is clean: detection, dilution and confusion
set the per-capita rate at which an individual is met and caught, satiation per
encounter bounds the yield, handling is unchanged; herring school size does not
depend on local density, so on the shelf (random search, no school following)
schooling is a constant factor on a per prey species and period. It must be
computed inside the per-item derivation against rule A's own radius, only in the
periods when the prey school (clupeids by day), keeping the stomach-derived
night/day ratios.

Its size is NOT well constrained. Theory spans S 0.005-1 (capped); the field puts
day-schooled clupeids near the protected end (Andersen 2017: day risk <= 0.04 of
night). On today's library the central pair factors are 0.57-0.83 on the
planktivore pairs and 0.24 on WC -> mackerel; Norway pout (clupeids-only
0.71-0.91) is the largest open lever.

The gate cannot settle it: today's dispersed a already reach only 0.69 of the
booked planktivore M2. A schooling factor moves the yardstick away from the
booking unless another factor in a (capture p, R) is currently too low.

The hide floors would lose their schooling meaning; whether they go to 1 for fish
predators or keep a documented residual is the user's choice. Juvenile benthivore
habitat refuges are the only residual with evidence.

## User decisions

- 2026-11-11: investigate Norway pout schooling (school size, packing, diel
  cycle) before any schooling factor is set; nothing changed yet.
- 2026-11-11: the hide floors (planktivore pair floors 0.25 vs fish, mackerel
  0.7) are decided only once the schooling factor is in a.
