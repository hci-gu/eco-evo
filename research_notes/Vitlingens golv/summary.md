# Whiting's hide floor - summary (2026-11-11)

Research subagent; the main agent saves this text. Nothing changed in the library, code, tests,
manifest, AGENTS.md or resume; no training; nothing set from model output. Per-source numbers:
`evidence.md` (E1-E13, model facts M1-M9, derived D1-D3). Grades: [F] field, [L] lab, [A] abstract
or snippet only, [S] secondary, [D] derived here, [J] judgement. Seal and porpoise pairs on whiting
are out of scope (research_notes/"Sal och tumlare golv").

Files:
- `scripts/floor_mechanics.py` -> `scripts/floor_mechanics_stdout.txt`: library facts on the pairs
  and floors, the cut a floor makes, the jellyfish-shelter bound from the box GOV length table
  (research_notes/"Arsungarnas andel" tables/stage_length_biomass.csv), the stationary-prey
  encounter factor.
- `papers/`: Unpaywall records only (no open PDF for E2, E3, E6, E7, E12). E1 (Dahl 1961) and E4
  were read as VLIZ page scans, not stored.

```
python3 -I floor_mechanics.py <repo> > floor_mechanics_stdout.txt     # < 1 s
```

## Question

Whiting's species `visibility_floor` 0.5 ("whiting on the bed hide better than midwater schools",
judgement) acts on its two fish predators (water column, benthivores; neither pair carries a pair
floor) and on what the benthivores, the water column, mackerel and the planktivores observe of
whiting next tick. Once schooling (F) and capture p with pre-strike vigilance q 0.8 are in the
encounter a (research_notes/"Fangst och stim i fiskparen"), and depth, bed height and light are in a
through the band overlap and rule A, is there a real whiting refuge state left that (a) costs
feeding and (b) is not already in a? How much protection does it buy, what floor follows, and what
does the floor mean for the observation?

## Model (verified)

- Rest = hide: the rested share neither eats nor moves and pays 1.0 x resting_metabolism (3.87
  MJ/t/tick, SMR at the bottom layer) (M7). Predators see B (1 - rest (1 - floor)); the same floor
  sets each observer's view next tick; a pair floor is read only on a checked predation pair
  (`interactions.py:785-800`, `observations.py:39-78`).
- Pairs on whiting (M2): benthivores -> whiting a 0.0226, stage_access [0.68, 0.24], night/day 0.168;
  water column -> whiting a 0.0359, [0.41, 0.08], 0.137; both at the species floor 0.5. Seals 0.85
  and porpoises 0.75 have pair floors.
- One hidden share per cell and one floor per pair, applied before stage_access: J and A hide alike
  (M4).
- Already in a: whiting as prey 0.7 day / 0.5 night in the bottom 10 m (band, M5); light (rule A,
  night/day); schooling F 0.985-0.988 (sigma 0.1 / 0.05); capture p 0.2 (a lab per-strike value that
  already contains post-strike evasion) -> 0.59-0.62 with q 0.8 (M9). Every a assumes the prey
  cruising at 0.10 m/s (M6).
- Size of the effect today: at the fix97 annual-mean rest share 0.066 the floor 0.5 cuts predation
  and observation by 0.033 of B (D1).

## Findings

### 1. Candidate refuge states

a. Sheltering under jellyfish (Cyanea) - real, but small fish, pelagic, and not feeding-exclusive.
- Dahl 1961 (E1) [L/F, full text]: whiting 3-9 cm, July, Raunefjord. In tanks they "generally did not
  show any close association with Cyanea"; "now and then" a fish posted itself among the tentacles.
  When frightened by objects above they hide among the tentacles, sometimes instead dive away. Contact
  rarely harms them (few nematocysts vs 100-300 /mm2 on a goby). Fish caught among Cyanea near the
  surface had fed on copepods (7 of 7): association does not stop feeding. No predator in any trial:
  protection inferred, never measured.
- Lynam & Brierley 2007 (E2) [A]: whiting residual survival correlates with Cyanea abundance (R >=
  0.60, N 15, 1971-86, North Sea): stock-scale correlation, mechanism untested. Hay et al. 1990 (E3)
  [A]: the shelter test's outcome not retrievable. Eriksen et al. 2012 (E5) [F/A]: medusa-0-group
  spatial associations modest and area-dependent (Barents Sea).
- Who and when: pelagic 0-group in June-August near the surface, before settlement at ~8.5 cm
  (E7 [A], early June - early August; E8 [A], ~3 cm fish in DVM in June). Their predators in that
  phase include whiting itself (E9 [A], "substantial cannibalism"), which the model leaves out.
- In the box (D2) [D]: whiting <= 9 cm are 0.027 of the Q3 whiting biomass (2001-25; 0.043 in
  1991-96) and ~0 in other quarters. Caveat: the table is a q-corrected GOV bottom-trawl catch and
  does not see the pre-settlement pelagic 0-group near the surface; 0.027 is the share of the
  MODEL's whiting biomass (GOV-based, so it excludes those fish too) that could shelter, not the
  field share of sheltering fish. In tonnes the argument holds anyway: 3-9 cm whiting weigh grams. Even with every such fish sheltered and perfectly protected the
  cell-level floor would be 0.973 in Jun-Aug (annual equivalent 0.993); at half associated with
  residual exposure 0.5, 0.993. The engine's floor acts on the whole cell's J + A, so a jellyfish
  floor would be applied to adults it does not concern.
- Not in the band either way: these fish are near the surface (E1, E8), the model places whiting as
  prey in the bottom 10 m (M5). That is a profile question for the recruits, not a floor.

b. Structure, bed and habitat choice - no refuge on the box's open mud.
- Atkinson et al. 2004 (E6) [A, L]: whiting prefer sand > gravel > emergent structure; a shadow
  (predator stimulus) pushed SMALL whiting toward emergent structure and LARGE whiting toward open
  sand. Sizes and time shares not available; feeding not measured.
- Walker-Milne et al. 2025 (E10) [A/S]: juvenile whiting (20-120 mm) in the field in deeper water
  over smoother terrain, "just outside the edges of algal patches, not inside them" (cod in algae).
  So juvenile whiting do not live in structure; they forage at its edge.
- The box's whiting live over 100-300 m mud and sand (no algae, little emergent structure). Large
  whiting's predator response (toward open sand) is a habitat choice, and the habitat is what they
  have anyway.
- "On the bed" (M1's reason) is already the band profile (whiting as prey 0.7 / 0.5 in the bottom
  10 m) [D]. A height shift cannot be a refuge against both fish predators: cod sit at the bed (0.70
  of day time in the bottom 2 m), saithe above it (exp 15 m), so moving down raises overlap with
  cod and lowers it with saithe, and moving up does the reverse [D on M5]. One floor on both pairs
  cannot express that, and the overlap is in a already.

c. Reduced activity at night - not a whiting state.
- Whiting feed by day and by night depending on prey and position: pelagic whiting at night,
  demersal by day (Pedersen 2000, E11 [S]); sandeel taken mostly 22-24 h (E11 [A]); benthic prey at
  night, fish at dawn and dusk (Rindorf 2003, E11 [S]); no adult day/night catch difference (E12
  [S]); no diel difference in krill feeding (E13 [S]). Predator darkness is in rule A's night/day
  (0.137, 0.168). No daily inactive period was found that trades feeding for safety.

d. A motionless, resting whiting - the one residual not in a, small and derived.
- Every fish-pair a assumes the prey cruising at 0.10 m/s (M6); the engine's rest is a state at SMR
  (M7), i.e. without swimming. A motionless prey lowers the Gerritsen-Strickler relative speed (D3)
  [D]: saithe 0.984-0.993 by day, 0.956 at the night speed; cod 0.81-0.92 (0.4 BL/s, 30-50 cm;
  0.66 at 0.2 BL/s). Weighted by the day share of each pair's a (night/day 0.14-0.17), the water
  column's factor is ~0.98-0.99 and cod's ~0.85-0.92.
- Not quantified [J]: a motionless prey is harder to detect by vision and lateral line (lowers the
  factor further) but may be less vigilant (raises it). No whiting data on either. Nothing shows
  that wild whiting rest motionless; whiting carry a swim bladder and hover, so the reading "rest =
  motionless" is the engine's SMR cost, not a field observation [J].

### 2. Protection beyond F, q and the profiles

| candidate | stage | months | protection beyond a | grade |
|---|---|---|---|---|
| Cyanea shelter | pelagic 0-group 3-9 cm | Jun-Aug (pelagic phase ends at settlement ~8.5 cm) | cell floor >= 0.973 even if perfect; feeding continues | [L/F] E1, [A] E2, [D] D2 |
| structure / habitat | small juveniles (tank) | - | none on the box's mud; field juveniles avoid structure | [A] E6, E10 |
| height above bed | all | - | in the band; opposite for cod and saithe | [D] M5 |
| night inactivity | all | - | none: whiting feed at night; darkness in rule A | [S/A] E11-E13 |
| motionless at rest | all | - | v_rel factor ~0.98-0.99 (WC), ~0.85-0.92 (cod) | [D] D3 + [J] |

With one floor per pair applied to J and A alike, any stage-specific refuge (the Cyanea shelter is
0-group only) would be paid out to adults too; the motionless-prey factor is stage-free (it depends
on the predator's speed vs the prey's), so it is the only candidate the engine's form fits.

### 3. What the floor means for observation (blend of senses)

A whiting resting on or near the mud still smells, and at km scale smell and sound carry the blend;
the lateral line and sight do not reach that far (research_notes/"Gom och sinnen"). A motionless
whiting gives a weaker hydrodynamic signal at cm-m, nothing a km-scale observer would miss [J]. So
for the observers that do not eat whiting (mackerel; planktivores, for whom whiting is a predator)
the floor has no sense behind it below 1, and for planktivores it leaks whiting's last activity state
(the predator-role case of "Gom och sinnen"). With the coupling kept (user), a pair floor below 1 on
a fish predator also lowers what it observes: under the alternative 0.9 on cod, 0.007 of B at rest
0.066.

### 4. Mechanical consequences of the proposal

- Today's cut 0.033 of B at the annual-mean rest share (0.5 of B when a cell fully rests) goes to 0
  at floor 1 (0.0066 on the cod pair under the alternative 0.9; 0.1 at full rest) (D1). The fish pairs'
  realised predation on whiting rises at most ~3 % in the annual mean, more in months when whiting
  hide more; the gate (budget_gate) has no floor term, so its numbers do not move.
- Mackerel, planktivores, the water column and the benthivores observe whiting's whole biomass (the
  benthivores at 0.9 of the rested share under the alternative).
- Rest becomes mostly a feeding pause at SMR (1.0 x RM against eating's feeding_cost 1.50 x RM: it
  saves 0.5 x 3.87 = 1.9 MJ/t/tick) with a small cod refuge; the trained rest share may fall. Action and observation shapes are
  unchanged; payoffs change, so retrain with the combined section.
- Seals and porpoises keep their pair floors (out of scope); a species floor of 1 does not reach them.

## Proposed floors (not applied)

| floor | today | central | alternative / range | grade | basis |
|---|---|---|---|---|---|
| species `whiting.visibility_floor` (mackerel, planktivores; every pair without a floor) | 0.5 | 1.0 | 1.0 | [J on F/L/A] | no feeding-exclusive refuge found; the km-scale blend of senses is not cut by a rest |
| water column -> whiting (pair) | 0.5 (species) | 1.0 (no pair floor; falls to species) | 0.98-1.0 | [D] D3 | saithe search fast; a motionless prey changes v_rel by 1-2 %; no refuge vs saithe found |
| benthivores -> whiting (pair) | 0.5 (species) | 1.0 (no pair floor) | 0.9 (0.8-1.0) if "rest = motionless" is adopted | [D] D3 + [J] | cod search slowly (0.4 BL/s): a motionless whiting is met ~0.85-0.92 as often; detection and vigilance terms unquantified |

The jellyfish shelter would at most justify 0.973 on the fish pairs in Jun-Aug (annual equivalent
0.993; D2), applied by the engine to adults too; it is a bound, not a proposal.

The 0.9 alternative is a CONVENTION question, not a whiting number: "rest at SMR = motionless prey,
met at v_rel(v, 0)" would apply equally to a resting herring vs cod, a resting benthivore vs saithe,
and so on, and no other floor in the library (or the parallel floor notes) uses it today. If the
user adopts it, it should be derived for every fish pair at once; until then 1.0.

Order: the same rule as mackerel's 1.0 - into the combined section together with F and q in a, not
before. For whiting the dependence is weak: F is ~0.985 (whiting hardly school), and today's p 0.2 is
a per-strike lab value that already includes post-strike evasion; only q 0.8 (pre-strike escape) is
new. The 0.5 is not schooling protection, so the floor does not carry what F would take over.

## Gaps

- Hay et al. 1990 (E3) full text: the shelter test's result, share of medusae with fish, fish per
  medusa, depth. No measurement of predation on Cyanea-associated vs free 0-group whiting anywhere.
- Cyanea abundance in the Skagerrak box offshore (Gullmar data are coastal and depend on North Sea
  inflow; Grondahl 1988 not read).
- Atkinson et al. 2004 (E6): fish sizes, time shares, whether feeding stopped under the stimulus.
- No whiting behaviour toward cod or saithe (tank or field); no record of whiting resting motionless
  on the bed; no data on detection of motionless vs moving prey by gadoids.
- The pelagic 0-group's position (near the surface in Jun-Aug, E1, E8) vs the model's bottom-10-m
  profile for all whiting as prey: a profile question for the recruits (research_notes/"Rekryternas
  bottendjup" may cover it).

## Verdict

Whiting have one documented shelter behaviour, under Cyanea medusae, and it belongs to pelagic
0-group fish of 3-9 cm in June-August that keep feeding on copepods while associated (E1 [L/F]); its
protective value is inferred from tank behaviour and a stock-scale correlation (E2 [A]), and in the
box it concerns ~3 % of the whiting biomass for three months (D2), so it cannot carry a floor of 0.5
on the whole group. Structure is not a refuge for whiting (field juveniles stay outside algal
patches, E10; the box is open mud), height above the bed is in the band and cuts opposite ways for
cod and saithe, darkness is in rule A, and whiting feed at night. The 0.5 judgement ("on the bed")
is therefore already in a [D/J]. The one residual not in a is that a resting fish at SMR is not
swimming, which lowers the encounter rate with slow-searching cod by ~10 % and with saithe by ~1 %
(D3, [D] on the a's own assumptions, [J] that rest means motionless).

Proposal [J on F/L/A, D]: species floor 1.0 and no pair floors on the two fish predators (rest =
a feeding pause at SMR). Derived alternative for the user, as a library-wide convention: a motionless
resting prey met at v_rel(v, 0), which gives benthivores -> whiting 0.9 (0.8-1.0) and the water
column ~1.0. Effect is small (cut 0.033 -> 0 of B at today's rest share, 0.007 under the
alternative), the observation cut ends for every observer of whiting except seals and porpoises
(their pair floors), and the change goes in with the combined section.

## User decisions

- 2026-11-11: species floor 1.0 (no fish pair floors) in the combined section, with F
  and the new p; the "resting prey is motionless" convention is not adopted.
