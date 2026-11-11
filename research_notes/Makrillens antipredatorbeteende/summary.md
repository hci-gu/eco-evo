# Mackerel anti-predator behaviour - summary (2026-11-11)

Research only: no code, library, manifest or resume changed, nothing fitted to
model output. Per-source numbers: `evidence.md` (E1-E23). Computations:
`scripts/ecopath_ns1991_m2_mackerel.py`, `scripts/tuna_take_box.py` (stdout next
to each). PDFs: `papers/` (E1, E3, E8 via the Unpaywall helper). Grades: [F] field,
[L] lab/pen, [A] abstract or snippet only, [S] secondary, [D] derived here,
[J] judgement. "Analog" = another species.

## Question

What does "hiding" mean for mackerel (Scomber scombrus 0.94, Trachurus trachurus
0.06)? Is there field or lab evidence for a state that trades feeding for safety,
and how much protection does it buy?

## The model (as given by the caller, not re-derived)

Each 6 h tick a decision maker splits its biomass per 1 km cell over move (4), rest
and eat. Rest = hide: the rested share neither eats nor moves and pays 1.0 x
resting_metabolism (mackerel 4.59 MJ/t/tick, the obligate swimming floor; eat
1.02x, move 1.03x). Predators see B (1 - rest (1 - floor)); mackerel's
`visibility_floor` 0.7 is a judgement copied from midwater saithe, so full hiding
cuts predation by at most 30 %. Hidden biomass is also hidden from other groups'
km-scale observation next tick. Mackerel's only predator is the water column
(saithe 0.80, spurdog, hake, anglerfish, ling, pollack); booked M2 0.021, implied
~0.0004 /yr (the pair is OFF on layer overlap). Seals and porpoises unlinked,
seabirds muted. Mackerel are in the box mid-May to November, 0-50 m, ~30 cm.

## Findings

### 1. Anti-predator responses

- No field or lab observation of Atlantic mackerel or horse mackerel schools under
  attack by a real predator was found. The mackerel-specific evidence is:
  - Obligate, highly synchronised schooling. In net pens wild mackerel show
    collective responses to a predator model; crowding above ~30 kg/m3 suppresses
    them (E3) [L]. Denser is therefore not monotonically safer.
  - Escape from vessels: strong avoidance, schools escaped ~37 % of purse-seine sets
    in the northern North Sea, ~3.5 BL/s (E7) [A].
  - School sizes 1-7000 t; all schools < 100 m, 65 % at 0-40 m; no day/night
    difference in migration (E5) [A]. In the upper 40 m and >= 6 C "during both day
    and night" (E4) [F]. E4's line that larger northern schools were "a common
    response to predation pressure" and "traded off safety over maximizing feeding"
    has no predator data behind it ([J] in source).
  - Mackerel keep schooling down to ~1.8e-7 uE/m2/s (E2 via E1) [S] and feed to
    ~1e-7 (E1) [L]: they can school and feed through the night near the surface
    (E1), and in the field they do feed at night (copepods until midnight, krill
    and myctophids 22-08 h; no feeding below the mixed layer; E6) [A]. The dense-day
    / dispersed-night pattern of many clupeoids is NOT documented for mackerel.
- The attack-response repertoire (flash expansion, fountain, vacuole, split,
  compression, diving, herding) is documented in herring and menhaden: E8 [F,
  analog], E9-E11 [A, analog]. Responses last seconds to minutes and recur every
  15-25 min in a herring school (E10). Large herring schools (> 460 m2) were not
  attacked by killer whales; attacked schools were denser and rounder (E9).
- Horse mackerel: the day-on-bed / night-dispersed pattern is documented for
  T. capensis and T. trecae, not T. trachurus (E13) [A, analog]; it is tied to where
  the prey are, and the opposite holds in T. murphyi. Horse mackerel is 6 % of the
  group.

### 2. Does the response stop feeding?

- The one mackerel datum that couples school state and feeding: in a tank the
  non-feeding school occupied a ~110 deg arc at every light level, the school
  feeding in full light the whole tank (342 deg) (E1) [L]. So the compact school is
  the non-feeding school. But Macy found school dispersal uncorrelated with feeding
  rate across light levels, and mackerel feed without breaking school (E1). No
  measured cost (intake inside compact vs loose schools, time budgets, NND vs
  feeding) exists for mackerel or horse mackerel.
- The measured anti-predator manoeuvres (analogs) are short and collective; none
  describes a state held for hours in which feeding stops. Nothing found supports a
  6 h "stop eating to be safe" state in mackerel.

### 3. Protection bought

- No source gives a "share still exposed" for mackerel. The analogs give
  group-size effects, not a compact-vs-loose fraction: menhaden per-capita risk
  rises steeply only below ~2 m2 of school (E8) [F analog]; herring schools
  > 460 m2 were not attacked (E9) [A analog]; the oddity/confusion advantage in
  minnows is present at 8 and gone at 15 fish, and a school of 10 was safer than
  1-4 fish (E12) [A/S analog].
- Mackerel schools are tonnes to thousands of tonnes (E5), always far above those
  thresholds. The schooling protection is therefore always on, in every share of the
  cell's biomass, eating or not. The project convention puts schooling in the
  prey's hide action, not in a; the verdict says what this finding means for
  that.

### 4. Who eats ~30 cm mackerel, and how much

| Source | Area | Predation on mackerel | Grade |
|---|---|---|---|
| SMS key run (E15) | North Sea | M2 = 0 by construction (predator only); M = 0.15 assumed | [F, structural] |
| Ecopath 1991 (E14) | North Sea, all ages | M2 0.068 /yr: baleen whales 0.036, squid 0.010, seabirds 0.006, fish 0.013 (cod 0.004, hake 0.003, monkfish 0.0025, saithe 0.0018, spurdog 0.0015); the model's water-column members 0.0087; group with horse mackerel 0.070 | [D] |
| Bluefin tuna (E16, E17) | Skagerrak-Kattegat, Aug-Oct | mackerel 13 % N, ~23 % W of stomachs, mean 32 cm; 500-1500 tuna -> M2 ~0.002-0.010 /yr, 5-31 t per season in a 3600 km2 box (tuna numbers unknown; 15 000 tuna -> 0.05-0.10) | [F] diet, [D] take |
| Minke whale (E20) | North Sea | mackerel 9.3 % of diet by weight (n = 7) | [A] |
| Harbour porpoise (E21) | Dutch coast, Danish/Swedish waters | not a main stomach prey | [A] |
| Gannets (E19) | NW Atlantic | take mackerel in the top 10 m (mean dive 2.7 m, max 9.7) | [F] |
| NW Atlantic (E18) | Gulf of St Lawrence | 28-44 kt/yr in 2012-21, > 2x landings; gannets, grey seals, dogfish (4-49 % of diet), tuna, small cetaceans | [F report] |

- In the box the predators that matter elsewhere are absent or seasonal: grey
  seals are rare in the Skagerrak, gannets are muted, spurdog is depleted, minke
  whales and dolphins are not model groups, tuna are a late-summer visitor. The fish
  predators the model has (water column) are of the order of 0.01 /yr in the
  North Sea-wide Ecopath, the same order as the booked 0.021; the model's 0.0004
  comes from the pair's layer overlap, not from the booking.
- Verdict on Q4: predation on adult mackerel in the box is small, not zero. The
  bases differ and are not merged: the North Sea-wide 1991 Ecopath gives 0.013 /yr
  from all fish predators and 0.0087 from the model's water-column members (E14,
  [D]); the model's own booking on the water-column pair is 0.021 /yr (box diet
  flow, research_notes/"Rovfiskens uppdelning"); tuna add an estimated
  0.002-0.010 /yr at 500-1500 fish (E16-E17, [D], tuna numbers unknown). The
  largest single non-model term is tuna, and it grows with the tuna population
  [D/J].

### 5. Could "hide" be something other than "stop feeding and stay"?

- A depth shift is the one alternative with a physical basis: gannets take
  mackerel only in the top ~10 m (E19), tuna feed at 0-20 m (E22), so mackerel at
  20-40 m are out of plunge-diver reach. But (a) mackerel are held in the mixed
  layer by temperature (>= 6-8 C) and do not feed below it (E4, E6); (b) the
  model's only predator, the water column, lives DEEPER (saithe over 150-250 m
  beds), so a downward shift would raise, not cut, overlap with it; (c) the model
  has no vertical dimension inside the 0-50 m layer. A depth refuge would need a
  surface predator in the model to mean anything.
- The other natural reading is "compact, non-feeding school" (E1), but its
  protection is unmeasured, and schooling protection does not depend on whether
  the school eats (section 3).
- No source supports darkness or cover as a refuge: mackerel are visible and
  schooling around the clock, and the open 0-50 m layer has no shelter.

## Gaps

- No observation of mackerel or horse mackerel under attack by tuna, dolphins,
  porpoises, gannets or whales; no packing density of wild mackerel schools; no
  DST depth record of adult Atlantic mackerel; no measurement of intake in compact
  vs loose mackerel schools.
- Number of bluefin tuna in Skagerrak-Kattegat; tuna diet by month and area (E17
  full text not read; garfish dominates in Kattegat and the Sound).
- E14 is North Sea-wide 1991 and all ages; its whale and squid terms are not
  box terms.
- T. trachurus diel behaviour in the North Sea: no study found.

## Verdict

"Hide" for mackerel is a modelling device, not a documented biological state [J,
built on F/L]. Every measured anti-predator response of schooling pelagic fish is
collective and lasts seconds to minutes (E8-E11); mackerel are obligate schoolers
that school and feed day and night in the open mixed layer (E1, E2, E4, E6), so
there is no cover to retreat into and no evidence of a 6 h state that forgoes
feeding for safety. The closest real counterpart is a compact, non-feeding school
(E1, lab), but its extra protection over a feeding school is unmeasured; the
protection mackerel do get (dilution, confusion, collective escape) comes from the
school itself and is always on, eating or not.

Floor: no source moves the 0.7 up or down; it stays [J]. The only mechanism that
could make "rest" safer than "eat" is a tighter school, whose added benefit in
schools of tonnes is likely small (E8, E9: dilution saturates far below mackerel
school sizes) and which crowding can even impair (E3). As a direction that points
to a floor near 1 - but only once the convention below is settled; on its own it
means nothing. A depth refuge is the only alternative
semantics with field support (E19), and it protects against surface
predators the model does not have. In practice the choice matters little today:
the only pair on mackerel is OFF on layer overlap (implied M2 ~0.0004), so the
floor barely enters mackerel's realised mortality; it does still decide how much
of mackerel's biomass other groups observe after a rest tick.

The decision this sits on (a project convention, not a value). AGENTS.md section 5
(from section 220) says every fish pair's a comes from encounter theory with "prey
refuges and schooling left to the prey's hide action". Under that convention the
evidence above cuts both ways, and the user has to choose which switch to throw:
- Keep the convention: then the 0.7 floor IS mackerel's schooling protection. The
  evidence says that protection is always on and does not exclude feeding (E1:
  mackerel feed inside the school; E5, E8, E9: dilution saturates far below
  mackerel school sizes), so a learned, feeding-exclusive hide action is the wrong
  carrier for it: the field describes a constant protection, which a hide action
  delivers only to the share that stops eating. A floor near 1 under this
  convention would remove schooling protection from mackerel altogether.
- Change the convention: move schooling (dilution/confusion) into the encounter a
  of every fish-prey pair, not only mackerel's. Then mackerel's floor has nothing
  left to represent except a tighter non-feeding school or a depth refuge, and the
  evidence for either as a protective state is weak (sections 2, 5).
No floor value is recommended here; the finding is which of these two the floor
is supposed to stand for.

Grade of the verdict: [J] resting on [L] (E1, E3), [F/A] analogs (E8-E11) and
[F]/[D] predation numbers (E14-E19).

## User decisions

- 2026-11-11: keep mackerel's floor 0.7 for now; investigate whether schooling
  protection (dilution, confusion) belongs in the encounter a of the fish-prey
  pairs instead of the prey's hide action (research only, no change).
- 2026-11-11 (research_notes/"Gom och sinnen"): keep the hide-observation coupling.

## Addendum (2026-11-11): the floor after the combined derivation

User: investigate mackerel's floor while the recruit shape is derived. Analysis
only, nothing applied.

What the floor 0.7 acts on (code: `interactions.py:785-800`, `observations.py:39-78`;
mackerel has no pair floor, so the species floor applies everywhere):
- predation by its only predator, water column -> mackerel (stage_access 0.14,
  OFF): implied M2 0.0004 /yr today, 0.00007 /yr under the combined capture-p x
  schooling derivation (research_notes/"Fangst och stim i fiskparen", factor
  x0.244: p 0.20 -> 0.549, schooling F 0.089 at the new p). Full hiding would
  remove at most 0.3 of that, ~0.00002 /yr, against a booked 0.021 /yr;
- what six observers see of mackerel the next tick (benthivores, water column,
  whiting, planktivores, porpoises, seals): at fix97's rest share 0.028 the cut
  is 0.3 x 0.028 = 0.008 of the biomass.

What a floor below 1 would stand for, once the combined derivation is in a:
1. Schooling (dilution, confusion, collective escape): now in a through F
   (research_notes/"Stimskydd i motes-a"); mackerel is always schooled (sigma 1
   day and night), so the whole pair carries it. A floor < 1 counts it twice.
2. Individual vigilance / pre-strike escape: in the capture p as q 0.8
   (research_notes/"Fangstsannolikhet for fiskbyten"); for a schooled prey k
   replaces p, and collective escape is in k. A floor < 1 counts it again.
3. A real refuge that costs feeding: none found. Mackerel school and feed day
   and night in the open 0-50 m layer; no cover; darkness is in rule A and the
   night/day ratio; a depth refuge works only against surface predators
   (gannets, tuna) that the model does not have, and a downward shift moves
   mackerel toward the water column, not away (findings 1, 2, 5 above).
4. Observation: under the blend-of-senses reading (research_notes/"Gom och
   sinnen") a non-feeding mackerel school still smells, sounds and moves; the
   coupling itself is kept (user), but the 0.3 cut has no sense behind it.

Verdict [J on F/L, D]: with schooling and vigilance in a, nothing is left for
mackerel's floor to represent; the evidence supports 1.0 (no hide protection; rest
= a pure feeding pause at the swimming cost). The effect is negligible either way
(< 0.0001 /yr of M2, < 1 % of observed biomass), so the choice is about
consistency, not about the dynamics. A floor of 1 is consistent only together with
the schooling factor in a; with today's a (no F) the 0.7 is the only schooling
protection mackerel has.

User decision (2026-11-11): mackerel's visibility_floor 1.0 goes into the
combined section, together with the schooling factor in a, never before it.
