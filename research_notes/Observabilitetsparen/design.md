# Design: neighbour weight (reading B) and the place of swimming speed (2026-11-13)

Design note for review: nothing in code or library changed, no training. It builds on `summary.md`,
`fish.md` (sections 2 and 7 for movement), `mammals.md` and `code.md`. Part 3 (mammal attack rates from
encounter theory) is a separate investigation in research_notes/"Daggdjurens attackrater" (a-umlaut).

## 1. Neighbour weight w per observation pair

### 1.1 Rule

For each observed group k of decision maker i and each in-grid neighbour cell:

    obs_neighbour = c + w_ik (n - c)

with n the target's (visibility-scaled) density in the neighbour and c the same channel in the observer's
own cell. w = 1 is today's observation; w = 0 says "no information beyond my own cell". Blending towards c
instead of multiplying by w keeps an uninformed observer from reading "empty there". Outside the grid the
channel stays 0 as today (`shift_field`; GPU `obs_valid`). The juvenile channel of a staged target takes the
target's w. The own biomass in the neighbours and the observable impacts keep w = 1 (open point 1.6).

Reading B: a neighbour channel is what the group met there while ranging during the tick. The cell holds
many individuals; w is the share of them whose track met the target in that neighbour, so the blend is the
group's averaged knowledge.

**The blend does not remove information for any w > 0.** The policy also sees c, so n = (obs - (1 - w) c) / w
is a linear read-out (first-layer weights of order 1/w); obs-norm keeps the map linear. A soft blend is a
learnability prior - the signal is smaller against the noise ARS sees - not an information limit. Any
deterministic invertible map (scaling, saturation) has the same property. The forms that limit information:
- **binary**: w in {0, 1} per pair (code.md's per-pair neighbour switch), cut at a threshold on the derived w;
- **stochastic**: per tick and cell, the neighbour slot shows n with probability w and c otherwise (or a
  Poisson count); noise in every observation, and CPU/GPU parity only with identically seeded draws.
The derived w (1.3) is the ranking for either.

### 1.2 Derivation

    w = P(track enters a given neighbour) x E[1 - exp(-n) | entered]

- P from a correlated random walk with each member's measured path L and net displacement D per 6 h
  (`scripts/neighbour_weight.py`; start uniform in the cell).
- n = expected encounters with the target while in that neighbour:
  - the observer searching (prey, or no link): n = (B / m_unit) 2 R_obs x path inside;
  - the target preying on the observer: the predator approaches what it detects, so its swath counts,
    n = (B / m_unit) 2 R_pred L_pred x (path inside / L_obs);
  - a diffuse target (plankton and krill layers, bed fauna): n -> infinity, w = P.
- B the box-mean density (manifest inference starts / 3600 km2), m_unit one detectable unit (an animal, a
  pelagic school by day, single planktivores at night, half the time each), R detection radii.

Inputs (central; the low and high cases vary the [J] ones):

| quantity | central (range) | grade, source |
|---|---|---|
| herring L / D per 6 h | 17 km / 4 km (2-11) | [F] fish.md 2 |
| pout, sprat, blue whiting L / D | 3 / 0.5 km (0.2-1) | [F] Onsrud 2005 speeds, D [J] (7.1, 7.3) |
| mackerel L / D | 29 / 8 km (5-15) | L [F] Diaz 2013, D [J] |
| saithe L / D | 20 / 3 km (0.5-8) | L [F], D [J] (coastal saithe site-attached, 7.1) |
| spurdog; hake, ling, anglerfish | 12 / 6 km; 1 / 0.05 km | [F/J] 7.1 |
| whiting L / D | 3 (2.2-6.5) / 0.7 km (0.3-2) | [F] Onsrud speeds, D [J] |
| cod; haddock; flatfish | 2 / 0.3; 3 / 0.7; 0.5 / 0.1 km | [F] cod home range; haddock as whiting [J] |
| porpoise L / D | 14 / 5.5 km (4-7) | [F/D] mammals.md 1 |
| seal | broad ARS 0.5 (20 / 5.9), transit 0.2 (20 / 17), focussed 0.3 (5 / 0.1) | [F/D] mammals.md 1 |
| fish detection radius | 10 m (5-30) | [J] vision / lateral line in a dark box |
| porpoise sonar | schools 200 m (140-260), gadoids 54 m, saithe 80 m | [D] mammals.md 2 |
| seal vibrissae | 25 m (15-35) | [D] mammals.md 2 |
| pelagic school mass | 5 t (0.5-50) | [J] - the most uncertain input |
| unit masses | cod/haddock 0.5 kg, whiting 0.3 kg, water column 2 kg, porpoise 45 kg, seal 80 kg | [J] |

### 1.3 Result (tables/neighbour_weight_{central,low,high}.txt)

| observer -> target | w central (range) | today |
|---|---|---|
| planktivores -> zooplankton, krill | 0.38 (0.31-0.38) | 1 |
| planktivores -> benthivores, whiting, water column | 0.36 (0.30-0.36) | 1 |
| planktivores -> mackerel | 0.005 (0-0.10) | 1 |
| **planktivores -> porpoises** | **0.10 (0.09-0.11)** | 1 |
| planktivores -> seals | 0.002 | 1 |
| mackerel -> zooplankton, krill | 0.42 (0.28-0.54) | 1 |
| mackerel -> planktivores | 0.22 (0.18-0.27) | 1 |
| mackerel -> water column | 0.38 (0.26-0.44) | 1 |
| water column -> layers, benthivores, whiting | 0.48 (0.31-0.48) | 1 |
| water column -> planktivores | 0.25 (0.20-0.25) | 1 |
| water column -> mackerel | 0.003 (0-0.03) | 1 |
| water column -> porpoises | 0.05 (0.02-0.08) | 1 |
| whiting -> layers, epifauna, benthivores, water column | 0.31 (0.16-0.44) | 1 |
| whiting -> planktivores | 0.16 (0.08-0.32) | 1 |
| whiting -> porpoises | 0.07 (0.04-0.07) | 1 |
| benthivores -> bed fauna, krill, whiting, water column | 0.19 (0.10-0.22) | 1 |
| benthivores -> planktivores | 0.10 (0.05-0.14) | 1 |
| benthivores -> porpoises | 0.04 (0.02-0.04) | 1 |
| every fish -> seals | <= 0.004 | 1 |
| porpoises -> planktivores | 0.24 (0.23-0.31) | 1 |
| porpoises -> benthivores, whiting, water column | 0.37 (0.33-0.45) | 1 |
| seals -> planktivores | 0.16 (0.15-0.22) | 1 |
| seals -> benthivores, whiting, water column | 0.30 | 1 |
| zooplankton -> everything | 0 (movement_speed 0; or leave, inert) | 1 |
| no-link pairs (summary.md 4) | as the observer's searching value for the target class | 1 |

Reading the table:
- No w reaches 0.55: a 6 h track enters at most about half of the four neighbours, whatever the
  displacement (a straight long track leaves the 3 x 3 in one direction; a tortuous short one stays home).
  The "low" and "high" columns are therefore not monotone in D.
- The residents are not 0: a cod near its cell's edge crosses it. 0.1-0.2 for the benthivores.
- Rare point targets are near 0: one seal per ~20 km2 and one porpoise per ~2 km2 are rarely met.
  planktivores -> porpoises is 0.10 because porpoise sonar finds schools at ~200 m over a 14 km path;
  the herring learn of a porpoise by being found by it. The script uses the school radius by night too;
  at night the porpoise finds single fish at ~40 m, so 0.06-0.10 is the fairer range.
- Pooling: w = P(enter) treats each unit's track as its own knowledge. If the N units of a cell pool what
  they met (herring schools split and join every ~35 min), the group knows 1 - (1 - P)^N: with 5 t schools
  N ~ 1 per cell and nothing changes; with 0.5 t schools N ~ 10 and w -> ~1 for diffuse targets.
- Today every entry is 1.

### 1.4 Code (both engines; widths unchanged)

1. Library: a pair key `neighbour_weight` on `<obs>_observes_<target>` (default 1, in [0, 1]); the loader
   builds `env.obs_neighbour_w[i]` aligned with `obs_others_idx[i]` and the juvenile channels.
2. CPU `observations.py:112-118`: `obs[i, base + 1 + k] = inside_d * (c_k + w_ik (shift_field(n_k) - c_k))`
   with `inside_d` an explicit in-grid mask per direction (`shift_field(ones, d)`): today the zero fill of
   `shift_field` is the only edge mask, and without the mask the edge would read (1 - w) c.
3. GPU `ecosystem.py` `obs_indices` / `observe` (:603-630, :830): a second index tensor pointing each
   neighbour slot at the same channel in the own cell, and a weight tensor [D, C, F]:
   `bank[idx] * valid * W + bank[centre_idx] * valid * (1 - W)` - one extra gather.
4. FG editor: the Observability matrix cell gets the number (checkbox = on/off as today, the weight as a
   field), and the round trip keeps it (the test_fgconfig_preserves_library pattern).
5. Tests: w = 1 bit-identical to today (CPU and GPU); w = 0 gives c; edges stay 0; CPU/GPU parity.
6. Input widths are unchanged, but the channels mean something new: every checkpoint is invalid, retrain.

### 1.5 Approximations

- w is fixed at box-mean densities. The encounter count really scales with the local density (a dense
  neighbour is noticed more). A per-cell form `w = P (1 - exp(-kappa n))` is possible but makes "nothing
  met" read as "like here"; kept out.
- The blend is deterministic. Reading B with point targets is really a noisy count; a Poisson draw would
  carry that but adds noise to every observation and breaks CPU/GPU float parity.
- The same-cell channel stays w = 1 (open point 1.6).

### 1.6 Open points for the user

0. The form: binary cut (threshold on w, e.g. 0.10 puts planktivores -> porpoises off as in the blind test),
   soft blend (a prior only), or stochastic mask (1.1).
1. The own biomass in the neighbours (conspecifics): w = 1 (schools share information; Pednekar 2024's
   chained senses), or the observer's diffuse w?
2. Observable impacts (fishing effort) in the neighbours: w = 1 assumed (vessels are loud at km) - or not.
3. The same-cell channel of rare point targets: by the same arithmetic a herring meets a seal in its own
   cell with n ~ 0.01. The predation itself acts in the cell, so the prey meets its consequences there;
   whether the same-cell channel should be weighted too (and towards what - the box mean?) is open.

## 2. Swimming speed: capacity stays, the time cost of a move becomes speed-dependent

### 2.1 movement_speed stays 1 for every mobile group

`movement_speed` is a capacity: the share of the moving biomass that crosses into the neighbour in one tick
(`movement.py:440-447`; GPU `velocity`). Every member's sustained capacity exceeds 1 km per 6 h
(0.05 m/s): cod directed bouts 0.2-0.6 m/s, plaice migrating 13-19 km/day, whiting and pout 0.10-0.16 m/s
measured even when foraging (fish.md 2, 7.1, 7.3); porpoise 0.65 m/s, seal 1.1-1.6 m/s. Setting it below 1
from the typical net displacement would hard-code residency, a behaviour the policy must learn. The engine's
one cell per tick caps the fast groups alike (porpoise net 4-7 km, herring 2-11 km per 6 h): at that scale
neither outruns the other in reality either. Kept at 1.

### 2.2 Where speed belongs: the time a move takes

> **Built in section 276** (both engines, `lib/world/relocation.py`, library key `relocation_speed`). Decisions
> taken there: the moving share forages in the cell it leaves; its prey choice is the policy's eat preference
> read before a sub-unit cell collapses to one action; the cell's growth stays a cell-level law and the moving
> share takes its foraging share of the intake net of new tissue and of the new tonnes. `movement_cost` is kept
> and paid for the tau share only; re-reading it from the swimming cost at U is open. Why it was built: fix103's
> porpoises could not afford to follow the moving planktivores (resume 276).

Today a DM's action shares are biomass shares, and the moving share eats nothing in the tick: it pays
`movement_cost` x RM and gets no food (`movement.py:289-306`; gains go to the eating share only). A move of
1 km takes a herring ~25 min, a cod ~50 min, a pout ~2 h: the rest of the tick a real fish forages. So the
model charges a whole tick of food for what costs a fraction tau of it, the same fraction for every group.
This is the "eats little while moving" of the porpoise probes (section 272.4 notes).

Proposal: a DM key `relocation_speed` U (m/s), tau = min(1, 1 km / (U x tick length)), and

- the moving share forages for (1 - tau) of the tick with the eat actions' prey choice p_j: the predator's
  foraging time becomes T + (1 - tau) M (M the move share) in the shared-handling response; with no eat
  action it does not eat (the policy chooses);
- its cost is tau x movement_cost + (1 - tau) x feeding_cost (x RM), and its share of the gains follows it to
  the destination;
- tau = 1 is today's rule, bit-identical.

tau per group (U = sustained cruising / directed speed; member-weighted):

| group | U m/s (members) | tau | grade |
|---|---|---|---|
| planktivores | herring 0.6 (0.5-0.9), pout / sprat / blue whiting 0.13 (0.10-0.16) | 0.55 x 0.077 + 0.45 x 0.36 = 0.20 | [F] fish.md 2, Onsrud 2005 |
| mackerel | 1.33 | 0.035 | [F] Diaz 2013 |
| water column | saithe 0.9 (0.72-1.05), spurdog 0.5, hake / ling / anglerfish 0.2 | 0.8 x 0.051 + 0.1 x 0.093 + 0.1 x 0.23 = 0.073 | [F/J] |
| whiting | 0.2 (0.13 juveniles measured - 0.3 adults) | 0.23 (0.15-0.36) | [F/J] |
| benthivores | cod 0.35, haddock 0.2, flatfish 0.2 | 0.31 x 0.13 + 0.39 x 0.23 + 0.30 x 0.23 = 0.20 | [F/J] |
| porpoises | 0.65 | 0.071 | [F] van Beest 2018 |
| seals | 1.4 | 0.033 | [F] |

So a move costs 3-7 % of a tick's foraging for mackerel, saithe, porpoises and seals, and 20-23 % for the
slow groups. That is where the members' speeds differ in the grid. It makes relocation cheap for every group;
it does not give fast prey a better escape. The escape at the encounter scale (burst speed, schooling,
endurance in a chase) stays in the pair's a and capture p (part 3). Today's `movement_cost` (1.0-2.0 x RM)
was set as a whole tick of activity; with tau its meaning for the tau share has to be re-read from swimming
cost at U, not rescaled.

Code (both engines): predation's foraging time T and the shared-handling response
(`interactions.attack_choice`, `multi_prey_rate`; GPU), `movement.apply_energy_costs` (cost and the routing of
gains to the moving share), the loader (tau from U and `--tick-length`, a "flux"-type conversion in
`tick_time.py`), the FG editor field, tests (tau = 1 bit-identical; CPU/GPU parity). The action layout is
unchanged; the move actions mean something new: retrain.

### 2.3 Not recommended

- movement_speed < 1 from typical displacement (hard-codes residency; 2.1).
- Multi-cell moves or 1 h ticks to separate the fast groups: they are capped alike, and at 1 km per 6 h the
  data give no asymmetry between porpoises and herring; 1 h ticks cost ~6x.

## 3. Order of work

1. The user decides 1.6 (form, own biomass, impacts, pooling).
2. Build part 1 first, with part 3's mammal a if the user takes them: part 1 is the measured lever (one pair
   moved the porpoises 0.34 -> 0.60 at year 1).
3. Gate on the long-run probe of that retrain: is the porpoise start fall gone, does each group's realised
   diet stay on its booking.
4. tau (2.2) only if the porpoises or another group still pay visibly for moving; built in the same retrain
   as part 1 it could not be told apart from it.

## 4. Decisions (user, 2026-11-13)

1. Part 1 form: **soft blend** `c + w (n - c)` (accepted knowing it is a learnability prior, 1.1).
2. Own biomass in the neighbours: **follows the observer's diffuse w** (its P(enter); porpoises 0.39,
   seals 0.30 from the CRW, fish as the "diffuse" rows of 1.3). No observable impact exists in the library
   (`observable_impact_vars` empty), so 1.6 point 2 is moot.
3. Part 3, porpoises: **only porpoises -> planktivores a 0.0225 -> 1.12** in the retrain; the bed pairs,
   their depth tables and the porpoise diet booking are investigated separately.
4. Part 3, seals: **unchanged**; first find out whether the Skagerrak seal "gadids" include Norway pout.
5. Part 2 (tau): after the retrain, only if the porpoises still move heavily and starve at the start.

Frozen fix100 probes behind the order (research_notes/"Daggdjurens attackrater" section 10): porpoises after
one year 0.34 (base), 0.38 (a 1.12), 0.60 (planktivores blind to porpoises), 0.74 (both).

## 5. Own cell and neighbours under one rule (2026-11-16)

Design note for review. Nothing in code or library changed. It answers open point 1.6.3 (the same-cell channel)
and the user's request that what a group may know about other groups, in its own cell and in the von Neumann
neighbours, follows ONE method for every group: from movement (swimming speed), senses and whatever else
bears on it. Script: `scripts/own_cell_weight.py` (imports `neighbour_weight.py`); tables:
`tables/own_cell_weight_{central,low,high}.txt`.

### 5.1 What fix102 showed

fix102 is the first run after section 274 (30 x 30 grid, warm-up 60 d with re-anchoring). The test froze its
policies at update 524 and ran the long-run probe world on the CPU engine. Porpoise biomass is given at the
year ends, two seeds:

| arm (all other policies current) | porpoises yr 1 / 2 / 3 |
|---|---|
| all current | 0.34 / 0.15 / 0.09; 0.27 / 0.16 / 0.07 |
| all untrained (PolicyBank seed 0; reproduces probe 1 exactly) | 1.03 / 1.06 / 1.07 |
| planktivores untrained | yr 3 0.50; 0.62 |
| every fish group untrained | 0.88 / 0.86 / 0.91 |
| seals untrained | 0.31 / 0.20 / 0.13 |
| porpoises untrained | 0 by year 2 |
| planktivores blind to porpoises (own cell and neighbours zeroed) | 1.02 / 1.09 / 1.12; 0.66 / 0.67 / 0.70 |
| planktivores blind in the own-cell slot only | yr 3 0.21; 0.18 |
| planktivores blind in the four neighbour slots only | yr 3 0.41; 0.30 |
| neighbour slots set to the own-cell value (presence kept, gradient removed) | yr 3 0.06; 0.06 |

What this shows:
- **Food is not the limit.** With untrained fish, the porpoises hold the SCANS level in a closed box, even with
  fewer planktivores (0.67). Their fill stays at 0.7-0.8 in every arm.
- **The decline is learned flight by the planktivores.** They leave cells where they see porpoises. Being blind
  to porpoises costs the planktivores only about 3 % of their biomass.
- **Presence in the own cell is enough.** The gradient towards a neighbour adds nothing.
- **The neighbour weight 0.10 did not limit the information.** Each neighbour slot read 0.9 c + 0.1 n, so it
  repeated the own-cell signal four more times.
- **The porpoises hardly move.** Their executed move share is 1-6 % over the whole training, the same in fix101.
  Their reward falls as the planktivores' move share rises (fix101: 0.24 -> 0.37 near update 900).

### 5.2 A deterministic weight does not limit information

- **Own cell.** A weight towards a fixed value, m + w (c - m), is affine in c. The per-slot observation
  normalisation (ARS-V2) removes any affine map exactly, so the policy sees what it saw before.
- **Neighbours.** The blend c + w (n - c) can be inverted, given c, for any w > 0 (1.1). The fix102 test shows
  the practical side: with a small w, the slots carry the own-cell value instead.

So the forms that limit information are binary (a pair off) or stochastic. The stochastic form below is the one
that follows from reading B.

### 5.3 The rule (one rule for every pair, own cell and neighbours)

For each tick, cell and observation pair, the group met the target with probability

    p = 1 - exp(-n),   n = expected encounters at the LOCAL visible density (not the box mean).

| region | met | not met |
|---|---|---|
| own cell | the target's density c | 0 (searched there, found nothing) |
| neighbour | the neighbour's density n | the own-cell reading (never there) |

- In a neighbour, p includes P(enter) from the CRW (1.2).
- A diffuse target (plankton and krill layers, bed fauna) has p ~ 1 and is read exactly. Rare point targets
  read as noisy.
- The local density replaces the box-mean approximation of 1.5: a dense cell is noticed, an empty one reads 0.
- n is the same encounter count the pair's attack rate is built from (searcher swath; the predator's swath where
  the target hunts the observer), so information and predation share their inputs.

Costs:
- **CPU/GPU parity** needs the same draws on both engines: a counter-based hash of (seed, tick, world, cell,
  pair) evaluated identically in numpy and torch, not either library's generator.
- **Training noise** rises: every observation of a rare target is noisy. ARS sees that as variance.
- **Checkpoints:** the channels change meaning, so every checkpoint is invalid.

### 5.4 What n depends on, and what is in the derivation

| factor | status |
|---|---|
| path L and net displacement D per 6 h (speed, tortuosity) | in (CRW, 1.2) |
| detection radius per sense and target: fish vision and lateral line ~10 m; porpoise sonar 200 m on schools, 54 m on gadoids, 80 m on saithe; seal vibrissae 25 m | in (1.2 table) |
| hearing: porpoise clicks are inaudible to every model fish; seal roars 50-140 m against ship noise | in (gives no range; summary.md 2) |
| the predator's swath: the prey learns of a predator by being found | in |
| detectable unit: pelagic schools by day, single fish by night; animal masses | in |
| hide floor: the observed density is the visible one | in (engine) |
| light: vision radius with light at depth | engine (rule A, section 269), not linked; the same R(E) should serve capture and information |
| vertical overlap: band profiles f_m(z) | engine (section 269), not linked; an encounter needs a shared depth band |
| pooling between units (chained senses, Pednekar 2024) | out; ~1 school per cell at box densities, so small |
| memory across ticks (porpoises return after ~2 d; Iorio-Merlo 2022 for seals) | out; the policy is feed-forward |
| directional odour plumes downstream of bed prey | out; neighbours are isotropic |
| consequences: own losses as a cue | out; losses are not an input |
| the target's own spatial layout | see 5.6 |

### 5.5 The own cell in numbers (central case)

P(met in the own cell during 6 h). There are two frames:
- **"track":** the real path leaves the cell. A herring spends 1.4 of 17 km there, a porpoise 0.6 of 14 km.
- **"confined":** the observer searches its own cell all tick. The engine keeps a cell's biomass in place and
  computes a whole tick's encounters there.

"unit" is one detectable unit per km2: a lone animal, school or fish (for fish targets a very low density).

| observer -> target | track, box mean | track, one unit | confined, box mean | confined, one unit |
|---|---|---|---|---|
| planktivores -> porpoises | 0.42 (0.82 at 6 per km2) | 0.53 | 0.95 (0.87-0.98) | 1.00 |
| planktivores -> seals | 0.01 | 0.19 | 0.04 (0.02-0.06) | 0.54 |
| planktivores -> mackerel | 0.03 | 0.15 | 0.09 | 0.44 |
| planktivores -> benthivores, whiting, water column | 0.93-0.98 | 0.03-0.10 | 1.00 | 0.19-0.29 |
| water column -> porpoises | 0.15 | 0.24 | 0.69 | 0.89 |
| whiting, benthivores -> porpoises | 0.31-0.40 | 0.48-0.60 | 0.54 | 0.78 |
| every fish -> seals | 0.01-0.03 | 0.11-0.40 | 0.04 | 0.54 |
| water column -> mackerel | 0.006 | 0.03 | 0.06 | 0.29 |
| mackerel -> planktivores | 0.47 | 0.47 | 0.71 | 0.72 |
| porpoises -> planktivores | 0.58 | 0.58 | 1.00 | 1.00 |
| seals -> planktivores | 0.52 | 0.52 | 0.75 | 0.75 |
| mammals -> demersal fish groups | 0.93-0.96 | 0.06-0.09 | 1.00 | 0.51-0.89 |

Recommendation:
- **Own cell: the confined frame.** Information then follows exactly the encounters the engine computes for
  predation in that cell. It is the same tick of search, and the prey meets the predation there.
- **Neighbours: the CRW as in 1.2.** The engine computes no encounters there, and the cell is a coarse grain of
  the real ranging.

The two frames count some path twice (a whole tick in the own cell plus the neighbour visits). That is accepted
as the price of matching the engine's predation.

### 5.6 What the rule does to the porpoise question

- **The rule mostly limits rare targets.** That means fish -> seals (0.04), -> mackerel (0.06-0.09) and rare
  targets in the neighbours.
- **Planktivores -> porpoises stays informative** at 0.95 at the box mean in the confined frame. In the trained
  runs the porpoises sit 4-6 per km2 in 75-100 cells, so a herring in a porpoise cell is found with p ~ 1.
- **Expected effect on the fix102 decline: about none.** This is computed, not run. A frozen run with the
  stochastic mask would measure it.
- **The model's porpoise layout is the anomaly.** Real porpoises are solitary or in groups of 1-3 at ~0.5 per
  km2 and range 14 km per 6 h. In the model they clump and stay put. This is open, not a finding: how much the
  clumping comes from the spawn and how much from the learned staying.

### 5.7 Speed has two places

Speed enters in two places:
1. **Information:** the path L above.
2. **The time a move takes:** tau (2.2). Today every group pays a whole tick of foraging for a 1 km move.
   - It takes a porpoise ~7 % of the tick (tau 0.071) and a slow fish ~20 % (planktivores 0.20, whiting 0.23,
     benthivores 0.20).
   - tau comes from each group's measured U, so it is the same rule for all.

fix102 inverts the trigger in section 4 item 5. The porpoises do not "move heavily and starve"; they hardly move.
Prey flight works because the predator never follows. tau is the speed lever that could change that. It cannot
be tested on frozen policies, only by a retrain.

### 5.8 Open points for the user

1. **Frame for the own cell:** confined (recommended) or track.
2. **Form:** the stochastic rule of 5.3 for every pair, or a binary cut by a threshold on p.
3. **tau (2.2):** in the same retrain as 5.3, or separately. If trained together, their effects cannot be told
   apart; two runs would separate them.
4. **Light and vertical overlap in n (5.4):** link them now, or later.
5. **The porpoise layout (5.6):** investigate the spawn and the staying, separately from this note.

### 5.9 Decisions and the build (user, 2026-11-16; resume section 275)

Decisions:
1. **Own cell:** the confined frame.
2. **Form:** the stochastic rule of 5.3, for every pair.
3. **tau:** in a separate run.
4. **Light and vertical overlap (5.8.4) and the porpoise layout (5.8.5):** not decided, not built.

The rule has three outcomes, one uniform u per tick, observer, target, region and cell:

| u | the observer shows |
|---|---|
| u < P(met) | the density |
| P(met) <= u < P(entered) | 0: it was there and met nothing |
| u >= P(entered) | the own-cell reading of that channel after its own draw: it was never there (neighbours only) |

P(entered) is 1 in the own cell, so there the outcome is met or 0. The juvenile channel of a staged target follows the
target's draw. The own biomass, the energy and the own juveniles in the own cell are never drawn.

Library form (each `<obs>_observes_<target>` pair): `encounter_own` and `encounter_neighbour`, lists of [share, kappa].
- kappa is in encounters per 6 h tick per t/km2 of visible density; `.inf` marks a diffuse target.
- The own shares sum to 1; the neighbour shares sum to P(enter).
- P(met) = sum_k share_k (1 - exp(-kappa_k n)).
- `scripts/encounter_components.py` derives every pair from the inputs of 1.2:
  - own: one component per member x unit;
  - neighbour: the same rate per km of path, with the entered path split into two equal-count bins.
- The bins matter. With the mean path alone the neighbour P(met) at the box mean over-stated the 1.3 w by up to
  14 % (planktivores -> porpoises 0.119 against 0.104), because of Jensen's inequality. With two bins the largest
  error is 4.4 %, and a test holds it within 5 %.
- `scripts/apply_encounter.py` writes the listing into the library (66 pairs, replacing `neighbour_weight`). The
  seabird pairs are not derived; seabirds are muted.
- Inherited approximation (1.3): porpoise sonar is taken at the school radius by night too, so porpoises ->
  planktivores own reads [[0.5, 1.12], [0.5, 56000]].

Rules in code:
- **Loader.** Both keys are required together, and the own shares must sum to 1. The pair may not also carry
  `neighbour_weight`. Once one active pair carries the rule, every observation pair between active groups must,
  self pairs included, or the project is refused.
- **Tick length.** kappa scales with tick hours / 6. That is exact for the confined own cell; for the neighbours the
  6 h P(enter) is kept, an approximation.
- **Random numbers.** They come from the counter-based hash of `lib/gpu/random.py`, with a numpy twin in
  `lib/environments/ecosystem_env/encounter.py`, keyed by (world key = `current_world_seed` / rollout keys, tick,
  observer, target, region, cell). Both engines draw bit-identical uniforms. Observations from the same state agree
  up to the ulp of expm1, and a test checks this.
- **Parity.** The engines' state differs by float noise, so a draw can flip where u lies within that noise of P.
  Over many ticks the CPU and GPU trajectories therefore agree statistically, not to float noise. The fixture parity
  tests pass over 12 ticks.
- **Cost.**
  - Training (train_gpu, 30 x 30, 16 deltas, cuda-graph): 14.6 against 11.4 s per update, +28 %. The first form,
    which hashed int64 and looped over 12 components, cost +150 %.
  - CPU engine: +57 % per tick at 30 x 30 and +74 % at 60 x 60.
- **FG editor.** It keeps the lists on save. It locks the Neighbour Weight cell of a pair with the components and
  refuses a weight there; the components themselves are not edited in the editor.

Frozen fix102 policies with the rule (two seeds, as in 5.1): porpoises 0.35 / 0.18 / 0.04 and 0.27 / 0.12 / 0.075.
That is no change from 0.34 / 0.15 / 0.09 and 0.27 / 0.16 / 0.07, as 5.6 computed. Whiting fell to 0.42-0.44 at
year 3: policies trained on full information now see noisy inputs, which says nothing about a retrained run.
Every checkpoint is invalid: retrain.
