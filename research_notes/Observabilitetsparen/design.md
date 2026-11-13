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
