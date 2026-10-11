# Model context: how the Mareld simulator defines and uses natural mortality (M1) for the "gadoids" FG

Scope: model / repo facts only, read from the local repository /home/simon/eco-evo
on 2026-10-12 (branch coev-repro, HEAD ef06a09 plus uncommitted edits to
mareld_resume.txt). Each "source" is a file:line reference in that repository, not
a web URL. Tags: **[M]** model fact read in the code or config; **[R]** stated in
the repo's own documentation (resume, report, research notes); **[C]** computed for
this note from repo numbers.

## 1. What the library sets for gadoids (species block, predation pairs, muted predators)

### Takeaway
Gadoids carry `natural_mortality` 0.000468 per 6 h tick (= 0.683 /yr). The library
comment says this is M0 0.392 + non-FG predation 0.171 + benthos (crab) predation
0.121, from North Sea Ecopath 1991. `r_max` is 0.511 /yr. In mareld2.yaml the
budget counts two predation pairs on gadoids: cannibalism 0.2168 and porpoises
0.0557. Seals (0.0299) are muted. Pelagic fish (0.0799) and seabirds (0.0078) are
unchecked.

### Cited Findings
- [M] Species block `gadoids`, decision maker: `maintenance_level: 0.5` (u),
  `max_energy_reserve: 2500.0` (MJ/t), `resting_metabolism: 6.0`,
  `energy_content: 5500.0` (MJ/t), `max_intake_rate: 0.04`, `starve_rate: 0.012`,
  `interference: 1.0`, `metabolism_q10: 2.2`, `metabolism_t_ref: annual_mean`
  — fgconfig/fg_library.yaml:198-243 (u :201, reserve :202, metabolism :203,
  energy_content :214, starve_rate :240, interference :242).
- [M] `r_max: 0.511` with the comment "biomass-weighted FishBase CMSY priors over
  cod, whiting, haddock, saithe, hake and the other gadoids (Norway pout and blue
  whiting excluded: they eat zooplankton)" — fgconfig/fg_library.yaml:221-224.
- [M] `growth_rate: 0.00200951`, labelled DERIVED: "recomputed by the loader for
  every run; the number stored here is what the loader computes for mareld2.yaml
  (seals and seabirds muted) with --mortality on; with --mortality off the loader
  leaves M1 out. Edit r_max, natural_mortality or the predation matrix, not this
  value." — fgconfig/fg_library.yaml:216-225.
- [M] M1 composition comment, verbatim: "M1 = M0 (0.392) + predation by non-FG
  groups (0.171) + by benthos, which cannot hunt in the tick (0.121) = 0.683 /yr,
  North Sea Ecopath 1991. Cannibalism is the gadoids_preys_on_gadoids pair. Was
  7.0e-05." followed by `natural_mortality: 0.000468` —
  fgconfig/fg_library.yaml:226-230.
- [C] Per-tick to per-year conversion with the repo's loss rule
  (`-TICKS_PER_YEAR * ln(1 - m)`, TICKS_PER_YEAR = 1460 at 6 h ticks,
  lib/world/growth_budget.py:57, 71-78): -1460 * ln(1 - 0.000468) = **0.6834 /yr**.
  The previous value 7.0e-05 per tick is about 0.102 /yr.
- [M] Pairs where gadoids are the prey (`*_preys_on_gadoids`):
  - `gadoids_preys_on_gadoids`: preys_on **true**, predation_mortality **0.2168**,
    assimilation 0.8, handling_time 159.0. Comment: "~5 % of the gadoid diet in
    North Sea Ecopath 1991 (juvenile cod, whiting, haddock, saithe eaten by the
    adults); M2 0.2168 /yr is the report's gadoid-on-gadoid share" —
    fgconfig/fg_library.yaml:545-553.
  - `porpoises_preys_on_gadoids`: preys_on **true**, predation_mortality
    **0.0557**, max_intake_rate 0.2, assimilation 0.84, h 28.6,
    visibility_floor 0.75 — fgconfig/fg_library.yaml:649-664.
  - `seals_preys_on_gadoids`: preys_on **true**, predation_mortality **0.0299**,
    but seals are muted in mareld2.yaml — fgconfig/fg_library.yaml:497-501.
  - `pelagic_fish_preys_on_gadoids`: preys_on **false**, predation_mortality
    0.0799 — fgconfig/fg_library.yaml:586-590.
  - `seabirds_preys_on_gadoids`: preys_on **false**, predation_mortality 0.0078
    — fgconfig/fg_library.yaml:689-693.
  - `benthic_community_preys_on_gadoids`: preys_on false and no M2. Benthos is an
    NDM and cannot hunt in the tick, so crab predation sits in M1 —
    fgconfig/fg_library.yaml:511-514, :226-228.
  - `zooplankton_`/`phytoplankton_preys_on_gadoids`: false —
    fgconfig/fg_library.yaml:617-620, 749-752.
- [M] Pairs where gadoids are the predator (`gadoids_preys_on_*`), which affect
  the prey's budgets rather than the gadoids' own:
  - `gadoids_preys_on_pelagic_fish`: M2 0.2997, assimilation 0.8, h 217.0,
    functional_response 3, dark_ratio 0.6, light_threshold_deg -5.0 —
    fgconfig/fg_library.yaml:431-467.
  - `gadoids_preys_on_benthic_community`: M2 0.0048, assimilation 0.7, h 87.0 —
    fgconfig/fg_library.yaml:539-544.
  - `gadoids_preys_on_zooplankton`: M2 0.089, assimilation 0.8, h 130.0 —
    fgconfig/fg_library.yaml:570-580.
  - `gadoids_preys_on_gadoids`: see above.
  - Phytoplankton, porpoises, seabirds and seals: preys_on false —
    fgconfig/fg_library.yaml:554-569.
- [M] mareld2.yaml: decision makers gadoids, pelagic_fish, porpoises and
  zooplankton are active. `seabirds` is `muted: true` (mareld2.yaml:83-87) and so
  is `seals` (mareld2.yaml:88-92). Every impact variable (pelagic_trawling, rotor,
  windfarm_noise) is also muted (mareld2.yaml:106-134). Gadoid initial biomass is
  1250-5000 t, with 2500 t for inference (mareld2.yaml:71-74). Temperature layer
  for gadoids: `at_50m` (mareld2.yaml:69).
- [M] The loader drops muted FGs from the active predator set, "so their
  predation share is not paid for" — lib/config/config_loader.py:711-714.

### Inferences
- [C] The full Ecopath 1991 natural mortality of the gadoid pool, M = 1.06 /yr
  (report, see section 5), is split in the model like this:
  - M1 0.683, which is applied as a loss and paid for in g;
  - checked M2 0.2725 (cannibalism 0.2168 + porpoises 0.0557), paid for in g
    and, in principle, taken as explicit predation;
  - 0.118 /yr that is neither applied nor paid for in mareld2: pelagic fish
    0.080, seals 0.030 (muted), seabirds 0.008.

  The sum 0.683 + 0.2725 + 0.118 = 1.07 matches the report's M = 1.06 up to
  rounding.
- The number that a literature M0 recommendation would replace is the **0.392**
  component inside M1. The other two components, non-FG predation 0.171 and
  benthos 0.121, are not M0 in the strict sense, but they live in the same
  `natural_mortality` parameter.

### Gaps
- The library does not break down by species which non-FG predators make up the
  0.171. The report names them only in general terms: "sharks, rays, flatfish,
  other fish, carnivorous zooplankton, larvae" (reports/Litteraturbaserad growth
  rate per FG.md:76-78).

## 2. How M1 enters the derived growth rate (growth_budget.py)

### Takeaway
For a decision maker, g = (r_max + M1 + sum of checked M2) / ((1 - u) * 1460).
The budget is paid in full only at a full energy reserve (s = 1), which the code
documents as "ideal conditions". Every unit of M1 therefore adds one unit of
gross production at s = 1. A smaller M0 lowers both the loss and g, and leaves
net low-density growth at r_max.

### Cited Findings
- [M] Budget docstring: "decision maker: gross = r_max + M1 + sum_j M2_ij; g =
  gross / ((1 - u) * TICKS_PER_YEAR)" — lib/world/growth_budget.py:12-14.
- [M] "r_max ... For a DM the literature maximum NET population growth at low
  density (recruitment and individual growth included, all natural mortality
  removed)" — lib/world/growth_budget.py:19-23.
- [M] "M1 the FG's own natural_mortality (per tick, converted to a per-year rate).
  It holds every loss the model does not resolve as predation by another FG." —
  lib/world/growth_budget.py:24-26.
- [M] M2 is counted only while the pair is checked (`preys_on`), the predator is a
  decision maker and the predator is active in the project —
  lib/world/growth_budget.py:27-31 (docstring) and :103-109 (code).
- [M] "1 - u the largest energy surplus, reached at a full reserve (s = 1, where
  the hunger gate closes). Ideal conditions mean a full reserve." —
  lib/world/growth_budget.py:32-34.
- [M] With `--mortality off` the loader passes `include_m1=False`, "removing a loss
  removes its growth compensation" — lib/world/growth_budget.py:36-39, :187-188.
- [M] `--mortality` defaults to **off** on train.py (train.py:1735) and
  inference.py (inference.py:1039). The library's stored g of 0.00200951 assumes
  `--mortality on` (fgconfig/fg_library.yaml:217-219). The ten-year probe world in
  resume section 156 ran with mortality on (mareld_resume.txt:14163-14167).
- [C] Check against the stored value: (0.511 + 0.6834 + 0.2168 + 0.0557) / (0.5 *
  1460) = 1.4669 / 730 = **0.0020095**, equal to the library's 0.00200951.
- [R] Before section 141.4 checked the cannibalism pair, g was 0.00171 =
  (0.511 + 0.683 + 0.056) / 730. That is the figure in reports/Litteraturbaserad
  growth rate per FG.md:268 and :301-304 and in resume 129.2
  (mareld_resume.txt:12686-12707). Section 141.4 says "gadoids_preys_on_gadoids on
  (cannibalism, M2 0.2168 ...): derived gadoid growth_rate 0.00171252 ->
  0.00200951" (mareld_resume.txt:13333-13341).
- [R] Section 129.1 states the rule: M1 "now holds ONLY losses that are not
  predation by an FG: M0, predation by non-FG groups, and predation by NDMs (they
  cannot hunt in the tick)" — mareld_resume.txt:12663-12684.
- [R] The definition was decided by the user: "growth_rate is the maximum growth
  under ideal conditions, taken from the literature. It includes recruitment and
  individual growth, and excludes natural mortality: gross P/B = r_max + M1 + sum
  of checked M2_ij" — reports/Litteraturbaserad growth rate per FG.md:64-78.

### Inferences
- [C] Sensitivity: each 0.1 /yr change in M1 moves g by 0.1/730 = 1.37e-4 per tick
  (about 6.8 % of the current g). Production at a given s moves by 0.1 * (s -
  u)/(1 - u) /yr, while the loss moves by the full 0.1 /yr. Below s = 1, a lower
  M0 therefore helps the gadoids on net, and a higher M0 hurts them: at s = 0.72
  the loss changes by 0.1 and the production by only 0.044.
- [C] Because g contains M1, the relation is not "lower M0 means more gadoids"
  one for one. At s = 1 a change in M0 is exactly neutral. The net effect is
  (s - 1)/(1 - u) * dM0, which scales with how far the realised reserve fill
  sits below 1.

### Gaps
- None for the formula itself.

## 3. How M1 is applied per tick (population_change.py, natural_mortality.py, loss accounting)

### Takeaway
Every tick, M1 removes a constant fraction `natural_mortality` of each cell's
biomass and the same fraction of its energy reserve (keep = 1 - rate). The loss
is booked as "natural". It runs before the gated growth step and does not depend
on behaviour, because exposure weighting is not set for gadoids.

### Cited Findings
- [M] `natural_mortality` is multiplied by `env.mortality_multiplier`
  (`--mortality_multiplier`, default 1.0) and applied only if
  `env.apply_natural_mortality` — lib/environments/ecosystem_env/population_change.py:171-176.
- [M] Scalar path: `keep = np.float32(max(0.0, 1.0 - natural_mortality))`, then
  `fg.energy_reserve *= keep` and `fg.biomass *= keep` —
  population_change.py:183, :198-200.
- [M] The loss is booked to `env.loss_natural[fg_id]` ("Book M1 like the other
  causes, so the loss breakdown can show it (section 132)") —
  population_change.py:201-207. The breakdown categories are starvation,
  predation, impact, natural and total —
  lib/environments/ecosystem_env/loss_accounting.py:6-21.
- [M] Exposure-weighted M1 (visual/tactile split, section 139) is "Off unless an FG
  sets m1_visual_share or m1_tactile_share; the tick is then bit-identical to the
  constant M1" — lib/environments/ecosystem_env/natural_mortality.py:34-35,
  :59-64. The gadoid block sets neither key (fgconfig/fg_library.yaml:198-243),
  so gadoids use the scalar path.
- [M] Growth step right after M1: `energy_surplus = energy_level -
  maintenance_level`; `rate = growth_rate` if surplus >= 0, else `starve_rate`;
  `biomass_delta = biomass * rate * energy_surplus` —
  population_change.py:208-217.
- [M] Mass balance is on by default (`mass_balance=True`,
  lib/environments/ecosystem_env/environment.py:55). Positive growth is capped
  by the reserve above the maintenance line and debited from it: `dB = min(B * g *
  q, E_avail / energy_content)`, `R -= dB * energy_content` —
  population_change.py:88-142, :222-223.
- [R] The tick order is decisions, impact mortality, predation, movement, growth;
  M1 is applied inside the population-change (growth) step — AGENTS.md section 5;
  population_change.py:171-244.

### Inferences
- [C] At 0.000468 per tick, the per-tick survival is 0.999532, so annual survival
  from M1 alone is exp(-0.683) = 0.505. The model removes half the gadoid biomass
  each year from M1 alone, before explicit predation and starvation.
- Since M1 scales the reserve along with the biomass, it leaves the reserve fill
  s = R/(B*ME) unchanged. It does not by itself push the population toward
  starvation.

### Gaps
- None.

## 4. What the gadoids FG represents (species pool and weights)

### Takeaway
"gadoids" pools North Sea 1991 Ecopath groups: cod, whiting, haddock and saithe
(juvenile and adult), hake, "other large gadoids" and "other small gadoids". Norway
pout and blue whiting are excluded. By Ecopath 1991 biomass (pool B 1.96 t/km2)
the pool is about 29 % whiting, 26 % saithe, 20 % haddock, 12 % cod, 12 % other
gadoids and 0.7 % hake. About 44 % of the pool biomass is juveniles.

### Cited Findings
- [R] Pool definition: "cod, whiting, haddock, saithe (juvenile and adult), hake;
  'other large gadoids': pollack, tusk, ling, greater forkbeard; 'other small
  gadoids': poor cod, bib, silvery pout, four-bearded, five-bearded, three-bearded
  and shore rockling" — reports/Litteraturbaserad growth rate per FG.md:268-275.
- [R] Excluded: "Norway pout ('zooplanktivorous, trophic level 3.59')" and "Blue
  whiting (euphausiids)". "Norway pout is the largest gadoid biomass in the North
  Sea (1.39 t/km2), so this choice matters" — same file :277-283. Variant not
  applied: with Norway pout and blue whiting included, g would be 0.0027 (:49-51).
- [R] Per-species r_max (FishBase CMSY): cod 0.51, whiting 0.49, haddock 0.50,
  saithe 0.52, hake 0.48. Large others about 0.55, small others about 0.57.
  Biomass-weighted 0.51 (CL about 0.33-0.76). Myers et al. 1997 cod r_m: Kattegat
  0.53, Skagerrak 0.82, North Sea 0.56 — same file :285-291.
- [R] Ecopath 1991 biomasses (t/km2) from Table 3.3: cod juv 0.079 / adult 0.161,
  whiting 0.222 / 0.352, haddock 0.284 / 0.104, saithe 0.281 / 0.22 —
  research_notes/Litteraturbaserad growth rate per FG/fish.md:26-34, :152. Hake
  0.014, other gadoids (large) 0.0486, other gadoids (small) 0.1909 —
  research_notes/Litteraturbaserad growth rate per FG/ecopath_ns1991/params.json
  (groups 21, 24, 25).
- [R] Data-quality caveat in params.json: rows 15 and 17 (juvenile whiting and
  haddock) have shifted columns (B 2.36 and 2.0 are really P/B values), so the
  biomasses above come from fish.md. The report also notes that rows 24-25 ("other
  gadoids") of the printed Ecopath table are duplicated and that several gadoid
  diet columns do not sum to 1 — reports/Litteraturbaserad growth rate per FG.md:395-401.
- [C] Pool shares by biomass (total 1.9565 t/km2, matching the report's 1.96):

  | Group | Share of pool biomass | Split (juvenile / adult) |
  |---|---|---|
  | Whiting | 29.3 % | 11.3 / 18.0 |
  | Saithe | 25.6 % | 14.4 / 11.2 |
  | Haddock | 19.8 % | 14.5 / 5.3 |
  | Cod | 12.3 % | 4.0 / 8.2 |
  | Other small gadoids | 9.8 % | - |
  | Other large gadoids | 2.5 % | - |
  | Hake | 0.7 % | - |

  Juveniles of the four main species make up 44.2 % of the pool.
- [R] Pool Ecopath values: B 1.96 t/km2, P/B 1.46, F 0.40, so M = 1.06. "It is high
  because juvenile whiting and haddock and the small gadoids have P/B 2.0-2.4" —
  reports/Litteraturbaserad growth rate per FG.md:294-295.
- [R] Ecology of the umbrella group: "haddock mainly benthivorous, cod eats crabs,
  Nephrops, shrimp, polychaetes; whiting and saithe are fish / krill eaters" —
  mareld_resume.txt:14334-14336 (section 156.10).

### Inferences
- [C] The pool is dominated by whiting, saithe and haddock, not cod, and is
  juvenile-heavy. A literature M0 that is cod-centric (for example the
  assessment convention M = 0.2 for adult cod) would describe only about 8 % of
  the pool biomass, the adult cod.

### Gaps
- Biomasses for the individual species inside "other large" and "other small"
  gadoids were not found in the repo; the report gives only the r_max values.

## 5. Where the 0.392 M0 comes from, and what the repo already knows about gadoid M0 / P/B

### Takeaway
The 0.392 is the Ecopath 1991 "other mortality" of the pool, M0 = P/B x (1 - EE),
derived from a fished state (F about 0.40-0.44). The library takes M = P/B - F, so
the model is unfished by construction. The repo's own notes flag Ecopath M0 as
something that "absorbs balancing error". Per-stage M0 varies widely, from 0.03
(adult haddock) to 1.09 (juvenile haddock).

### Cited Findings
- [R] Split of M for the pool: "M = 1.06 splits into M0 0.39, predation by non-FG
  groups 0.17, by benthos (crabs; NDMs cannot hunt, so it stays in M1) 0.12, and
  predation by FGs: gadoids (cannibalism) 0.217, pelagic fish 0.080, porpoises
  0.056, seals 0.030, seabirds 0.008" and "M1 = 0.39 + 0.17 + 0.12 = 0.683 per yr =
  4.68e-4 per tick" — reports/Litteraturbaserad growth rate per FG.md:296-300.
- [R] Validation table: gadoids M 1.06, M2 by Mareld predators 0.086 (porpoises
  0.056 + seals 0.030), M2 by unmodelled predators 0.58, M0 0.39 —
  reports/Litteraturbaserad growth rate per FG.md:388-391. Computed M2 matches the
  report's EE-based M2 (cod juvenile 1.06 vs 1.02, whiting juvenile 1.97 vs 1.92)
  — same file :382-386.
- [R] Ecopath convention used: "P/B = Z = F + M2 + M0 (Allen 1971), where ... M0 =
  'other'/unexplained mortality = P/B x (1 - EE)" —
  research_notes/Litteraturbaserad growth rate per FG/fish.md:4.
- [R] Per-stage decomposition, North Sea 1991, computed in the earlier research
  from Mackinson & Daskalov 2007 Tables 3.3/3.5 (fish.md:53-70):

  | Stage | P/B | F | M2 | M0 |
  |---|---|---|---|---|
  | Cod juvenile | 1.79 | 0.66 | 1.02 | 0.11 |
  | Cod adult | 1.19 | 0.77 | 0.12 | 0.30 |
  | Whiting juvenile | 2.36 | 0.11 | 1.92 | 0.33 |
  | Whiting adult | 0.89 | 0.52 | 0.31 | 0.06 |
  | Haddock juvenile | 2.00 | 0.22 | 0.68 | 1.09 |
  | Haddock adult | 1.14 | 0.92 | 0.19 | 0.03 |
  | Saithe juvenile | 1.00 | 0.20 | 0.12 | 0.69 |
  | Saithe adult | 0.95 | 0.53 | 0.06 | 0.36 |

- [R] "Biomass-weighted M0: ... cod+whiting+saithe 0.32; cod+whiting+haddock+saithe
  0.43. ... This is a flag, not a prescription: Ecopath M0 absorbs balancing
  error." — research_notes/Litteraturbaserad growth rate per FG/fish.md:98. The
  aggregate table gives cod+whiting+saithe B 1.315, P/B 1.26, F 0.42, M2 0.52,
  M0 0.32, M 0.84 (fish.md:163).
- [R] Exploitation dominates adult gadoid P/B. For adult saithe and monkfish "P/B
  nearly equals F", and Ecosim no-fishing runs gave "unrealistic" recovery for
  adult saithe, cod and other large gadoids, so Mackinson & Daskalov increased
  biomass to reduce F/Z — research_notes/Litteraturbaserad growth rate per
  FG/fish.md:39, :112.
- [R] Cod convention: "M = 0.2 was accepted by the researchers studying each
  population" (Myers, Mertz & Fowlow 1997), and "the value of M assumed invariant
  in most cod stock assessments" (Bjornsson et al. 2022) —
  research_notes/Litteraturbaserad growth rate per FG/fish.md:91.
- [R] ICES SMS key runs split M into residual M1 and predation M2 per species and
  age, but "the 2023 key-run values are published only as figures; I could not
  extract numeric M1/M2" — research_notes/Litteraturbaserad growth rate per
  FG/fish.md:84-89.
- [R] Unfished-state reasoning: at steady state with F = 0, P/B tends to M and the
  age structure shifts toward older fish, so unfished P/B is lower than the
  exploited Ecopath values. Adult-gadoid-only M is 0.375 /yr —
  research_notes/Litteraturbaserad growth rate per FG/fish.md:108, :117-118.
- [R] Section 141.1: "Ecopath 1991 is a fished state: F ~0.44 gadoids, ~0.40
  pelagics (derived from Table 3.5); the library takes M = P/B - F, so the model is
  F = 0 by construction. ICES today (subagent): F ~0.12 gadoids, ~0.20 pelagics
  (biomass-weighted, inference). 2 yr, 4 seeds, r_max - F: gadoid mean 1.31 ->
  1.22 (-0.19: 1.16)" — mareld_resume.txt:13307-13311.
- [R] The earlier report found that "Mortality, not growth, is where the library
  was furthest off. M1 rose 4-7x for the fish ... Most natural mortality of gadoids
  comes from sources Mareld does not model (M0, other fish, crabs); that part
  stays in M1." — reports/Litteraturbaserad growth rate per FG.md:53-56. When
  predation by modelled FGs was moved out of M1, gadoid M1 fell from 0.98 to 0.68
  /yr (same file :586-588).

### Inferences
- Two figures for the 1991 fishing mortality coexist in the repo: F 0.40 for the
  pool (report :294) and F about 0.44 for gadoids (resume 141.1). They differ in
  scope (the full pool vs a narrower derivation from Table 3.5), but both say
  that the 1991 state was heavily fished.
- [C] The 0.392 is biomass-weighted across a juvenile-heavy pool. Juvenile
  haddock (M0 1.09, 14.5 % of B) and juvenile saithe (M0 0.69, 14.4 % of B) carry
  much of it, while adult whiting and haddock have M0 of only 0.03-0.06. In an
  unfished age structure with more adults, a biomass-weighted M0 would likely be
  lower. This is an inference from fish.md:118, not a repo computation.

### Gaps
- No numeric ICES SMS residual M1 per species and age was retrieved (fish.md:84,
  :103). This is the most direct literature counterpart to the model's M0 and
  remains open.

## 6. What the model probes measured about gadoid decline (resume sections 141, 154, 156)

### Takeaway
In the ten-year null-scenario rollout (resume 156) gadoids fell from 2300 to 1520 t
(-1.3 %/yr, slowing). Their reserve fill sat at about 0.6-0.76, well below s = 1,
so they realised only part of the designed production. The 0.217 /yr cannibalism
paid for in g was almost never taken (realised about 0.001), and porpoise
predation fell from 0.087 to 0.020. The structural reason the probes give is the
growth budget being "paid at s = 1 only".

### Cited Findings
- [R] Setup (156.1): frozen fix53 checkpoint, deterministic probe world 60x60 with
  mortality, migration, currents, mass balance, daylight + temperature, calendar
  start May 28, 14600 ticks = 10 years, every impact muted (null scenario) —
  mareld_resume.txt:14163-14179.
- [R] Trends, annual means year 1 -> 10 (156.2): "gadoids 2300 -> 1520 (-1.3 %/yr,
  slowing)"; "Not at equilibrium after 10 years; everything past year 1 is also
  outside the 1460-tick training episodes" — mareld_resume.txt:14180-14185.
- [R] Realised vs budgeted M2 (156.4), per year over years 1..10 —
  mareld_resume.txt:14208-14222:

  | Pair | Budgeted (library) | Realised |
  |---|---|---|
  | gadoids -> gadoids | 0.217 | about 0.001 ("growth paid for a loss never taken") |
  | gadoids -> benthos | 0.0048 | 0.33 -> 0.17 (see 156.10) |
  | gadoids -> herring | 0.30 | 0.19 -> 0.14 |
  | porpoises -> gadoids | 0.056 | 0.087 -> 0.020 |

- [R] Section 156.4: "The derived g = budget / ((1 - u) * 1460) pays the full
  budget at s = 1 only (growth_budget.py, 'ideal conditions'), where the hunger
  gate closes intake. s sits at 0.6-0.75, so gadoids and porpoises realise 40-50 %
  of the designed production; for porpoises that does not even cover M1. By
  design, but it is a structural reason the top predators decline." —
  mareld_resume.txt:14223-14229.
- [R] Section 154.6 (fix53 checkpoint of 01:32, probe world from May): gadoid
  "reserve fill s 0.69-0.76, hunger gate 0.48-0.62 (u 0.5), so they are not full".
  Diet May-Aug zooplankton 59-77 %, herring 18-32 %; Oct-Feb benthos 55-61 %,
  herring about 40 %; cannibalism 0 % — mareld_resume.txt:14022-14030.
- [R] Section 156.9, open items: "(2) growth budget paid at s = 1 only; (3) gadoid
  cannibalism absent, benthos share ~2x Ecopath (156.10)" —
  mareld_resume.txt:14310-14316.
- [R] Section 156.10: the model's gadoid diet in years 2-10 is benthos 65 %,
  herring 26-28 %, zooplankton 7-10 %, gadoids 0 %. The renormalised Ecopath
  figures are about 28 / 40 / 25 / 6 %. Q/B is 5.7 vs 4.3-4.5. Verdict:
  "direction and seasonal switch plausible, share elevated, not one of the gross
  deviations" — mareld_resume.txt:14317-14342.
- [R] Section 141.3 (2 yr, 4 seeds, trained policies, before 141.4): gadoid mean
  1.31 (current), 1.30 (Type III), 1.21 (Type III + r_max - F) —
  mareld_resume.txt:13323-13332.
- [R] Section 141.4, frozen rig after enabling cannibalism and zooplankton prey:
  "Gadoids now hold a stable stock (drift 1.09) on four prey" (greedy 0.343 PASS,
  eat 0.372 PASS) — mareld_resume.txt:13333-13352.
- [C] Realised production at a given reserve fill. From the growth step
  (population_change.py:208-217), production per year = g * 1460 * (s - u) =
  2.934 * (s - 0.5):

  | Reserve fill s | Production (/yr) |
  |---|---|
  | 0.60 | 0.29 |
  | 0.69 | 0.56 |
  | 0.71 | 0.62 |
  | 0.73 | 0.67 |
  | 0.75 | 0.73 |
  | 0.76 | 0.76 |
  | 1.00 | 1.467 (= r_max + M1 + M2) |

  The designed loss is M1 0.683 plus realised predation of about 0.02-0.09
  (porpoises 0.087 -> 0.020, cannibalism about 0.001; 156.4). So at s of about
  0.71-0.73, production of 0.62-0.67 /yr falls just short of the losses, which
  fits the slow -1.3 %/yr decline. This is approximate: the mass-balance cap and
  cells below u (starvation branch) make the realised mean production lower than
  the formula at mean s.

### Inferences
- [C] In this frame M1 (0.683) is by far the largest loss on gadoids in the
  rollout: about 8-30x the realised explicit predation. Lowering M0 by dM0 changes
  the net growth rate by about dM0 * (1 - s)/(1 - u), which is +0.054 per 0.1 cut
  at s = 0.73. That is comparable in size to the observed -0.013 /yr trend. The
  model would therefore be sensitive to an M0 revision of even 0.05-0.1 /yr. Read
  this as a first-order estimate, not a measured result.
- The unspent cannibalism budget (0.217 paid, about 0.001 taken) works in the
  opposite direction to M1: it gives the gadoids extra production for a loss they
  never suffer. Without it the decline would be steeper, all else equal. An M0
  revision and the cannibalism and s = 1 issues (156.9 items 2 and 3) are coupled
  in the gadoid budget.

### Gaps
- The figures "gadoid production 0.61-0.67 /yr vs M1 0.683 + predation 0.06-0.08"
  and "s ~0.71-0.73" do **not** appear verbatim in mareld_resume.txt (grep for
  0.61-0.67 / 0.71 / 0.73 near "gadoid" found nothing in sections 154-156). The
  production range can be reproduced as a computation (table above, s 0.71-0.73
  gives 0.62-0.67). The closest repo s ranges are 0.6-0.75 (156.4) and 0.69-0.76
  (154.6). For predation, the repo gives porpoises 0.087 -> 0.020 and cannibalism
  about 0.001 (156.4), not 0.06-0.08.
- No repo probe has varied gadoid `natural_mortality` (or `--mortality_multiplier`
  for gadoids alone) in a multi-year rollout. The effect of an M0 change on the
  ten-year trend has not been measured.
- Note that 156.1-156.4 use the 20:57 checkpoint, which the trainer overwrote at
  22:27 (mareld_resume.txt:14293-14297). Gadoids were "unchanged within the seed
  spread" in the 22:27 runs (:14289-14290).
