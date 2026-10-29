# Plan: a juvenile compartment inside the planktivore and benthivore groups (alternative B)

Status: plan only (2026-10-29); decisions taken (section 6). Nothing built. Data: `summary.md` in this folder. Code map: an
Explore pass over both engines (file:line references below are from it; `ENV/` =
lib/environments/ecosystem_env/, `GPU/` = lib/gpu/, `FG` = lib/world/functional_group.py).

## 1. What it is for

The prey-side shares derived so far are mostly juveniles inside adult pools: water column ->
benthivores f 0.075 (settled haddock inside a 19 cm gape), whiting -> benthivores 0.05, mackerel ->
planktivores 0.025 (0-group herring and sprat). A constant f cannot carry the seasonal pulse
(age 0 absent in May, 10-50 % of the August group, growing 12 -> 16 cm by November). A juvenile
compartment J makes that pulse a state: it fills when the 0-group arrives, grows, is eaten by the
small-gape predators and graduates into the adults.

## 2. Design

**Storage: J inside B (as the infauna refuge's A).** B stays the group total, so every reader of B
(rewards, biomass.jsonl, probes, heatmaps, frozen_years, the growth budget) is untouched. J is a
label on part of B.

**J carries the reserve pro rata.** The fill s = R/(B ME) is computed on B (FG:131-152, GPU
ecosystem.py:525-529); a reserve-less J would dilute s and move hunger, the starvation gate and the
growth cap. With R shared pro rata, eating J removes reserve as today's rule does
(ENV/predation.py:150-154), prey quality and `prey_includes_reserve` stay as they are, and the
growth budget's ideal intake is unchanged. (Box juvenile herring max fat ~6100 MJ/t against the
group's 6250, params_planktivorous_fish.md: close enough for a first version.)

**Flows on J (per tick):**
- Recruitment: a share rho(t) of the group's new tonnes goes to J - the income-growth tonnes for the
  planktivores (ENV/movement.py:345, GPU ecosystem.py:859), the positive growth delta for the
  benthivores (ENV/population_change.py:349, GPU :1159). Internal to B: g and the budget unchanged.
- Graduation: J -> adults at rate gamma, exact step J * exp(-gamma) (as ENV/refuge.py:63). Internal to B.
- Losses scale J by the same factor as B (bucket a): fishing (ENV/impacts.py:132-168, GPU :511-523),
  M1 (ENV/population_change.py:255-292, GPU :1120-1121), regional capacity loss (:297-309, GPU
  :1128-1132), starvation reduction (:332-350, GPU :1156-1159), extinction sweep and empty-cell
  zeroing (:373-430, GPU :1214-1218), the accessibility mask (ENV/grid_masks.py:37-51).
- Movement: J moves with the action fractions like B (bucket b): CPU apply_energy_costs /
  apply_movement and every (b, r) helper (ENV/movement.py:103-557: transfer, edge emigration,
  immigration concentration, split suppression, cancelled emigration, residues, currents); GPU
  movement / immigration / cancel_emigration / follow_residues / suppress_splits / advect
  (GPU/ecosystem.py:750-937, 1228-1272). The split logic itself decides on B and needs nothing.

**Predation: stage access per pair.** Generalise `accessible_fraction` f to a pair of shares
`stage_access: [f_J, f_A]`: the pair sees f_J J + f_A (B - J), and removes from each part in that
proportion. f_J = f_A = f is today's pair (inert, bit-identical); [1, 1] is the whole group. This
lets one pooled predator carry members with different gapes (saithe takes both stages, mackerel the
juveniles) without a second pair. Sites: the prey tensor next to accessible_rows (ENV/predation.py:
18-19, GPU :665-666) becomes per pair; apply_pair_access (ENV/interactions.py:753-763, GPU :687-691)
already rebuilds a per-pair visible biomass; Holling a_eff per pair (ENV/predation.py:57-63, GPU
:703-708); the harvest cap must be per compartment (ENV/predation.py:104-113, GPU :724-727 - today
one cap per prey column); removal split J / A (ENV/predation.py:142-172, GPU :735-748); diagnostics
by stage (ENV/predation.py:129-149). The eat mask (ENV/decisions.py:49-56) stays on B.

**Observations: none new in the first version.** The layout is unchanged, checkpoints load (their
meaning shifts, as with any library change). J/B varies in space only through local recruitment and
stage-selective predation; whether predators need to see it is a result to check, not an
assumption. A J channel later changes the dimension formula in three places (ENV/state.py:319-326,
train.py:1505-1557, GPU ecosystem.py:405-429) and invalidates every checkpoint.

## 3. Parameters and where they come from

| key | meaning | source |
|---|---|---|
| `juvenile.recruitment` rho(t) | share of the group's new tonnes entering J, by month | the arrival window: 0 Jan-May, the 0-group in by August (`summary.md`: Q2 age 0 = 0, Q3 0.10-0.53); haddock settlement mid-Jul to mid-Aug; herring metamorphosis Jun-Jul |
| `juvenile.graduation_rate` gamma | /yr | residence until the fish outgrow the small gapes (~age-1 summer): age 0 + 1 share falls Q2 -> Q3 of the second year |
| `juvenile.start_share` | J/B at the start date, by month | `summary.md` group table, 2001-25 level |
| pair `stage_access` [f_J, f_A] | per pair | member gapes x stage length distributions (the 223.31 method; age-0 lengths 9-16 cm, age 1 15-25 cm) |

rho and gamma are set from the survey stage table (field data). Validate on other field data: SMS
2023 partial M2 by prey age (age 0 vs 1+), the saithe and mackerel stomach shares already in
research_notes, never on model output.

Pairs to set (first pass): water column -> benthivores (replaces f 0.075), whiting -> benthivores
(0.05), water column -> planktivores (f 1 today; mackerel/horse mackerel on J, saithe both),
porpoises -> planktivores (small herring), seals -> benthivores (mostly adults), benthivores ->
planktivores (cod: both).

## 4. Work packages

1. **Research (no code).** rho(t), gamma, start shares; stage_access for the six pairs; SMS M2 by
   prey age for validation; trawl selectivity on J (the Skagerrak small-mesh sprat fishery takes
   juvenile herring, so "J fished like B" is the first-version default).
2. **CPU engine.** fg.juvenile (FG:96-129, cast at ENV/state.py:110-116); bucket a, b, c sites above;
   stage access in predation; recruitment and graduation in the DM growth branches.
3. **GPU engine.** A J tensor through import_state (GPU/ecosystem.py:444-450), the rollout buffers
   and CUDA-graph snapshot (GPU/rollout.py:42-48, 284-309), reset / _tick / snapshot (:95-359), the
   trainer warm-up and start state (GPU/trainer.py:130-132, 240-252, 303-313), the spawner
   (GPU/spawn.py:231-244), the step signature (GPU/ecosystem.py:1274-1307) and its callers.
4. **Loader, library, editor.** FG keys validated like exposed_share (FG:81-94, today non-DM only)
   with DM cross-checks (config_loader.py:828-853) and tables (ENV/state.py:376-423); the pair key
   parsed in build_interaction_matrices (ENV/interactions.py:84-151); tick rescale for gamma
   (lib/world/tick_time.py:159-205); editor fields and help text (fgconfig.py:782-787, 4566-4575,
   5032-5066; field_help.py; a two-value cell type for stage_access). Growth budget: no change.
5. **State writers.** inference.py:378 and :416 (b0 overrides) and viability.py:905-937 (colocate)
   rewrite B and must rescale J; GPU state_dict (GPU/trainer.py:379-389) and rollout snapshot
   (GPU/rollout.py:356-359) would drop J. Viewer: an optional J heatmap; frozen_years: diet and M2 by
   stage.
6. **Tests.** New tests/test_juvenile_stage.py: inert without the keys (bit-identical B, R),
   J <= B always, graduation exact, recruitment share, stage-split removal, J moves with B, fishing /
   M1 / starvation scale J. Extend the GPU parity harness (_assert_parity compares b and r only,
   tests/test_gpu_ecosystem.py:196-241) to J. Helper-signature tests: test_split_threshold.py:124-374,
   test_ecosystem_refactor.py. Pair shares (test_pair_shares.py) for the generalised f.
7. **Validation run.** pytest + 1-generation smoke; frozen 3 years on the current checkpoints (obs
   unchanged): J/B by quarter vs `summary.md`, realised M2 on J vs SMS age 0, the water column's
   benthivore take vs booked 0.083, the mackerel-side switch; then a training run --init-from the
   current policies.

## 5. Decisions for the user

1. **Timing driver.** A monthly rho table is a calendar input (life history, not behaviour: the
   behaviour rule is not touched, but it is the first calendar-keyed biology). Alternative: spawning
   from the reserve at a temperature or day-length cue plus a fixed larval delay (more code, still a
   calendar in disguise).
2. **Stage definition.** Age 0 only (the August pulse) or age 0 + 1 until the second summer (what
   the saithe gape takes). It sets gamma and how large J is (10-50 % vs 25-80 % of the August group).
3. **Whiting.** Its own juveniles are prey of saithe and cod (water column -> whiting f 0.47 today):
   the same compartment on the whiting group, or later.
4. **Observation channel now or later** (later keeps the checkpoints).
5. **Fishing on J:** as on B (default) or a selectivity share.

## 6. Decisions (user, 2026-10-29)

1. Timing: a monthly recruitment table rho(t) derived from the survey data.
2. Stage: age 0 + age 1, until the second summer (gamma ~ 1/yr; J 25-80 % of the August group).
3. Whiting: later; water column -> whiting keeps f 0.47 for now.
4. Observation channel: now. Every DM that observes a staged group gets a J channel; all
   checkpoints become invalid and training restarts from scratch (fix80 stopped by the user).
5. Fishing on J: its own selectivity per impact, derived before the build (work package 1).

Consequences for the plan: section 2's "no new observation channel" is replaced by a J channel
(dimension formula in ENV/state.py:319-326, train.py:1505-1557, GPU ecosystem.py:405-429; the
observation builders ENV/observations.py:20-107 and GPU ecosystem.py:553-590; inference.py's
mismatch hint); the impact tables get a juvenile selectivity share (ENV/impacts.py:132-168, GPU
:452-523) with its own tests.
