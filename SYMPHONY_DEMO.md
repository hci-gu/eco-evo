# Symphony noise and trawling demo

Start the existing inference API with the dedicated demo profile:

```sh
uv run --locked python -u api.py --host 127.0.0.1 --port 8000 \
  --project mareld_symphony_demo.yaml
```

Omit `--project` to retain `mareld2.yaml`. The selected project applies to both
`GET /v1/models` and `POST /v1/runs`. Existing checkpoint directories are still
discovered under `results/`; no new checkpoint format or inference engine is used.

In Symphony, select and capture **MARELD POC review area**, enable **Shipping noise**
and/or **Bottom trawling** under **Demo impacts**, preview the inputs and run.
The synthetic example area is outside the source dataset. Baseline runs have both
impacts off. Keep comparisons sequential: the current inference engine seeds a
global NumPy RNG, so concurrent requests are not isolated paired experiments.

## What the demo means

The spatial patterns come from Symphony's historical relative indices. Shipping
noise is not measured dB, and bottom-trawl catch is a swept-area proxy. Both are
aggregated to the same 1 km grid as the simulated domain, quantized into up to 16
positive bands, and held constant for the whole simulation.

For positive band `k` (1-16) and strength `s` (0-2), Symphony sends:

- Noise: `min(140, 60 + 60 * (k / 16) * s)`.
- Trawling: `0.2 * (k / 16) * s`.
- Zero source or zero strength: zero additional pressure.

The API must not rescale these intensities again. These are illustrative mappings,
not calibrated ecological predictions. Source coverage and the assumption that
uncovered portions contribute zero are displayed in Symphony.

The demo profile enables `windfarm_noise` and `bottom_trawling` as **non-observable**
impacts. Both use the current biological library. The porpoise noise table
increases metabolic energy costs; noise does not directly cause biomass mortality
in this table. At noise 90, the metabolic multiplier is 1.275.

Bottom-trawling tables now use fishing effort in MW-h/km2/year, with different
responses for fish, porpoises, benthic infauna, mobile epifauna and Norway lobster.
At intensity 0.1, the direct infauna loss is 0.000003082 of biomass per six-hour
library tick; the planktivore loss is 0.0000000404483814. The infauna table is
linear to intensity 100 (loss 0.003082) and clipped above it. These replace the
older, stronger illustrative demo responses. The frontend's relative-index
mapping remains illustrative, so its intensities are not measured fishing effort
and visible effects can be small. No additional intensity rescaling is applied.
Existing growth, recruitment, predation and migration remain active. Indirect
food-web effects may change biomass outside the pressure mask.

Neither impact adds inputs to the policy networks. The demo does not claim learned
noise/trawling avoidance. Species, daylight and other project settings match the
current `mareld2.yaml`, including nutrients, temperature, depth habitat, juvenile
compartments and open populations. The original project remains unchanged. The
demo removes configured baseline and inference fields and mutes the other impacts.

## Species and checkpoints after the upstream update

Always discover the output channels through `GET /v1/models`. The current model
has 15 active groups: `benthivorousFish`, `predatoryFish`, `mackerel`, `whiting`,
`planktivorousFish`, `porpoises`, `seals`, `benthic_infauna`, `mobile_epifauna`,
`norway_lobster`, `phytoplankton`, `subsurface_phytoplankton`, `krill`,
`microzooplankton` and `zooplankton`. Preserve the returned order. Symphony's
EcoTwin adapter already uses the discovered species list for requests and result
validation; it does not require a fixed six-species output.

`benthic_community` remains an accepted alias for `benthic_infauna`, as upstream
defines it. The old `codfish` and `pelagicFish` channels no longer describe the
current model; rediscover species instead of relabelling or combining groups.

The group and observation changes invalidate old checkpoints. All five local
checkpoint directories found during the 2026-11-10 update contain the former
`gadoids`/`pelagic_fish` policies. They were preserved and need retraining.
Discovery lists directories, with `missing_policy_files` and an empty `species`
list for incomplete sets, so Symphony filters those out before model selection.
Loading still validates all required policies and their dimensions; incompatible
runs submitted directly return HTTP 422
`MODEL_CHECKPOINT_MISMATCH`, and their status endpoint records the failure.
Use policies trained for the current project and library. The demo and current
main project have the same policy input/output layout. A one-generation smoke
checkpoint only verifies execution; it is not a trained ecological result.

## Existing HTTP contract

`POST /v1/runs` remains multipart JSON `request` plus PNG files `texture`, `depth`
and repeated `mask` file fields. Every file has the exact request grid size;
rows are north to south, columns west to east. Binary masks are black outside,
white inside (RGBA PNGs are accepted). JSON `masks: [{id, file}]` references each
uploaded filename. Each pressure uses:

```json
{
  "id": "noise-08",
  "type": "noise_pressure",
  "scope": {"kind": "mask", "mask_id": "noise-08"},
  "parameters": {"intensity": 90},
  "time_window": {"start_step": 0, "end_step": 48}
}
```

Types retain their mapping: `noise_pressure` -> `windfarm_noise` and
`trawling_pressure` -> `bottom_trawling`. For any enabled input, Symphony first
sends a whole-grid zero reset for **both** types, followed by nonempty bands.
The API processes entries in order and overwrites their selected cells. Bands of
one impact are disjoint; different impacts can overlap. Land cells are kept zero.
At most 32 masks and 34 pressure entries are sent. The output tensor is unchanged.

With both inputs off, the API accepts `pressures: []` and no masks, as well as
the two whole-grid zero resets sent by the current Symphony backend. The demo
has no configured inference impact files, so both maps start at zero in either
baseline case. A zero-strength enabled input sends reset entries only.

Mask parser errors include duplicate uploaded filenames, duplicate IDs, missing
references, invalid PNGs and dimension mismatches. Non-finite, negative or
Float32-overflow intensities return 422. A nonzero pressure with no active consuming
response in the selected model also returns 422 (`UNSUPPORTED_PRESSURE`), instead
of silently accepting an inert map. Zero resets remain compatible with muted
impacts. Partial-run windows and unsupported pressure types remain rejected.

## Files and validation

- `api.py`: multipart parser repair, pressure validation and `--project` selection.
- `mareld_symphony_demo.yaml`: separate demo profile, with no inferred/external pressure maps.
- `fgconfig/fg_library.yaml`: current upstream responses, used without demo overrides.
- `lib/environments/ecosystem_env/impacts.py`: apply configured biomass mortality
  to all functional groups before predation, including non-decision makers.
- `api.py` also accepts a model whose only consuming response is benthic mortality;
  energy-only tables still require a decision maker.
- `tests/test_api_demo_impacts.py`: parser, map reconstruction, direct ecological
  terms, model/observation compatibility, and four real-engine HTTP scenarios.
- Existing `tests/test_api_inference_defaults.py` continues to check API/CLI defaults.

Run focused tests:

```sh
uv run --locked python -m pytest tests/test_api_demo_impacts.py \
  tests/test_impact_loading.py tests/test_api_inference_defaults.py -q
```

The direct-effect tests verify the noise multiplier and current library mortality
fractions described above. They also check zero pressure, mask boundaries,
the infauna response cap, loss accounting, empty cells, and shorter tick lengths.
HTTP tests use isolated generated checkpoints and compare baseline, noise-only,
trawling-only and both, asserting identical initial states, distinct final states,
lower benthos inside the trawling mask, and a baseline-identical empty-mask run.

Historical integration verification on 2026-10-12 (older biology and checkpoints):

- All 31 focused tests passed.
- Full suite: 896 passed, 100 skipped, one failure in
  `tests/test_viz_hover.py::test_the_tooltip_appears_with_only_pump_events_running`.
  This unchanged Pygame pixel test also fails in isolation with the dummy display.
- A one-generation training run on the demo project completed for all four decision
  makers. Its checkpoints were written outside the repository, under `/tmp`.
- Existing checkpoint `2026-10-08` completed four sequential HTTP runs (baseline,
  noise-only, trawling-only and both). Each produced a finite nonnegative tensor
  of shape `[2, 3, 4, 2]`, equal initial states and distinct final states.
- Symphony's actual 39 x 39 review grid completed all four 48-step scenarios via
  its frontend proxy, each returning 13 frames for six species. Result input
  labels matched the toggles. The combined UI result recorded 1,447 noise cells
  and 754 trawling cells with strength 1.0 for each.

Training smoke command (choose an unused output directory):

```sh
uv run --locked python train.py --project mareld_symphony_demo.yaml \
  --grid '6*6' --generations 1 --iter-per-gen 1 --n_eval_ticks 4 \
  --n_deltas 2 --top_deltas 1 --workers 1 --mortality on --migration on \
  --run-name /tmp/eco-evo-symphony-demo-training-20261012 < /dev/null
```

Implementation notes (2026-10-12): existing API mortality/migration/current settings
and the shared mass-balance default were preserved. No training algorithm changes,
new dependencies, scheduler, persistence, policy observation channels, or raster
upload schema were introduced. Project history is recorded here because the
working tree already had `mareld_resume.txt` deleted; that deletion was preserved.

## Upstream sync and verification, 2026-11-10

Fast-forwarded `coev-repro` by 63 commits, from `77175d0` to `14b8290`.
Local API and single-cell control work was retained; the action-diagnostics
merge keeps both the upstream diagnostics switch and the control projection.
The existing deletion of `mareld_resume.txt` was retained, so this local
integration record lives here. A safety stash retains the pre-sync tracked work.

The demo manifest now matches the upstream species, simulation and spawn
settings, with only its external-impact configuration changed. The inference
builder uses `DEFAULT_MASS_BALANCE`, fixing the API/CLI discrepancy without
removing the explicit legacy opt-out. API runs continue to use the CPU engine,
with mortality, migration and standard currents enabled.

Validation:

- 37 API tests passed, including all four impact scenarios and API/CLI parity.
- The 39 x 39 HTTP test uses the discovered 15-species list, a land boundary and
  an internal land hole, RGBA pressure masks, 49 ticks and sampling every four
  ticks. It verifies 14 frames including the final partial interval, float32
  bytes, species order, finite nonnegative biomass and the run-status endpoint.
- 91 focused API, control, determinism, impact and mass-balance tests passed.
- Full suite: 1619 passed, 222 skipped, 12 failed. All 12 failures also reproduce
  in an untouched checkout of `14b8290`: one rounded bed-share sum in
  `test_probe_diets.py`, two heatmap-scroll tests, one hover test and eight
  member-tooltip tests. They are unrelated to this integration.
- One generation of training completed for all eight decision makers on a
  6 x 6 grid. Outputs stayed under `/tmp/eco-evo-symphony-sync-training.l3wntk/`.
  Those real training checkpoints also completed an HTTP request returning
  a `[2, 3, 4, 15]` tensor. They are smoke-test artifacts, not validated policies.
- Python compilation and `git diff --check` passed.

Before a normal Symphony session, train a new checkpoint set with the current
library and the intended training settings, using a new run name, for example:

```sh
uv run --locked python train.py --project mareld_symphony_demo.yaml \
  --run-name symphony-current < /dev/null
```

Then start the API with the demo command above and refresh Symphony's model list.
