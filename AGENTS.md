# AGENTS.md - Mareld / Poseidon Nord

Distilled project context for AI agents. Keep it short; the full history lives in
`mareld_resume.txt` (see "Deep context" at the bottom).

## 1. What this project is

Agent-based marine ecosystem simulator for the Mareld / Poseidon Nord offshore case.
Functional groups (FGs) live on a 60x60 grid (1 km/cell, 6 h/tick) and the
decision-making FGs are driven by small policy networks trained with ARS
(Augmented Random Search), co-evolution on by default.

`Strategi.pdf` is authoritative for the mathematics ("The Tick"); `Method.pdf`
defines observation/neighbourhood conventions.

## 2. Language policy (hard rule)

- All code, comments, variable names, config keys, UI strings, prints and docs in **English**.
- **Chat answers and status updates to the user: Swedish.**
- Swedish display names only as an optional UI overlay (`sv_mapping` in `fgconfig.py`).
- `mareld_resume.txt` is **ASCII-only**; that restriction applies to that file only.
  Verify: `python3 -c "print(sum(1 for b in open('mareld_resume.txt','rb').read() if b>127))"` -> must print 0.

## 3. Repository map (essentials)

| Path | Role |
|---|---|
| `train.py` | ARS training entry point. Requires `--project`. Co-evolution default-on. |
| `inference.py` | Runs trained policies from `results/<run-name>/policy_<fg>.pth`. |
| `run_mareld_mvp.py`, `compare_scenarios.py` | MVP scenario run / "noll" vs "projekt" comparison. |
| `mareld2.yaml` | Project manifest (`--project` value). |
| `fgconfig/fgconfig.py` | Tkinter GUI for FGs, impacts, matrices, spawn strategies. |
| `fgconfig/fg_library.yaml` | Central biological library (species/interaction/impact definitions). |
| `lib/environments/ecosystem.py` | `EcosystemEnvironment` - "The Tick" pipeline, heavily optimised. |
| `lib/runners/trainer.py`, `parallel_worker.py` | ARS trainer (CRN, ARS-V2 obs-norm, top-b) + multiprocessing worker. |
| `lib/world/`, `lib/config/config_loader.py` | Grid, `FunctionalGroup`, project/library loading. |
| `lib/spawn/`, `lib/viz/`, `tools/` | Spawn strategies, live pygame visualiser, offline plot/calibration tools. |
| `lib/environments/ecosystem_env/source_tracking.py` | `--local_reward`: per-cell source tracking, `reward(c)=B(c,t+1)/A(c,t)`. |
| `lib/diagnostics/viability.py`, `tools/viability.py` | Long-term viability rig - frozen behaviour, no ARS. Two factors (behaviour x spawn geometry); the verdict comes from the normative corner `--spawn colocated --behaviour greedy`. The hand-coded arms allocate the eat mass by marginal energy return (water-filling), never evenly - section 92. Criterion in `VIABILITY.md` (sections 90, 91, 92). |
| `tests/` | pytest suite - keep green. |
| `results/<run-name>/` | Checkpoints `{'state_dict': ..., 'obs_stats': {...}}`. |

## 4. Functional groups

- Decision makers (have a policy net): `zooplankton`, `pelagic_fish`, `gadoids`,
  `porpoises`, `seals`, `seabirds`.
- Non decision makers (logistic growth toward carrying capacity):
  `phytoplankton`, `benthic_community`.
- English snake_case IDs are canonical.

## 5. The Tick (per tick, random FG order)

`_calculate_decisions` -> `_apply_impact_mortality` -> `_apply_predation`
-> `_apply_movement` -> `_apply_growth`

- Decisions: batched `torch.bmm` over all DMs; per-DM compact observation layout
  driven by the Observability matrix; `Rest` doubles as a hide action.
- Impacts: linear interpolation in `impact_table`, clipped to endpoints.
- Predation: vectorised, Holling type II with Beddington-DeAngelis crowding
  (`a_eff = a / (1 + a*h*B_prey_visible + w*B_pred)`), hidden (rested) fraction
  protected, energy gain buffered. `w` is the per-FG `interference` parameter
  (`interference: 0` = pure Holling; `gadoids: 1.0`, `pelagic_fish: 0.7` in the
  library, section 86). It is what removes the 2-cell vertical bands.
- Movement: slice-assign, per-action metabolic cost; energy follows biomass.
- Growth: NDM logistic; DM `s = R/(B*ME)`, `q = s - u` (`maintenance_level`, default 0.3).

## 6. Commands

```bash
# Train (mandatory --project). Non-interactive: the [Y/n] prompt blocks otherwise.
python3 train.py --project mareld2.yaml --run-name <name> < /dev/null
python3 train.py --project mareld2.yaml --species pelagic_fish \
    --generations 1 --iter-per-gen 1 --workers 1     # smoke test

# Tests and syntax check
python3 -m pytest tests/ -q
python3 -m py_compile fgconfig/fgconfig.py lib/environments/ecosystem.py \
    lib/runners/trainer.py lib/runners/parallel_worker.py train.py

# Is the world viable at all? Frozen behaviour, no training. Exit 1 = not viable.
python3 tools/viability.py                                  # ~4 min, see VIABILITY.md
python3 tools/viability.py --spawn configured               # geometry as drawn
python3 tools/viability.py --ticks 300 --seeds 1 --behaviour eat   # smoke test

# Can each DM close its energy budget over its RATION? Seconds, no rollout.
# "min share" = minimum share of the best prey in the diet; a pair that
# cannot pay on its own is low-quality food, not pure loss (section 92).
python3 tools/probes/budget_gate.py

# GUI, inference, plots
python3 fgconfig/fgconfig.py
python3 inference.py --project mareld2.yaml --run-name <name>
python3 tools/biomass_html.py results/<run-name>      # -> plots.html
```

Useful flags: `--profile sanity|info|deep` (forces action-hack-free settings),
`--visual`, `--rollouts_per_delta`, `--mortality on`,
`--mortality_multiplier FACTOR` (scales every FG's `natural_mortality`;
`keep = 1 - FACTOR*rate`, needs `--mortality on`, section 88), `--migration on`,
`--policynetwork LAYERS NODES [ACTIVATION]`, `--resume`,
`--local_reward` (per-cell source-tracked reward, sections 79 and 80;
available on `train_gpu.py` as well). Prefer
`--local_reward_norm grid` with the default `log` metric: `sum` rewards
spreading thin and `mean` rewards killing the worst cells (sections 82, 84).

## 7. Working conventions

- **Biology drives, not action-hacks.** Fix parameters/mechanisms before adding
  shaping terms; profiles deliberately disable `argmax_penalty`, `entropy_coef`
  and temperature annealing.
- Don't redo performance work (batching, vectorisation, slice-assign, CRN)
  without measuring first.
- Changing the Observability matrix, observable impacts, the FG set or the grid
  layout invalidates existing `.pth` checkpoints (retraining required).
- Commit **only** on explicit user request, with
  `--trailer "Co-authored-by: Junie <junie@jetbrains.com>"`.
- Untracked artifacts outside agent scope unless asked: `make_*.py`, `*.png`,
  `ab_test_*.py`, `resume.md`, `konvergensproblem.txt`, PDFs, `results/`.
- Always validate changes with the pytest suite plus a 1-generation training run.

## 8. Deep context - `mareld_resume.txt`

`mareld_resume.txt` (~5700 lines) is the full project history: every design
decision, calibration, A/B test and open problem, organised in numbered sections
(0 language policy, 1 repo layout, 3 the tick, 5 CLI, 6 project format,
7 GUI, 9-10 ARS/co-evolution, 11 known discrepancies vs the PDFs, 14 git log,
26-28 spawn and multi-world, 35-36 ecological fixes, 41 observability/hide,
46-49 starvation and reward shaping, 63-71 latest diagnostic arcs,
79 local per-cell reward, 90-92 the viability rig).

**Do not read it end-to-end** - it is large and will eat the context window.
Search it for the topic at hand and read only the matching section, e.g.:

```bash
grep -n "^[0-9]\+\(\.[0-9A-Z]\+\)\? " mareld_resume.txt   # section index
```

Keep it updated: when an arc of work finishes, append a new numbered section
there (ASCII only) rather than expanding this file.

`resume.md` is a separate, narrower resume for the `fgconfig/fgconfig.py` GUI.
