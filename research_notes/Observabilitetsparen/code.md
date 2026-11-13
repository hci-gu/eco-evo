# Observation layout today and the cost of a cell / neighbour split

Verified 2026-11-13 on the code at 302a126 (mareld2.yaml, seabirds muted).

## Layout

- One switch per (observer, target) pair: `<obs>_observes_<target>: observes: true` in the
  library becomes the observer's `observes` list; `state.py:316-328` builds
  `env.obs_others_idx[i]` from it.
- Every observed group appears in the observer's own cell AND in each of the four
  neighbour cells: centre `[B_own, E_own, B_others (k), J (m), impacts]`, each neighbour
  `[B_own, B_others (k), J (m), impacts]` (`state.py:353-370`;
  `observations.py:100-120`; GPU `lib/gpu/ecosystem.py:596-625`, `obs_indices`).
- A staged group's juvenile channel (J) follows the same rule (centre and neighbours).
- Hiding cuts the observed biomass by the pair's floor, in centre and neighbours alike
  (research_notes/"Gom och sinnen").

## Widths per decision maker (probe env, fix100 config)

| DM | inputs | observed groups | J channels | neighbour inputs |
|---|---|---|---|---|
| benthivores | 72 | 10 | 3 | 56 |
| mackerel | 62 | 8 | 3 | 48 |
| planktivores | 63 | 8 | 3 | 48 |
| porpoises | 52 | 6 | 3 | 40 |
| water column | 62 | 8 | 3 | 48 |
| seals | 52 | 6 | 3 | 40 |
| whiting | 67 | 9 | 3 | 52 |
| zooplankton | 22 | 3 | 0 | 16 |

About three quarters of every policy's inputs are neighbour channels.

Zooplankton has `movement_speed` 0 (its four move actions are masked, section 232.4): its
16 neighbour inputs (own biomass and phytoplankton, subsurface phytoplankton,
microzooplankton in four cells) describe cells it can never choose to enter. The currents
move it; at most the neighbours foretell next tick's inflow. Under either reading of a
neighbour channel these inputs carry no decision the group can act on [F, code].

## Cost of a per-pair scope (own cell only / own cell and neighbours)

A new per-pair key (e.g. `observes_neighbours: false` on the observation pair) read in:

1. `state.py`: a second index list per DM (`obs_neighbour_idx`, a subset of
   `obs_others_idx`), the juvenile channels split the same way, `per_dm_in_dim`.
2. `observations.py`: the neighbour loop over the subset; the visibility floors aligned
   with it.
3. GPU `ecosystem.py` `obs_indices`: the neighbour block over the subset.
4. `train.py`'s own input-width formula (it counted the set-point slot in 271).
5. The FG editor's Observability matrix (a tri-state cell or a second matrix) and its
   round trip.
6. Tests: layout parity CPU/GPU, widths, editor round trip.

Every change of the matrix or the scope changes input widths: all checkpoints invalid,
retrain. The build is worth it only if several pairs need "own cell only"; if the answer
is all-or-nothing per pair, today's switch suffices.
