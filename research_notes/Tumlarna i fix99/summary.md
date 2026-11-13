# Porpoises in fix99: why they starve

Diagnostic only; nothing changed in code or library. Frozen fix99 policies at
update 454 (copied 2026-11-12 19:23), probe world (30x30, unfished, the run's
progress/config.json), library = fg_library.yaml at 271, CPU, 2 years.
Scripts in `scripts/`, outputs in `tables/`.

## Findings

1. **The loss is starvation, not predation or the edge.** Base arm: 2.12 t
   starvation catabolism, 0.93 t extinction sweep (booked as starvation),
   0.14 t M1, out of 3.0 t at the start. 0.011 t was left after two years.

2. **Energetics are not the limit.**
   - budget_gate closes on any single prey at the start densities.
   - With full appetite (arm `room`: `appetite: reserve_room` on porpoises) the
     group fill is 0.88-0.97, and the porpoises still fall to 0.38 with 2.2 t
     starvation.
   - Closing the porpoises' edge (arm `closed`) does not help: B goes to 0.
   - Continuous fractions (arm `nounits`, env.unit_splits = False) end at 0.05.

3. **Cells sit still on empty water until they die.** Per-cell event tracing
   (`events_u454_*`):
   - The worst episodes are cells of 2-6 animals that stay 10-40 ticks on cells
     with 0.0-0.3 t of planktivores and benthivores. The grid means are 3.9 and
     1.6 t/cell.
   - Example, cell (21,6) in the `room` arm, ticks 56-80 (`cell_21_6_room.txt`):
     5.6 animals at s 0.47, with 0.03 t planktivores in the cell and 101-232 t
     one cell north.
   - The raw policy is eat on one prey 0.88-0.97 and move east 0.01-0.10. After
     `merge_subunit_fractions` the move is 0 (0.095 x 0.143 t is below the 0.05 t
     unit).
   - The cell starves to zero in 24 ticks while eating 1.000.

4. **The movement signal is weak, and the policy collapsed onto eat before using
   it.** training.jsonl action statistics (cell-mean over active cells, after the
   unit merge; DM index 3 = porpoises, 5 = seals):
   - Porpoise move share is 0.00 at update 1, while the fish are at 0.3-0.55 and
     seals at 0.01-0.03.
   - It stays 0.00-0.04 in fix96, fix97, fix98 and fix99 alike.
   - Entropy falls 1.5 -> 0.15-0.5, and eat is 0.96-0.99.
   - A cell of N animals moves only if one direction's probability is at least
     1/N. Training cells hold about 2-6 animals (spawn floor 5 units), so a
     random policy (about 0.05 per direction) rarely moves, and a small change
     of that probability is merged away.
   - The gradient is weak but not absent. Seals sit in the same unit regime
     (0.055 t, a similar number of animals) and learned some movement: 0.01 ->
     0.06-0.18 within 200-300 updates. Why porpoises did not is not resolved.
   - The immobility is old, not new in fix99.

5. **The prey patch moves away; the porpoise does not follow.**
   - In the porpoise cells planktivores start at 2.5-12x the grid mean (spawn on
     planktivores). They fall to 0.1-0.7x within months, after which the diet
     shifts to benthivores or whiting.
   - Compared with fix97 (end checkpoint, lib269, same script, 1 year;
     `tables/budget_fix97_base.txt`):
     - Planktivores in the porpoise cells stay at 2-8x the grid mean in fix97,
       against 0.1-0.7x in fix99.
     - The planktivore share of the diet is 0.56-0.82 in fix97 and 0.05-0.19 in
       fix99 Jun-Nov, when benthivores carry 0.6-0.9.
     - Ration and fill: 18-21 t/t/yr at fill 0.77-0.79 in fix97, against 24-30
       at 0.65-0.70 in fix99.
     - Starvation stops after the first summer in fix97 (1.49 t left at the year
       end), but continues all year in fix99.
   - What changed in 270: the four fish -> planktivore floors went from 0.25 to
     1.0 (porpoises 0.95 -> 1.0). Resting no longer hides a planktivore from fish,
     so moving is its only defence, and the planktivores also observe porpoises.
   - The data fit "the prey leaves and the immobile porpoise cannot follow".
     Planktivore movement rates and avoidance of porpoises were not measured
     separately.

## Effect size

Porpoise biomass is about 0.1 % of the fish, and their booked M2 on any prey is
at most 0.02 /yr. The web is not affected. Viability (14/15) and the porpoises'
own trajectory are.

## Options (user's decision)

- A. Start at the SCANS-IV level (0.519 /km2, about 84 t on 60x60; resume
  264.11, where the manifest's 12 t has no source). With 7x the animals per cell
  the move threshold 1/N falls to about 0.03, so a move gradient exists. Retrain.
- B. Stochastic rounding of sub-unit fractions in `merge_subunit_fractions`
  (both engines): a fraction p N < 1 moves one whole animal with probability
  p N. Indivisible per tick, expected flow continuous in p, so ARS gets a
  stronger signal. Code change plus retrain.
  - Cost: the draws must come from the CRN stream shared by the +/- candidates
    and run inside the CUDA-graph step. Otherwise ARS sees noise rather than a
    gradient.
- Not recommended: entropy or exploration bonuses (section 128 keeps them off),
  or training without units and inferring with them (train/inference mismatch).

## Follow-up

- 2026-11-13: option A applied as section 272 (porpoise start 84 t, spawn range 42-168 t,
  booked M2 x7) together with a denser spawn (`spawn_floor_units: 30`, 30 animals per cell).
  B not applied.
- 2026-11-13: the 30-animal spawn floor dissolved within a day and was removed again (272.4).
  fix100 (warm-started at 84 t) crashes to 0.42-0.51 within 16 days of a probe start; with the
  planktivores blind to porpoises the first year ends at 0.60 instead of 0.34
  (`tables/fix100_u830_*`, `scripts/porp_blind.py`; resume 272.4).
