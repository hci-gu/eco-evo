# Fix97 as the realised baseline of sections 267-269 (2026-11-12)

Fix97 (results/pzbpgp248nsrpfix97, train_gpu.py, started 2026-11-11 13:33, five minutes after the section 269
commit 7153a6c) was trained on the 267-269 library; it was stopped at update 670 (68 long-run probes) so section
270 could be applied. Its checkpoints are invalid for 270 (payoffs and observation meanings changed): it is the
269 baseline to compare a 270 run with, not a warm start (its policies learned to hide when hiding protected
against fish).

How the table was made: `tools/probes/probe_diets.py` reads the library named in the run's
`progress/config.json`, which is fgconfig/fg_library.yaml (now 270). To judge fix97 against its own bookings,
copy `progress/` to a scratch dir, point `config.json` "library" at the 269 library (`git show
7153a6c:fgconfig/fg_library.yaml` or the commit before 109d25f) and run probe_diets on the copy. Output (probe 68,
three probe years): `tables/probe_diets_fix97_lib269.txt`.

## Findings (realised / booked)

- The krill excess was NOT closed by 269 in the trained run: realised krill M2 4.6-5.1 /yr vs booked 2.54 (x1.8-2.0),
  where the 269 budget_gate said 1.04. Per predator: planktivores x1.6-1.7 (0.82 of the booking), whiting
  x1.6-1.9, water column x2.1, mackerel x6.1-6.6.
- Much of it is RATION, not diet shares: realised Q/B mackerel 6.3-7.0 (field 2.75, x2.3-2.5), benthivores 6.0-6.4
  (4.46), planktivores 6.8-7.5 (6.2), water column 3.3 (2.83). The full-reserve / autumn appetite question
  (deferred in 268) shows here.
- Fish predators under-realised their fish prey (what 270 addresses): planktivore M2 0.22-0.25 vs booked 0.36;
  whiting -> planktivores 0.2, benthivores -> planktivores 0.2-0.55, whiting -> benthivores 0.14-0.24.
  Water column -> benthivores 0.87-1.28 of the 269 booking 0.0833.
- Other OFF rows: benthivores -> Nephrops x6.4-6.9; seals and porpoises -> water column ~0; zooplankton ->
  phytoplankton mixed vs subsurface split (0.24 vs 2.4 of the booked shares).

## Consequences

- budget_gate (start densities, booked rations) is not a sufficient judge: the trained outcome deviated x1.9 on
  krill where the gate read 1.04. Judge 270 with the long-run probe of a 270 run against this baseline.
- The next main lever is the ration (appetite at a full reserve), mackerel first; section 270 moved the fish-pair
  encounters but not planktivores -> krill (the largest krill flow) nor the rations.
