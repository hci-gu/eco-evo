# Capture probability and schooling in the fish pairs - summary (2026-11-11)

Combined derivation (user, 2026-11-11) of three notes, nothing applied:
- research_notes/"Fangstsannolikhet for fiskbyten": per-encounter capture p of a
  HUNGRY predator (user: hungry convention), central p per pair from the bluefish
  PPR line with q 0.8, r 0.3;
- research_notes/"Stimskydd i motes-a": the schooling factor F on the encounter a;
- research_notes/"Vitlinglojans stim": Norway pout sigma day 0.4 / night 0,
  N 10 000, R_s 13 m; blue whiting night 0.

The two factors are not independent: S = (R_s/R_i)^2 v_rel,s / (scan v_rel,i)
k / (N p), so the schooled share keeps its absolute rate while the dispersed share
rises with p, and F falls as p rises. F is therefore recomputed at each pair's NEW
p; the benthivore pairs use their own p (the schooling note had 0.2 where the
library has 0.3).

    factor on max_intake_rate = (p_new / p_library) x F(p_new)
    night/day' = night/day x F_night / F_day, stomach ratios kept (mackerel -> planktivores 0.16)

`scripts/combined.py` (stdout `scripts/combined_stdout.txt`, gate outputs and
`combined.json` in `tables/gates/`) reuses the pout note's school-factor algebra and
gate_p.py's runner; budget_gate.py runs unchanged on library copies in the session
scratchpad. Its `library` run is identical to today's gate and its `p_only` run to
the capture note's `central`.

```
python3 combined.py <repo> <scratch dir> ../tables/gates     # ~1 min
```

## Pair values (central; clupeids-only and pout-high readings in combined_stdout.txt)

| pair | a today | p | F at new p (day) | factor | a new | night/day today -> new |
|---|---|---|---|---|---|---|
| WC -> planktivores | 0.0174 | 0.20 -> 0.592 | 0.543 (0.514) | x1.61 | 0.0280 | 0.0594 -> 0.116 |
| WC -> benthivores | 0.0418 | 0.20 -> 0.592 | 1.000 | x2.96 | 0.124 | 0.165 |
| WC -> whiting | 0.0359 | 0.20 -> 0.592 | 0.988 | x2.92 | 0.105 | 0.137 -> 0.139 |
| WC -> mackerel | 0.000137 | 0.20 -> 0.549 | 0.089 | x0.24 | 3.35e-05 | 0.923 -> 0.916 |
| whiting -> planktivores | 0.0329 | 0.20 -> 0.566 | 0.682 (0.620) | x1.93 | 0.0636 | 0.212 -> 0.342 |
| whiting -> benthivores | 0.103 | 0.20 -> 0.522 | 1.000 | x2.61 | 0.269 | 0.565 |
| mackerel -> planktivores | 0.0248 | 0.20 -> 0.617 | 0.598 (0.209) | x1.85 | 0.0458 | 0.16 (stomachs, kept) |
| benthivores -> planktivores | 0.0214 | 0.30 -> 0.634 | 0.643 (0.592) | x1.36 | 0.0291 | 0.141 -> 0.238 |
| benthivores -> whiting | 0.0226 | 0.30 -> 0.617 | 0.985 | x2.03 | 0.0458 | 0.168 -> 0.171 |

Clupeids-only (pout and blue whiting not schooling): WC -> planktivores x2.22,
whiting -> planktivores x2.32, benthivores -> planktivores x1.88; pout-high
x1.05 / x1.27 / x0.78.

## Gate (implied / booked M2; equal cells unless noted; * = OFF)

| | library | p only | central | clupeids only | pout high |
|---|---|---|---|---|---|
| planktivores (prey total) | 0.69 | 1.24 | 0.89 | 1.07 | 0.65 |
| benthivores (prey total) | 0.62 | 0.86 | 1.06 | 0.95 | 1.19 |
| whiting (prey total) | 0.60 | 0.88 | 1.03 | 0.94 | 1.13 |
| krill (prey total) | 1.04 | 0.99 | 1.01 | 1.00 | 1.02 |
| WC -> planktivores | 0.78 | 1.18 | 0.88 | 1.04 | 0.67 |
| whiting -> planktivores | 0.54 | 1.30 | 0.95 | 1.10 | 0.65 |
| mackerel -> planktivores | 0.78 | 2.20 | 1.38 | 1.38 | 1.38 |
| benthivores -> planktivores | 0.61 | 1.22 | 0.81 | 1.10 | 0.48 |
| WC -> benthivores (field spread) | 0.97 (1.13) | 1.46 (1.71) | 2.00 (2.33) | 1.72 (2.00) | 2.36 (2.73) |
| WC -> whiting (field spread) | 0.64 (0.46) | 0.97 (0.70) | 1.31 (0.94) | 1.12 (0.81) | 1.54 (1.10) |
| benthivores -> whiting | 0.31* | 0.61 | 0.62 | 0.61 | 0.64 |
| whiting -> benthivores (field) | 0.12* (0.07*) | 0.27* (0.17*) | 0.29* (0.17*) | 0.28* | 0.31* |
| WC -> krill (field spread) | 1.39 (1.40) | 0.71 (0.72) | 0.97 (0.97) | 0.83 (0.84) | 1.14 (1.14) |
| WC -> zooplankton (field) | 0.38 (0.62) | 0.20 (0.32) | 0.27 (0.43) | 0.23 (0.37) | 0.31 (0.51) |
| whiting -> krill (field) | 1.46 (1.59) | 1.25 (1.40) | 1.33 (1.48) | 1.29 (1.44) | 1.40 (1.54) |
| mackerel -> krill | 1.67 | 1.54 | 1.61 | 1.61 | 1.61 |

OFF: library 5 pairs (benthivores -> whiting, WC -> mackerel, seals -> WC,
whiting -> benthivores, whiting -> epifauna at the field spread); every combined
reading 4 (benthivores -> whiting leaves OFF).

## Reading

- The planktivores move from 0.69 to 0.89 of their booked M2 (0.65-1.07 over the
  pout readings); every fish -> planktivore pair lies within 0.81-1.38. Mackerel ->
  planktivores at 1.38 is the largest, its F set by herring and sprat by day.
- The WC krill pair moves onto its booking (1.39 -> 0.97 of the booked share at the
  field spread): under shared handling the extra fish encounters take search time
  from krill. That is the krill-excess question of sections 267-269 seen from the
  predator side, here moved by fish-pair evidence, not by a krill value.
- Benthivores and whiting as prey reach their bookings (1.06, 1.03), but through the
  water column: WC -> juvenile benthivores 2.00 (2.33 at the field spread) and WC ->
  whiting 1.31 sit over the booking while benthivores -> whiting (0.62) and whiting
  -> benthivores (0.29, OFF, flagged open) sit under it. WC -> juvenile benthivores
  was already flagged in the vision and overlap notes as hinging on the overlap
  convention (243 time factors replaced by the band profile); it is the pair to fix
  before the totals mean anything.
- WC -> mackerel falls to x0.24 (mackerel always schooled, S ~ 1/p), still OFF;
  mackerel as prey has no other predator in the model.
- Night/day ratios rise on the planktivore pairs (fewer day encounters) and are
  unchanged elsewhere.

## Open before applying

1. WC -> juvenile benthivores (overshoot 2.0-2.3): overlap convention and window.
2. whiting -> benthivores (OFF, flagged open): whiting's height above the bed.
3. Hide floors: the planktivore pair floors 0.25 and mackerel's 0.7 lose their
   schooling meaning once F is in a (user: decide when a is done); with q in p, a
   floor below 1 would also double-count prey vigilance.
4. Norway pout's day state in the box (unobserved; clupeids-only vs central).

## User decisions

- 2026-11-11: investigate WC -> juvenile benthivores (overlap and window) first,
  then apply everything as one section. Nothing applied.
- 2026-11-12: applied as section 270 (commit 109d25f) with the decided
  corrections of the later notes: scripts/apply_270.py from
  tables/yaml_edits_270.txt (the "Langans indata" central library copy, the
  re-booking, the hide floors and the benthivores' recruit_depth_table).
