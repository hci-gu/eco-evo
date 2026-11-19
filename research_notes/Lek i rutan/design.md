# Spawning in the box: the model form (section 279)

The user's choices, 2026-11-18:
- the form: a closed reserve room plus a gamete energy drain;
- the benthivores' spawning share: 0.40, the probable value, not only the confirmed 0.19;
- refill: at once after each fish's spawning;
- scope: every group with a spawning share in the box.

The code is `lib/world/spawning.py`.

## Why not a cut in intake

A scratch check on frozen fix104 (u1267) drained 2 MJ/t/tick out of the benthivores' reserve for 90 days. That is 712 MJ/t, about 4x their annual gamete energy.
- Fill stayed at 0.99. The fish ate the energy back, and the ration rose from 6.1 to 7.8 t/t/yr.
- At 6 MJ/t/tick, fill fell only to 0.69, and the ration rose 45 %.
- The engines keep one reserve per cell. The model has a single pooled reserve per group and cell, so a cut in the intake of the spawning 40 % is refilled by the 60 % that feed.
- What a pooled reserve can carry is a lower top of the room: the group then cannot refill what its spawners lose.

## Form

Each `spawning` entry describes one member's mature fish in the box. Each fish starts spawning at an even spread over the window [spawn_start, spawn_end], given as day length and its sign, and spawns for `spawning_days`.

For one fish, e days into its spawning:
- gametes shed: `gamete_energy / spawning_days` per day;
- deficit: `(fast_cut x fast_loss + reserve_share x gamete_energy / spawning_days) x e`;
- the deficit is gone after its spawning (the user's refill choice).

For the group, per tonne:
- shed(t) = share x gamete_energy / spawning_days x P(spawning at t). Over the year it integrates to share x gamete_energy.
- cut(t) = share x deficit rate x E[e; 0 outside] / ME.

The engines:
- take shed(t) out of the reserve before the action costs;
- subtract cut(t) from the room's top (`env.set_point_table`).

Nothing else changes:
- No new observation slot.
- The set point's growth window stays its own.
- The growth budget pays P at the lowered fill: g = P / (1460 mean(min(1 - cut, s*) - u)).

## Mass

SMS P is the cohort Q1 -> Q1 increment, so it is net of the spawning loss (summary.md section 4). The term is therefore energy only and removes no tonnes.

## Known limits

- **The field trough is not reproduced.** The field May-Jul trough and its 3-6 month refill are not a spawning effect (summary.md section 5). With an immediate refill the group dips only:
  - benthivores to ~0.91 in March;
  - whiting to ~0.97 in March-May.
  The field start_fill bracket is 0.17-0.43. This stays open.
- **The research's 0.22 ME is a different quantity.** It is every spawner at its end-of-spawning deficit at once. The staggered starts and the immediate refill bring the group's cut down to at most 0.089. A slow refill would sit near 0.2.
- **The February ration drop is a pooled-room artefact.** The whole group's ration falls (3.6 against 6.0 t/t/yr), not only the ration of the ~0.16 of the group that are fasting cod. While the room top falls, the whole group stops eating.
- **The fast evidence is weak:** mostly [A] and [S] sources, and the haddock evidence conflicts. Long rough dab and whiting fasts are judgements.
- **Gonad build-up is not modelled.** It runs from October to the spawning; the gametes are charged when shed. The food-funded part is then eaten back while they are shed, not during the build-up.

## Frozen check (fix104 u1267, seed 0, one probe year, a library copy)

- **Benthivores.**
  - Fill by month: 0.99 0.94 0.91 0.94 0.99 0.99 ... (without the term 1.00 all year).
  - Ration: 3.6 in February (6.0 without), then 6.1-6.4 while refilling.
  - Gametes booked: 179 MJ/t of the start biomass (table: 181).
- **Whiting:** fill 0.95-0.97 in March-June; gametes 249 MJ/t.
- **Water column and planktivores:** no visible change.
