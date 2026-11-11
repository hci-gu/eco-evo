# Re-booking the planktivores' and mackerel's M2 on krill and zooplankton (2026-11-11)

Main agent. Question: what does re-booking the M2s from field-consistent rations do to the prey groups?
Library only, nothing changed. Script `scripts/rebook.py` -> `rebook_stdout.txt`. Ratios from
research_notes/"Planktivorernas energibudget" (Q/B 5.0 -> 5.8-6.6 at the field fill) and "Makrillens
ranson" (annual Q/B 2.2 -> presence-weighted SMS 2.75, x1.25, or the energetic need of the field fat
cycle 4.99, x2.27). Each predator's checked pairs scaled by the ratio, booked diet shares kept.

| case | krill M2 /yr | krill B/T, g = M2 | B/T, g kept 2.11 | zoo M2 /yr | zoo g change |
|---|---|---|---|---|---|
| library | 2.111 | 1.000 | 1.000 | 1.378 | 0 |
| plankt 5.8, mack x1.25 | 2.40 | 1.000 | 0.986 | 1.60 | +0.5 % |
| plankt 6.2, mack x1.25 | 2.54 | 1.000 | 0.979 | 1.71 | +0.8 % |
| plankt 6.2, mack x2.27 | 2.64 | 1.000 | 0.974 | 1.78 | +0.9 % |
| plankt 6.6, mack x2.27 | 2.77 | 1.000 | 0.968 | 1.89 | +1.2 % |

## Findings

1. **Zooplankton barely moves** (+0.5-1.2 % on g): its derived g is dominated by r_max 33.4 and M1
   ~22.7 /yr; the fish M2 is ~2 % of it.
2. **Krill: the pool level hardly depends on the choice.** With the exchange k 20 /yr, booking M2 2.4-2.8
   while keeping g at 2.11 (near the field P/B 2) puts the steady pool only 1.4-3.2 % below the outside
   density T; re-deriving g = M2 keeps B = T. Physically the kept g is the cleaner reading (local
   production at the field P/B, the extra consumption paid by import with the water), but the
   difference is within the noise of T itself [J].
3. **What re-booking changes is the yardstick.** The realised krill M2 in fix96's probe (4.7 /yr) is x2.23
   the booking today and x1.70-1.96 the re-booked values: the remaining excess is the diet shares and the
   full-reserve / autumn appetite (research_notes/"Makrillens ranson", "Vattenpelarens par per
   djupklass").

## Not covered

- Whiting's and the water column's rations were not re-examined (their krill M2 0.19 + 0.12); the
  benthivores' closure Q/B 4.94 vs booked 4.46 (x1.11) would move its krill pair 0.035 by +0.004.
- The planktivore M2s on other prey (mackerel -> planktivores 0.0186) do not enter any derived g
  (open population: g pays the somatic production).
