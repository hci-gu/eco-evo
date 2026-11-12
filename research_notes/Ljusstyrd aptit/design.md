# Design sketch: a light-entrained reserve set point (2026-11-12)

Status: built in section 271 (lib/world/set_point.py; the planktivores' library block). The s* item below was
falsified by the frozen check, see `frozen_check.md`. The user allowed (2026-11-12,
provisionally) an endogenous annual clock entrained by photoperiod as a physical driver, although at a fixed
latitude it carries the calendar. Evidence: `summary.md` and `evidence.md` in this folder. Preview arithmetic:
`scripts/set_point_preview.py` -> `tables/set_point_preview_stdout.txt` (on the fix98 tables in research_notes/
"Fix98 mot fix97").

## 1. Form

A defended reserve level s_T replaces the full reserve as the top of the room in `appetite: reserve_room`:

    room = max(s_T(t) B ME - R, 0)        (today: max(B ME - R, 0), i.e. s_T = 1)

Below s_T the fish eat at h = 1 as today, and the hyperphagia after a loss comes out by itself. Above s_T the room
is closed: eating gains nothing and costs the eat cost.

s_T of a member species is driven by the day length L and the sign of dL/dt only:

- The clock's progress g runs from 0 to 1. It starts at the onset day length L_on on FALLING L and reaches 1 at the
  low, L_low on RISING L.
- In between, g is linear in the photoperiod change travelled since the onset: (L_on - L) on the way down to the
  winter solstice, plus (L - L_ws) on the way up.
- Recovery: g falls linearly in L from 1 at L_low to 0 at L_top (rising). g = 0 from L_top to L_on.
- s_T = 1 - depth x g. It is continuous at both solstices.
- The group value is biomass-weighted: s_T,group = 1 - share x depth x g. The other members keep target 1.

The (L, sign dL/dt) pair at a fixed latitude is equivalent to the day of year. This is the rule tension the user
has accepted for now. The form uses no month keys and nothing from `start_fill`.

## 2. Values (planktivores; every value from the biology, none from model output or `start_fill`)

| parameter | central | range | source |
|---|---|---|---|
| member share | 0.554 (herring) | 0.575 with sprat | group weights (209, 243.3) |
| depth | 0.50 | 0.28-0.66 | Bull 1996: control lipid x0.49 Sep -> Apr under excess food (salmon), so 0.51; Berg 2024 herring condition swing 3.6-8.3 % of weight read as lipid, 0.28-0.66 |
| L_on (falling) | 13.64 h (1 Sep) | 11.05 h (1 Oct) - 15.06 h (15 Aug) | Metcalfe & Thorpe 1992: appetite falls from early September; Bull: onset early autumn, from midsummer in slow parr |
| L_low (rising) | 14.11 h (15 Apr) | L_ws 6.10 h (Berg phase) - 14.11 h | Bull: lipid low in April, no restoration after February deprivation; Berg: minimum at the solstice sample (+-1.5 months) |
| L_top (rising) | 17.44 h (1 Jun) | 14.11 h - 17.90 h | not measured; herring fatten May-Jul at the box, so the target must be open by then [J] |

The wet-tissue reading of Berg (0.04-0.08 fill) is dropped from the depth range. Herring replace lost fat with
water, so weight-at-length understates the fat loss.

Two phase cases, which name different predictions (table below):
- **Central (appetite timing).** Bull and Metcalfe measured appetite, in salmonids.
- **Berg phase (named alternative).** Berg measured herring condition, and allocation is not resolved there.

## 3. Preview (tables only, herring share, fix98 years 2-3)

Planktivore %BW/d, now → (a) cap only, policy keeps eating / (b) rest when the room is closed, eat just enough to
stay on s_T:

| case, depth 0.5 | Nov | Dec | Jan | Feb | Mar | Apr |
|---|---|---|---|---|---|---|
| now | 1.74 | 1.59 | 1.49 | 1.54 | 1.62 | 1.79 |
| central (a) | 1.08 | 1.06 | 0.90 | 0.75 | 0.86 | 1.91 |
| central (b) | 0.64 | 0.70 | 0.56 | 0.38 | 0.50 | 1.64 |
| Berg phase (a) | 0.79 | 1.14 | 1.55 | 1.79 | 1.86 | 1.62 |
| Berg phase (b) | 0.30 | 0.80 | 1.37 | 1.65 | 1.67 | 1.31 |

Depth 0.28-0.66 widens the central case to 0.63-1.20 in (a) and 0.24-0.79 in (b). The late onset changes Nov-Dec
only.

- The field's winter rations are 0.45-1.0 %BW/d. The central case lands inside them as a consequence. It was not
  chosen for that.
- The Berg phase predicts rations of 1.5-2.1 %BW/d in February-March, above the field: the fish refill after the
  solstice low. A retrained probe can tell the two cases apart.
- The rise in April is the target re-opening: the post-deprivation hyperphagia the salmon data show.

Caveats:
- The group's per-tonne costs stand in for herring.
- The group's own winter ration moves by 0.554 of these figures. Krill is 0.94 of the Dec-Jan ration, so the krill
  cut follows it.
- Material growth is scaled by (s_T - u)/(s - u). In case (b), winter growth runs on little food, which matches the
  field.
- The realised effect needs the frozen check and then the retrained probe (section 6).

## 4. Engine touchpoints

1. **Library.** One optional block on a DM with `appetite: reserve_room`, inert when absent:
   `reserve_set_point: {member_share, depth, onset_day_length_h, low_day_length_h, top_day_length_h}`.
   - The loader refuses it without the daylight calendar, without `reserve_room`, or with an inconsistent
     L_ws < L_low < L_top and L_on > L_ws.
   - FG editor fields go in the DM panel.
2. **Table.** A per-tick (T, N_dm) table `env.set_point_table` (1 for a DM without the block) on the daylight
   calendar, built next to `build_temperature_tables` in `interactions.py` and read with `light_index(env)`. The
   day length comes from `daylight.day_length_hours(lat)`, and the sign of dL/dt from the same series.
3. **CPU cap.** In `predation._reserve_room_cap` (line ~311): room = max(B ME s_T - R, 0).
4. **GPU mirror.** In `lib/gpu/ecosystem.py`:
   - the table as a tensor, as `ingestion_temp_table` (line ~243);
   - indexed by the rollout's year tick at the room in lines ~1003-1015.
5. **Growth budget.** `growth_budget.ideal_reserve`, reserve_room branch (lines ~275-293), and the kappa_m check
   (~719).
   - With a moving target the ideal reserve s* becomes the annual mean of s_T,group, capped by today's closed form.
   - Central: mean g 0.37, so s* = 1 - 0.554 x 0.5 x 0.37 = 0.90. The planktivores' derived g rises by
     (1 - u)/(s* - u) = 1.12.
   - **Falsified by the frozen check (`frozen_check.md`).** This assumed the fish grow all year at g (s - u). They
     do not: while the target falls (Sep-Mar) the reserve is released, intake stays below cost, and the material
     law's surplus term max(I - C, 0) (244.10) holds growth near zero. Production falls 0.58 -> 0.34 /yr.
   - **Applied instead (form 2, user).** Growth runs only while the target is not falling (15 Apr - 30 Aug), and
     g = P / (1460 mean(w (s_T - u))) = 0.00136 /tick pays P there. The frozen check restores production (0.64 /yr)
     and keeps the winter rations (`frozen_check.md`).
6. **Observation (the deciding item).**
   - Today the policy sees its own fill and the tick's light level, not the day length and not the target. When the
     room is closed it cannot tell that eating costs 1.65 RM for nothing.
   - Case (a) needs no observation. Case (b) does: section 154.5 showed that a rest probability near 0 gives a
     gradient near 0, so a fresh run must find rest from a signal that exists.
   - Proposal: one appetite channel for a DM with the block, the room fraction (s_T - s)/s_T clipped at 0. It
     is interoception: the fish perceives its hunger, not the calendar.
   - Since 270 a fish's rest is a pure feeding pause (floors 1.0), so the learned rest would mean exactly this.
7. **Prey side.**
   - `prey_includes_reserve`: herring are leaner as food in winter, so their predators' gain per tonne falls then.
     This is realistic, and nothing in the code changes.
   - The static quality at `reserve_reference_fill` stays.
   - budget_gate's booked qualities are unchanged.
8. **Starvation.** Unaffected: u 0.079 lies below the lowest group target (0.68 at depth 0.66).
9. **Checkpoints.** Invalid: the payoff changes, and the observation too if item 6 is taken.

## 5. Tests

- The table is continuous at both solstices, is 1 without the block and exactly 1 between L_top and L_on, and its
  minimum is 1 - share x depth.
- Group weighting.
- No intake when s >= s_T. Intake up to the room when s < s_T.
- CPU/GPU parity of the cap with the table.
- `ideal_reserve` with the mean target, and the derived g.
- The loader refuses the block without daylight or without `reserve_room`.
- The FG editor round trip.
- Observation width and value if item 6 is taken.

## 6. Sequence and the independent test

1. The code can be built and tested now, inert. Only a library edit reloads into the running fix98 probe (memory:
   library-edit-hits-running-probe).
2. When fix98 stops, add the library block.
3. **Frozen check.** Run the frozen fix98 policies with the block in the ledger. This gives case (a) mechanically
   and checks the arithmetic above.
4. Retrain (the user starts it).
5. **Independent test.** The Nov-Mar planktivore ration falls to 0.45-1.0 %BW/d (krill 0.10-0.46).
   - The monthly fill against `start_fill` is a secondary check only. It was never used to build s_T.
   - The rest share by month shows whether (b) was learned.
