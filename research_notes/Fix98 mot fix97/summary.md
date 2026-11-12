# Fix98 (section 270) against fix97 (269) (2026-11-12)

Snapshot: fix98 = results/pzbpgp248nsrpfix98 (train_gpu.py on the 270 library, started 2026-11-12 00:17, still
training), long-run probe 64 at update 630, one seed (20260530), three probe years. Fix97 = probe 68 at update 670
on the 269 library (research_notes/"Fix97 som baslinje"). Year-3 trends below are drift in a run that is still
training, not equilibria.

Method:
- `tools/probes/probe_diets.py results/pzbpgp248nsrpfix98` (library 270 = the run's own; clean in git):
  `tables/probe_diets_fix98.txt`.
- The ledger and its analyses (`research_notes/Ransonen i energi/scripts/`: ledger.py, limits.py, analyse.py,
  seasonal.py, fieldfill.py, prey_timing.py) on a frozen copy of the update-630 policies (CPU engine, the
  probe world): `tables/*_fix98.*`.
- `research_notes/Ljusstyrd aptit/scripts/effect_size.py` with the fix98 ledger in place of fix97's:
  `tables/effect_size_fix98.txt`.
- `scripts/bed_share.py` (benthivore biomass by bed class, both runs on their own libraries):
  `tables/bed_share_fix9{7,8}.txt`.
- `scripts/porpoise.py` (porpoise trajectory, first year).

## Rations and fill: unchanged by 270

- Realised rations, years 2-3 (fix98 / fix97, t/t/yr): planktivores 6.74 / 6.78, mackerel 6.24 / 6.33, whiting
  3.63 / 3.60, water column 3.25 / 3.25.
- Benthivores are the one exception: 4.92 / 5.83 (year 3 4.39, field Q/B 4.46), with their energy ratio to the field
  Q/B 1.12 / 1.15.
- Energy against the need on the field fill cycle is 1.06-1.08 for every resident group, as in fix97.
- Fill is 0.96-1.00 in every month for every fish group. The reserve room is the only binding limit: the
  pre-cap/post-cap gain is 1.5-2.8 for the planktivores, 1.3-1.7 for the benthivores and 1.4-2.2 for whiting.
- The planktivores' Nov-Mar krill ration is 0.96-1.49 %BW/d (field herring 0.10, group 0.24), and in March their
  fill is 0.99 (field 0.26).
- The trained policies learned no winter rest. The planktivores' eat share is 0.98 in December-February, while the
  room cap discards a third of the encounter. With h = 1 and growth paid out of the tick's food, eating always pays,
  so training will not produce a winter appetite drop on its own.

## Krill: the excess is unchanged

- Krill M2 realised 4.38 / 4.87 / 5.05 /yr against booked 2.54 (fix97 4.63 / 4.86 / 5.08).
- The year-3 excess of 2.51 /yr, by predator:
  - planktivores 1.00 (3.07 vs 2.07);
  - mackerel 0.69 (0.82 vs 0.125);
  - benthivores 0.43 (0.46 vs 0.035; new);
  - whiting 0.20;
  - water column 0.20.
- Herring-share set point (the light-cued appetite; Nov-Mar herring at the field krill ration, pout as realised),
  re-based on fix98:
  - five-group krill flow -10.7 to -17.6 % (central -16.6 %); fix97 gave -12.0 to -19.3 %;
  - whole planktivore group -21 to -30 %; whiting -1.6 to -3.3 %.
- The herring cut of 1 800-3 300 t/yr removes roughly half to all of the planktivores' own excess, about 3 700 t/yr
  on the CPU ledger's flow at the realised ratio 1.48. That leaves total krill M2 at about x1.6-1.75 the booking.
- The open pool relaxes at k = 20 /yr, so a herring cut raises krill B, and with it the other predators' encounter.
  The realised cut is therefore smaller than this counterfactual.
- Mackerel's excess is outside any appetite mechanism. Its ration (x2.3 the SMS Q/B) matches its energy need of
  about 6.0, so the low yardstick is the booking (research_notes/"Ransonen i energi").

## Benthivores at the field bed spread meet krill at x9-13 the booking (new)

- Biomass share by bed class (< 100 / 100-150 / 150-250 / >= 250 m), year 3:
  - fix98: 0.335 / 0.472 / 0.106 / 0.087;
  - fix97: 0.020 / 0.303 / 0.520 / 0.156;
  - GOV adults in the box: 0.308 / 0.441 / 0.191 / 0.060 (field_diet.FIELD_STAGE_BED_SHARES).
- Fix98's benthivores move onto the field's shallow beds over the three years (< 150 m: 0.44, 0.67, 0.81). There
  their krill pair's depth access is 0.0225 (< 100 m) against 0.0058 (100-150 m).
- Their krill share rises 0.04 → 0.16 → 0.25 (booked 0.027), and prey-side M2 runs x1.4 → x6.2 → x9.2 the booking.
- Their epifauna share falls 0.21 → 0.05 (prey-side M2 0.11 vs booked 0.54).
- Fix97 had none of this, because its benthivores sat over the deep beds.
- budget_gate's field-spread version spreads only the water column and whiting as predators. Check run:
  `scripts/gate_benthivores.py` adds a benthivore entry to FIELD_BED_SHARES in memory and runs budget_gate
  unchanged (`tables/gate_benthivores_{gov,fix98,fix97}.txt`). The library and the repository code are untouched.
  - The benthivores' encounter-implied krill share at the start densities:
    - GOV adult spread: 0.033 vs booked 0.027 (x1.25; prey side 0.044 vs 0.035);
    - fix98's own year-3 spread: x1.32;
    - fix97's spread: x0.56.
  - Every benthivore pair is within the band. The pair and its depth table agree with the booked diet at the field
    spread, so the bed spread is not the cause.
- The cause is local depletion under concentrated, saturated fish (`scripts/local_enc.py`, frozen update-630
  policies; `tables/local_enc_fix98.txt`). Each column of the table adds one step to the one before:

  | Year | Krill share at grid-mean prey densities | Realised local densities | + the light lever | + per-cell saturation | Engine's demand | Realised |
  |---|---|---|---|---|---|---|
  | 1 | 0.023 | 0.028 | 0.027 | 0.044 | 0.044 | 0.039 |
  | 2 | 0.028 | 0.046 | 0.044 | 0.170 | 0.173 | 0.164 |
  | 3 | 0.051 | 0.064 | 0.059 | 0.207 | 0.211 | 0.217 |

  The per-cell saturation step weights each cell's encounter by λ_j / (1 + Σ h λ). It reproduces the engine's demand.
  - Half the benthivores sit on 5.1 / 3.8 / 3.5 % of the cells (years 1-3).
  - Where they sit, each prey's density over its grid mean (year 1 → 3) is:
    - exposed infauna 0.69 → 0.33;
    - epifauna 0.53 → 0.17;
    - krill 1.20 → 0.67 (open pool, k 20 /yr).
  - The bed pairs are far above saturation (h λ >> 1). Depleting them barely lowers the cell's intake but shifts
    its composition to krill, which keeps being re-supplied.
  - The diet gate's pooled Σ a B over cells weights rich cells by their encounter rather than by their capped
    intake, so it cannot see this.
- Reading: not a pair error. It is a trained aggregation on locally depleted beds, and the field has no measure of
  it here. Its size is 17 % of the krill excess in year 3 and rising.
- The benthivores' interference is 0 (section 223.28, no fish data), so crowding is penalised only through the prey.
- Fix97 shows the start of the same rise (krill x0.6 → x2.0 on the prey side), deeper and slower.
- Nothing is applied.

## What 270 did to its own targets

Fish-prey realisation (prey side, realised / booked):

| Pair or group | fix98 | fix97 |
|---|---|---|
| Planktivores, total M2 | 0.26-0.27 | 0.22-0.25 |
| Benthivores → planktivores | 0.55-0.99 | 0.20-0.55 |
| Whiting → planktivores | 0.27-0.35 | 0.19-0.26 |
| Water column → planktivores | 0.74-0.83 | 0.82-1.08 |
| Benthivores → whiting | 0.07-0.13 | 0.17-0.30 |
| Water column → benthivores (new booking 0.058) | 1.19-1.36 | — |

- The booked planktivore M2 is 0.36.
- Whiting drifts to 1.45 of its start by year 3 (fix97 1.14).

Clean gains:
- Seals now eat the booked diet: benthivores 0.64-0.71 of their tonnes (booked 0.64, fix97 0.38); planktivores 0.29-0.36.
- Epifauna and Nephrops no longer decline: year 3 at 1.01 and 0.99 of start (fix97 0.73 and 0.87).
- Infauna M2 is x1.3 (fix97 x2.3). Nephrops M2 is x2.6-6.3 (fix97 x6.4-6.9).

## Porpoises die out (new; viability 14/15)

- In every probe since about 25, the porpoises fall below one animal within the first year (survival 200-1 500
  ticks). Fix97's porpoises survived at 0.15-0.5 of start.
- Frozen update-630 policies: the loss is all starvation, mostly in June-July of the first year (3.0 → 0.6 t).
- The group's mean fill stays at 0.7-0.8, above u = 0.5, so cells starve while the group mean does not.
- Their ration (21 t/t/yr in year 1) is higher than fix97's surviving porpoises (18).
- No porpoise parameter changed in 270; their prey's floors did (1.0 against fish predators).
- The cause (spatial, policy, or the prey's response) is not separated here. The biomass weight in the web is
  negligible.
