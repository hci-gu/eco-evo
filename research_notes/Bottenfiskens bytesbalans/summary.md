# Benthivores -> planktivores under shared handling: planktivore side or bed side? (2026-11-06)

Research only: nothing in the library, the manifest, the code or the tests was changed. Written by the research subagent, saved by the main agent. Tags: [F] field value or stated in a source, [D] derived here, [S] stock-assessment value, [J] judgement. No model output set any value.

Where things are (this folder): `evidence.md` (full trail, cited and tagged), `scripts/pout_vs_bed.py` (run with `python3 -I`, library sha 4ca160a1e9be), `tables/pout_vs_bed_stdout.txt`, `tables/budget_gate_stdout.txt`, `data/README.md` (nothing downloaded; inputs and commands).

## Verdict

Most of the gap sits on the planktivore side. No independent evidence shows the bed a's are wrong.

The gate needs the planktivore term (lambda_plk) 13.5x higher to reach the booked share, or 19-23x for the in-area field shares. That factor splits into three parts:

1. **x1.46: a bookkeeping error in the pooled term [D].**
   - `a x f_eff` multiplies the pool-average band share by the pool-average window.
   - But Norway pout is high on both and herring is low on both. Haddock's pout window (0.35) is also much wider than its herring window (0.075).
   - Rebuilding the term per species and per member with the same encounter code gives lambda_plk 0.0093 instead of 0.0064. Pout is 0.51 of that.
2. **x2-6: the pout biomass convention (user's call) [D/S].**
   - Box pout is 0.25 t/km2 from GOV at q = 1. That is a relative index, while the pool's herring is acoustic, i.e. absolute.
   - The Walker et al. 2017 catchability (q) correction gives 0.58 (Q1), 1.64 (mean of Q1 and Q3) and 2.70 (Q3).
   - Stock size per habitat area gives 1.4-2.8 (2016-24) and 0.45-0.9 (2024-25).
   - Mortality check: the booked benthivore take equals SMS's benthivore M2 on pout (0.29 /yr) at 1.4-1.7 t/km2. At 0.25 the benthivores alone would impose the whole stock's mortality (1.6-1.9).
   - The acoustic 0.2-0.3 is limited by the dead zone, because pout sit 4-10 m above the bed (Albert 1994; Aglen 1996).
3. **About x2.3 against booked (x3-4 against in-area) is left unexplained at pout 1.64.** It lies inside the combined uncertainty and cannot be pinned on either side:
   - the pout-term Monte Carlo reaches x3.2 at its 95th percentile;
   - the bed Monte Carlo can drop x2.3 at its 5th percentile;
   - a capture success of 0.5 for piscivores in general (Juanes et al. 2002) would add x1.7.

### Planktivore-side encounter factors

The factors checked were pout height above the bed, light, reaction distance, speed, band thickness, haddock capability and the window. None is wrong by a large multiple, and no range reaches more than x2 upward.

Two structural checks push the term down, not up:
- If cod search within 2 m of the bed, they reach only 0-6 % of the pout layer.
- Schooling lowers a gut-limited cod's encounter rate, since one pout is about 1.4x the 6 h ration of a 2 kg cod.

### Bed side

- **Lab rates support the bed a's order of magnitude, and nothing else bounds it.** Persson 2012 scaled to ~40 cm gives Crangon 0.032 and Palaemon 0.37 km2/t/tick, against the library's epifauna 0.15. No infauna rate has been measured.
- **Field intake data cannot settle it.** They cannot tell today's bed saturation (sum h lambda ~177) from a level 12x lower.
- **The exposure note no longer tests the bed.** Its diet "pass" (T2) used the planktivore term from before 223.26, which was about 10x larger.
- **Double counting is small.**
  - Between e, the refuge exposed share 0.2 and accessible 0.62, it lowers the bed by a few % at most (Amphiura tip mass, x0.9 on the refuge share).
  - Crab and Munida exposure counted on top of f raises the bed by up to ~10 %, which is the wrong direction.

## Parameters (current -> central (range))

| Key | Current | Central (range) | Tag | Basis |
|---|---|---|---|---|
| ben->plk `max_intake_rate`, pout unchanged | 0.0074 | 0.0107 (0.0079-0.0146) | D | per-species/member rebuild, x1.46 |
| pout B in the pool | 0.25 | 1.64 (0.58-2.70), user's call | D/S | Walker q, stock size per area, SMS mortality check |
| ben->plk a, if pout B changes | 0.0074 | 0.0248 (0.0155-0.0298) | D | same rebuild on the new pool; `stage_access` kept, J/B 0.38 -> 0.48 |
| ben->plk M2, if pout B changes | 0.145 | 0.483 / B_pool (0.103 at 4.71) | D | booked flow unchanged |
| bed a (0.036, 0.15) | - | unchanged; MC 5 % lower bound (0.016, 0.05) | D | no independent case for a cut |

## Predicted diet-gate shares

Shares are infauna / epifauna / Nephrops / planktivores.

| Case | Shares | Gate |
|---|---|---|
| current | 0.629 / 0.328 / 0.038 / 0.006 | OFF |
| a 0.0107, pout 0.25 | planktivores 0.009 | OFF |
| pout 0.58 | planktivores 0.014 | OFF |
| pout 1.64 | 0.612 / 0.319 / 0.037 / 0.032 | ok (ratio 0.43) |
| pout 2.70 | planktivores 0.049 | ok |
| pout 1.64, bed at MC 5 % | planktivores 0.072 | ok |

No pair leaves saturation:
- infauna 0.99 (0.97 with the bed at MC 5 %)
- epifauna 0.99 (0.97 with the bed at MC 5 %)
- Nephrops 0.92
- planktivores 0.60 -> 0.89-0.93

Derived growth rates use the ceilings, so no change in a moves them. Only re-booking pout does.

## Open points (user's call)

1. **Pout catchability convention and its knock-ons.** It moves the planktivore start biomass, every predator's M2 on the planktivores, their derived growth and regional K 4.8, and J/B, which moves every `stage_access` share on the pool.
2. **The x2-4 residual** stays unexplained.
3. **Visibility floor 0.25.** When planktivores hide in trained runs, the share can fall up to 4x further.
4. **Prey ranking per gut time.** (assim - sda) E/h is planktivores 20.9 > infauna 13.9 > epifauna 10.2 > Nephrops 8.8 MJ/t/tick.
   - A corrected planktivore term will therefore actually be eaten.
   - But a rate-maximising forager drops epifauna and Nephrops while the infauna's h lambda is above 2.7 / 1.7, and it is 75 now.
   - So the booked epifauna share 0.318 is not the expected learned diet. That is a separate question about the bed or the armour ceilings.
5. **Capture success.** p 0.3 -> 0.5 would apply to every fish pair from the section 220 encounter code, so it is not proposed for this pair alone.
