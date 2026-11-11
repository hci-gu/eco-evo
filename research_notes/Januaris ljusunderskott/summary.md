# January light deficit in the daylight lever - summary (2026-11-11)

Research subagent; the main agent saved this text. Nothing in the library, code, tests, manifest, AGENTS.md or
resume was changed; no training, no downloads, no model output read. Per-source numbers: `evidence.md` (model facts
M1-M8, light data E1-E8, derived D1-D10). Grades: [F] field data, [D] derived, [J] judgement, [M] model fact,
[S] secondary.

Files (run from `scripts/`; stdout and json in `tables/`):
- `winter_light.py` -> `winter_light_stdout.txt`, `winter_light.json`: monthly K(2-20), K(20-50) and PAR(2 m) /
  clear sky from the SMHI CTD PAR casts (par_casts.json of "Ljus under 50 m", read in place); the Secchi
  climatology against the derivation chain's Secchi table.
- `light_pairs.py` -> `light_pairs_stdout.txt`, `light_pairs.json`: the derived monthly a of every dark_ratio pair
  (rule A on the band overlap) under seven light readings against the engine's m(t). Execs `band_pairs_copy.py`
  (identical copy of "Overlapp i fiskparen"/scripts/band_pairs.py, which execs the section 269 chain in place);
  only the chain's `trans_month` and surface `E0` are replaced per reading.
- `flow_shift.py` -> `flow_shift_stdout.txt`, `flow_shift.json`: monthly flow shares, tonnes moved, predator intake
  and prey M2 per month, presence-weighted with the library arrays.

```
python3 -I winter_light.py <repo> > ../tables/winter_light_stdout.txt
python3 light_pairs.py <repo> <scratch dir> ../tables > ../tables/light_pairs_stdout.txt     # ~5 min
python3 -I flow_shift.py <repo> ../tables/light_pairs.json > ../tables/flow_shift_stdout.txt
```

## Question

"Sillens vinterprofil" (section 4) found the derived monthly a of the fish -> planktivore pairs at 0.66-0.73 (WC)
and 0.84-0.90 (whiting) of the engine's day-length multiplier m(t) in January (Nov-Mar 0.88-0.99). For every pair
with a dark_ratio: is the deficit real and how good is the January light; how large is it in consumption; should
the engine carry light intensity - (a) an intensity-based m, (b) a per-pair 12-month table, (c) leave it?

## Model (checked)

- m(t) = (rho + (1 - rho) F(t)) / annual mean, annual mean exactly 1 (M1); F a logistic in sun elevation around the
  pair's threshold (0 or -5 deg): day length and twilight only. `interactions.py` builds the table (ticks x DM x
  prey) once from `daylight.attack_multiplier`; the GPU copies it (M2). DMs observe only F at -6 deg, never
  intensity or the month (M3).
- 17 pairs carry a dark_ratio (M5): seven rule-A fish band pairs, two benthivore fish pairs, five krill pairs, four
  zooplankton pairs (incl. planktivores -> zooplankton, the largest booked flow). Whiting -> krill has none
  (stomachs); no mammal or seabird pair has one.
- The section 269 derivation's light (M7): surface 1.3e5 sin(h) lx x a constant cloud factor 0.5; K(2-20) = 1.56 /
  Secchi(month) from a 12-month Secchi table; below 20 m one K profile all year (0.109 / 0.091 / 0.06); gadoid eye
  Ke 0.02 lx, clupeid eye Ke 1 lx with a biting threshold 0.016 lx.
- Zooplankton pairs are not rule-A pairs; here they use the engine form made intensity-aware, rho + (1 - rho)
  g(E(t, z)), with g from the predator's eye over a judged layer [J] (planktivores at 15 m (8-25 m), mackerel 0-40
  m, whiting the bottom 10 m, WC on the saithe profile).
- Residual: each derived series is scaled on the engine's night rows so its night/day equals the library rho; the
  ratio derived / engine then isolates the intensity effect (factor 1.000 for rule-A pairs, 0.19-5.75 where
  night/day is from stomachs or the benthivore derivation).
- Reproduction (D1): R0 gives January 0.63 (WC) and 0.81 (whiting); the sibling note had 0.66 / 0.84 at its own
  night/day.

## Findings

### 1. The January light the derivation used is an n = 1 artefact [F]

| month | derivation K(2-20) (its Secchi, visits) | CTD cast K(2-20), median (n) | Slaggo-shaped Secchi K |
|---|---|---|---|
| Jan | 0.312 (5.0 m, A17 n 1) | 0.192 (10; 0.182 over beds >= 100 m, n 8) | 0.225 |
| Feb | 0.223 (7.0 m, n 1 + 2) | 0.193 (3) | 0.247 |
| Nov | 0.173 (9.0 m, n 1) | 0.182 (2) | 0.213 |
| Dec | 0.136 (11.5 m, A17 n 1 + A16 n 1) | no cast | 0.247 |
| Jul / Aug | 0.156 / 0.164 | 0.121 (4) / 0.156 (25) | 0.155 / 0.165 |
| Apr-Sep | - | 0.167 (42) | - |

- Ten daytime January casts from four winters (2018, 2019, 2024, 2025; IBTS Q1 and A16/A17) give K(2-20) 0.19
  (Jan-Feb p25-p75 0.17-0.23), at the real low winter sun. The derivation's January 0.31 comes from one Secchi
  visit, its December 0.136 from two single visits. The coastal Slaggo series (n 13-14 in Dec and Jan) has no
  Dec/Jan step (5.0 / 5.5 m).
- At depth (daytime median, D2): at 100 m in January the derivation has 0.022 lx, the casts' K gives 0.16-0.19 lx
  (~8.7x more); in December 0.43 lx vs 0.13-0.16 lx (~2.6x too bright). The "January" V (Dec 0.91, Jan 0.63, Feb
  0.90) comes from this pair of Secchi values, not the sun: December has the shorter, dimmer days.
- Surface: the manifest's monthly cloud transmission and Haurwitz's air-mass term instead of the constant 0.5
  lower the January residual by 0.02-0.03 (R1 -> R3); cloud transmission is the SMHI Nordkoster pyranometer ratio
  over Haurwitz (Jan 0.66, Nov 0.60, summer 0.73-0.78) [F]. The casts' own PAR(2 m)/clear sky (Jan 0.047, n 12;
  summer 0.21) is far below the pyranometer climatology (ship shadow at low sun, instantaneous overcast); used only
  as a dark bracket (R5) [J].
- Below 20 m in winter: K(20-50) has three winter casts (0.10-0.16; R4 uses 0.132 Dec-Feb). Below 65 m nothing is
  measured in any month (K_deep 0.06 [J] all year, E7). The Feb 2023 casts carried 0.33-0.50 mg m-3 chlorophyll at
  50-150 m (E8): some winters are darker at depth than any reading here.

Grade of the January light: K(2-20) [F, n 10], good; surface [F climatology]; 20-50 m [F, n 3], weak; below 50 m
[J], as in every month.

### 2. The deficit runs through the whole winter, not just January [D on F]

Readings: R0 the derivation as is; R1 Jan and Dec K(2-20) from the casts (0.192 / 0.191); R2 Secchi from the
Slaggo monthly shape scaled to the box median; R3 (central) R1 + Haurwitz x manifest cloud; R4 R3 + winter K(20-50)
0.132; R5 R3 + cast-dark winter surface (bracket); R6 R2 + Haurwitz x cloud.

Residual derived / engine (1 = the day-length lever already carries the swing); each cell R0/R1/R3/R6:

| pair (rho, thr) | Jan | Feb | Nov | Dec | Jul | Nov-Mar |
|---|---|---|---|---|---|---|
| WC -> planktivores (0.059, -5) | .63/.85/.83/.77 | .90/.89/.88/.84 | .93/.92/.89/.82 | .91/.81/.79/.69 | 1.10-1.11 | .87/.89/.87/.83 |
| WC -> benthivores (0.165, -5) | .65/.86/.84/.78 | .89/.89/.88/.83 | .93/.93/.89/.83 | .93/.83/.81/.71 | 1.11-1.13 | .88/.89/.88/.83 |
| WC -> whiting (0.137, -5) | .63/.86/.83/.77 | .89/.89/.87/.83 | .93/.92/.89/.82 | .93/.82/.80/.70 | 1.11-1.13 | .87/.89/.87/.83 |
| benthivores -> planktivores (0.141, 0) | .70/.91/.88/.83 | .90/.90/.89/.84 | .97/.96/.93/.86 | 1.00/.90/.87/.78 | 1.10-1.12 | .90/.92/.90/.86 |
| benthivores -> whiting (0.168, 0) | .68/.90/.88/.82 | .90/.89/.88/.84 | .96/.96/.92/.85 | 1.00/.89/.87/.77 | 1.11-1.13 | .90/.92/.90/.85 |
| whiting -> planktivores (0.212, -5) | .81/.92/.90/.88 | .95/.94/.94/.91 | .95/.95/.94/.90 | .94/.89/.88/.84 | 1.06 | .93/.94/.93/.91 |
| whiting -> benthivores (0.565, -5) | .87/.95/.94/.92 | .96/.95/.95/.93 | .98/.97/.96/.94 | .98/.94/.93/.90 | 1.06-1.07 | .95/.96/.95/.94 |
| WC -> krill (0.0245, 0) | .68/.94/.92/.87 | .94/.93/.92/.89 | .99/.98/.96/.90 | 1.02/.93/.91/.79 | 1.04-1.06 | .92/.95/.94/.89 |
| benthivores -> krill (0.036, 0) | .73/.96/.94/.90 | .96/.95/.94/.92 | 1.00/.99/.98/.93 | 1.03/.95/.94/.84 | 1.03-1.04 | .94/.97/.96/.92 |
| planktivores -> krill (0.57 stomach, 0) [hybrid] | .58/.83/.82/.75 | .82/.82/.81/.76 | .91/.90/.87/.78 | 1.00/.84/.82/.70 | 1.27-1.29 | .83/.85/.84/.78 |
| mackerel -> planktivores (0.16 stomach, -5) | .92/1.01/1.00/.98 | 1.00/1.00/1.00/.99 | 1.03/1.03/1.02/1.00 | 1.08/1.03/1.03/.98 | 1.01 | ~1.00 |
| planktivores -> zooplankton (0.03, 0) [J layer] | .95/.99/.99/.98 | .98 | 1.00/1.00/1.00/.99 | 1.02/1.00/1.00/.98 | 1.00 | .99-1.00 |
| mackerel / WC / whiting -> zooplankton [J layer] | .90-.98 | .96-.99 | .95-1.00 | .93-1.00 | 1.01-1.04 | .96-.99 |

Not a light residual (left out): WC -> mackerel (0.923; its swing is spurdog's Jun-Sep surface profile, 0.35-0.6
Oct-May, ~2.0 Jun-Sep); mackerel -> krill (0.42 stomach; member seasons make it erratic, R3 Jan 1.08, Mar 0.76, Dec
1.19); whiting -> krill (no dark_ratio; the engine is flat; the rule would give Dec-Jan 0.81-0.91, Jul 1.22-1.25,
but the library chose no light dependence on stomach evidence).

- The V is gone: every reading with field K in January removes it.
- What remains is symmetric: the gadoid band pairs sit at 0.80-0.92 in Nov-Feb and 1.08-1.15 in Jul-Aug (mean 1).
  Rule A adds intensity on top of day length, so the derived a swings ~10-15 % more over the year than m(t), under
  every reading [D on F for K(2-20); J below 50 m], largest in December, on top of an m(t) that already cuts Dec-Jan
  to 0.62-0.71 of the annual mean.
- Within a reading the five gadoid band pairs agree within 0.05 per month (WC pairs deepest); the whiting pairs show
  about half the effect; the zooplankton pairs about none (shallow, light saturates).
- The library's annual a's do not move: the equal-time annual a under R1-R6 is within -1.7 % to +1.5 % of R0 for
  every pair except mackerel -> krill (0.86-1.00; its clupeid eye sits near its threshold, D6).

### 3. Size in consumption [D, linear regime, presence-weighted with the library arrays]

Dec + Jan share of each pair's annual flow, engine -> derived, under R3 (tonnes: R1-R6 range); booked flow = M2 x
inference biomass:

| pair | booked t/yr | Dec+Jan share | t/yr moved out of Dec-Jan |
|---|---|---|---|
| planktivores -> krill [hybrid] | 37 278 | 13.8 -> 11.2 % | -986 (-926 to -1516) |
| benthivores -> planktivores | 1 739 | 8.5 -> 7.4 % | -19 (-16 to -32) |
| WC -> planktivores | 2 958 | 2.5 -> 1.9 % | -17 (-15 to -24) |
| planktivores -> zooplankton | 56 420 | 7.3 -> 7.3 % | -18 (-8 to -122) |
| whiting -> planktivores | 926 | 9.6 -> 8.5 % | -10 (-9 to -14) |
| WC -> krill | 2 142 | 3.8 -> 3.4 % | -8 (-7 to -17) |
| other pairs (whiting -> benthivores, benthivores -> whiting, WC -> benthivores / whiting, benthivores -> krill, mackerel pairs) | 64-3 150 | | <= 10 each |

Over the year a band pair re-times at most 3.3 % of its flow; planktivores -> krill 6.4 %. The WC's presence peaks
Jul-Sep, so the derived timing would raise the WC pairs' presence-weighted annual flow x1.03-1.05.

Dec-Jan intake per tonne of predator present, derived / engine (all booked pairs; pairs without light dependence
flat):

| predator | light pairs' share of booked diet | Dec-Jan, R3 | R1-R6 |
|---|---|---|---|
| water column | 0.98 | 0.86 | 0.77-0.88 |
| planktivores | 0.88 | 0.91 | 0.86-0.92 (krill pair) |
| whiting | 0.27 | 0.985 | 0.98-0.99 |
| benthivores | 0.11 | 0.993 | 0.988-0.994 |
| mackerel | 1.00 | 1.06 | 0.94-1.07 (box presence 0.1-0.2) |

- Whiting and benthivores lose 0.6-2 % of Dec-Jan intake: nothing for their winter budgets. The water column loses
  12-23 % per tonne but takes only 2-4 % of its annual flows in Dec-Jan (presence 0.16-0.52). The planktivores'
  8-14 % comes almost entirely from planktivores -> krill, the least certain row (the stomach night/day imposed on a
  rule series whose own night/day is 0.10; clupeid Ke 1 lx [J]).
- Prey side, Dec-Jan predation derived / engine: krill 0.85 (0.74-0.86), planktivores 0.93 (0.89-0.94), whiting
  0.91 (0.86-0.93), benthivores 0.96 (0.94-0.96), zooplankton 1.00. For the open pools negligible (D9): krill's
  exchange k 20 /yr vs M2 ~2.5 /yr gives ~2 % on the winter stock; for the planktivores a 7 % cut in their ~0.36 /yr
  M2 for two months is ~0.4 % of the stock.

## Mechanism options (no code change)

(a) A light-intensity multiplier per pair, m(t) = (rho + (1 - rho) g(E(t, z_p))) / mean, E from
`daylight.clear_sky_par` x monthly cloud attenuated to a pair depth z_p. Needs a fish-eye schedule function in
daylight.py, new pair keys (depth, eye Ke), a deep-water light climate in the manifest (K by layer and month; today
only the producers' Kd 0.29), loader and FG-editor fields; the GPU copies the table; the observation can stay F (an
intensity channel would invalidate checkpoints). Against it: the band pairs' overlap sits in the eye's knee over
deep beds and saturates over shallow ones, so one depth per pair misstates both (a per-bed-class table would be
needed, i.e. the derivation itself); and the deep K below 50-65 m it needs is unmeasured in every month (E7).

(b) A per-pair 12-month factor s(month) with annual mean 1, like outside_season, on the light term: m(t) = (rho +
(1 - rho) F(t) s(month)) / mean (the rule's night term is R_nv in every season; second order for rho <= 0.2, not
for whiting -> benthivores 0.565, planktivores -> krill 0.57, mackerel -> krill 0.42). The s values must be
recomputed as day-term factors from the derivation (not done here) and the product renormalised to mean 1
(`daylight.monthly_value` already interpolates 12 values circularly). Code: a pair key, ~10 lines where
`interactions.py` fills `light_mult_table`, an FG-editor field; the GPU needs nothing; observation unchanged;
checkpoint shapes valid but payoffs move (retrain). Values from the derivation on field light data, nothing from
model output; behaviour stays learned (a payoff, not an action); the DMs do not observe the month (as for
outside_season and temperature). One shared table would do for the five gadoid band pairs (within 0.05).

(c) Leave it: a timing error only - at most 3.3 % of a band pair's annual flow, at most 2 % of the residents' Dec-Jan
intake, 2-4 % of the WC's annual flows; the library's annual a's are unaffected.

## Gaps

- Deep light: no PAR below ~65 m in any month; winter K(20-50) n 3; the Feb 2023 deep chlorophyll (E8) suggests some
  winters are darker at depth than R4.
- December: no cast; rests on the Nov-Feb cast median and the Slaggo shape.
- Surface: the casts' surface PAR is not a climatology; the surface relies on the Nordkoster monthly ratio and
  Haurwitz.
- Zooplankton layers are judgements (planktivores -> zooplankton R0 January 0.83 at 25 m, 0.95-0.99 at 8-15 m;
  under R3 0.98-0.99 at every depth).
- planktivores -> krill: the residual depends on imposing the stomach night/day on a strongly light-limited clupeid
  day term; its Jul +27 % / Dec-Jan -18 % swing may reflect the clupeid Ke 1 lx [J] more than the herring.
- Linear regime only: Holling saturation and shared handling damp every consumption number.
- Time resolution: the section 269 derivation grid has 2 days per month.

## Verdict

1. January's deficit as a January feature is not real [F, 10 casts in four winters]: it comes from a one-visit
   January Secchi (5 m) next to a one-visit December Secchi (11.5 m). With the casts' K the January residual of the
   water-column pairs goes 0.63 -> 0.83-0.86, December 0.91 -> 0.79-0.83.
2. A winter-wide intensity deficit is real under every reading [D on F for 0-20 m; J below 50 m]: the gadoid band
   pairs sit at 0.80-0.92 of m(t) in Nov-Feb, mirrored by 1.08-1.15 in Jul-Aug - an amplitude mismatch (rule A sees
   intensity, m(t) only day length), largest in December.
3. In consumption it is small: residents lose at most 2 % of Dec-Jan intake per tonne; the mostly absent WC 12-23 %
   per tonne in months holding 2-4 % of its year; ~1 000 t/yr move out of Dec-Jan on planktivores -> krill, the
   least certain row; prey stocks move at most 2 %.
4. The evidence supports (c). (b), in the light-term form with recomputed day-term factors, is the ready path if a
   later planktivore winter-budget or krill winter-M2 question needs the timing. (a) is not supported (no deep
   winter light climatology; a single pair depth the band derivation contradicts).

## Consequences (flagged, not proposed)

- The section 269 chain's Secchi table (one_rule_pairs.py SECCHI) has n = 1 entries for January and December; when
  the chain is next run, January and December K(2-20) should carry the cast median (~0.19). The library's annual
  a's are unaffected (D6).
- The derived timing would raise the WC pairs' presence-weighted annual flow by 3-5 % (summer surplus x summer
  presence); the engine's equal-time annual-mean convention does not see it.

## User decisions

- 2026-11-11: option (c), leave the engine as it is; the section 269 chain's
  January / December Secchi entries (n = 1) are flagged for the next run of the
  chain.
