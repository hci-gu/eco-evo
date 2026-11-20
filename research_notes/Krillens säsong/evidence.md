# Krill availability by season: evidence trail (2026-11-20)

Main-agent fork. Tags: [F] full text, [A] abstract or search snippet, [S] held note / secondary, [D] derived here,
[J] judgement. Nothing in the library, code, tests, manifest, AGENTS.md or the resume was changed.

## S1. Field evidence per quarter (reused unless marked new)

**Stock level.**
- Library outside_season Q1-Q4 1.0 / 0.6 / 1.1 / 1.3 [J built on F inputs] ("Krill i Skagerrak" summary.md and
  evidence.md section 4; resume 245):
  - winter-night IKMT in the box high (Bohle & Moksness 1991) [F];
  - Norwegian Trench spring trawls low (IMR NSEC 2023/25) [S];
  - Kattegat individual DW rises 2.5-3x from March to November (Buchholz & Prado-Fiedler 1987 Table 1) [F];
  - reproduction April-October, peak July; maturity at 9-12 months; recruitment mainly from the I-group
    (Boysen & Buchholz 1984, abstract; new here) [A].
- No monthly stock series exists for the Skagerrak. Boysen & Buchholz 1984 is closed (AWI record, no PDF;
  data/boysen1984/page.html).
- Against the Q2 trough: whiting Skagerrak April 1991 krill 0.70 %BW/d, saithe 1991 Q2 pulse 24-49 %W ("Krillsidan").
  Bergstad et al. 2003 [S]: Skagerrak deep fish eat krill mostly in March and little in September, read in the
  note as availability, not stock.

**Presence over beds < 100 m (the dominant season term for bed feeders).**
- Relative to beds >= 250 m: Sep-Apr 0.43 (0.27-0.78), May-Aug 0.05 (0-0.1); annual 0.30 (0.18-0.55)
  ("Krill over grunda bottnar" rederive/summary.md; resume 262).
  - Winter: Bohle & Moksness 1991 east stations, night [D on F].
  - Summer: 0 of 14 daytime near-bed tows on < 100 m beds, June-July 2021-25, RV Dana HERAS, against 3 of 4 on
    100-170 m (P 0.005) [D on F]; only 2 tows inside the box; not aimed at krill [J].
- Stomachs agree with the winter side: Jan-Feb 1981 at 57-80 m, 21 of 23 cod and whiting samples held krill;
  whiting April 1991 mass share 0.48 [D]. No daytime sample on < 100 m beds October-April.
- The library holds the annual 0.30 in `outside_depth_table` (raw 0.38 at 50-99.9 m against 1.2-1.4 at >= 250 m).

**Day layer and night ascent.**
- Day 70-150 m: Oslofjord > 70 m; Kattegat 80-100 m at a 134 m site; 90-150 m at a 250 m site in August; below
  150 m in the Trench in May [F/S] ("Krill i Skagerrak" evidence.md section 3).
- July (Dana) 131-154 m over 157-166 m beds [D on F] (262).
- "Deeper in winter" (Bergstad et al. 2003, citing others) [S].
- Oslofjord 2021 echosounder work (UiO thesis; search snippet only, full text not fetched) [A]:
  - day range ~40-100 m "depending on season and weather";
  - shallowest by day at the highest light extinction;
  - deeper at night under moonlit winter skies than on cloudy spring nights;
  - higher on bright summer nights.
- Night: ascent to 10-50 m, 52-62 % of the stock stays below 65 m at a 134 m site (Tarling 1999) [F/D].
- No box measurement of the day layer by season. The direction and size over the box's bathymetry are unmeasured,
  so this term is not sized.

**Size and energy.**
- Year-class I DW 19-26 mg (31 Mar), 27-28 (7 May), 30-34 (7 Jul), 53-60 (30 Sep), 62-64 (26 Nov), 50-53
  (16 Feb); lipid peak 44-48 % DW in November [F].
- Energy 4.3 kJ/g WW in Q2-Q3 and ~6 in Q4-Q1 (Kattegat 20.5 / 26 kJ/g DW in July / November x DW/WW 0.20 / 0.27;
  Barents Sea 4.5 / 6.6 kJ/g WW, Nowicki 2023) [D on F]. The engine has one E, 5000.

**Susceptibility.**
- Dense winter day layer where the gadoids sit [J]. Krill voids under herring schools by day (Onsrud 2004,
  Christiansen 2024) [F/S]. No seasonal predation-susceptibility datum (e.g. spawning swarms) found in the notes.

## S2. Mapping onto the model's terms

| field term | model term | status |
|---|---|---|
| stock level by quarter | `outside_season` 1.0/0.6/1.1/1.3 | carried; shape [J]; the Q2 trough conflicts with Q2 stomach pulses (phase unsettled) |
| presence over < 100 m beds | `outside_depth_table` (annual 0.30) | carried as the annual mean only; the half-year season is MISSING (summer x6 too high, Sep-Apr x0.7 too low) |
| day-layer depth by season | band overlap f_m(z), depth_access_table per class | missing; unsized (no box data) |
| energy by season | `energy_content` 5000 | missing (one value) |
| ingestion Q10 on the krill a | the predator's `ingestion_q10` | carried; a fish term, not krill's; it raises the chain's summer encounter against the field's Q3 minimum |

## S3. Policy-free monthly multiplier on each predator's krill encounter [D]

`scripts/monthly_multipliers.py` -> `tables/monthly_multipliers_stdout.txt`. Linear regime: a x B_krill x E per
tonne of predator, relative to the library's annual mean. Placement: whiting and the water column at the GOV
bed-class shares in the box; the others on equal cells. Quarter means:

| predator (share of krill encounter on < 100 m beds) | lib | +shallow | +shallow+energy | +shallow+energy+flat stock |
|---|---|---|---|---|
| whiting (0.88) | 1.00 / 0.60 / 1.10 / 1.30 | 1.32 / 0.38 / 0.69 / 1.71 | 1.53 / 0.32 / 0.58 / 1.99 | 1.53 / 0.53 / 0.53 / 1.53 |
| planktivores (0.48) | same | 1.16 / 0.49 / 0.90 / 1.50 | 1.35 / 0.41 / 0.75 / 1.75 | 1.35 / 0.68 / 0.68 / 1.35 |
| water column (0.17) | same | 1.03 / 0.58 / 1.06 / 1.34 | 1.20 / 0.48 / 0.89 / 1.56 | 1.20 / 0.81 / 0.81 / 1.20 |
| mackerel (0.14) | same | 1.02 / 0.59 / 1.08 / 1.32 | 1.19 / 0.49 / 0.90 / 1.54 | 1.19 / 0.82 / 0.82 / 1.19 |
| benthivores (0.55) | same | 1.18 / 0.47 / 0.86 / 1.54 | 1.38 / 0.39 / 0.72 / 1.79 | 1.38 / 0.66 / 0.66 / 1.38 |

Whiting's month by month (+shallow): Jan-Mar 1.32, Apr 0.79, May-Jun 0.17, Jul-Aug 0.32, Sep 1.45, Oct-Dec 1.71.

## S4. Frozen test (sizing only) [D]

`scripts/seasonal_krill_test.py` wraps `fill_decomp_all.py` with process-local patches on top of library copy d
(planktivores x0.36, whiting and water column x0.16 on every pair; resume 294.9):
- shallow presence on the open pool's target, renormalised to the box total each month;
- krill E in `energy_gain_mat` and `handling_time_mat`;
- the stock season flattened.

Frozen fix105, seeds 0/1, one year. Per-seed stdout is in `tables/fill_decomp_d_*`; the seed-mean monthly fill is in
`tables/fill_summary_stdout.txt`. In the energy arms the decomposition's unc / feed rows read the krill share at the
reference E; the fill is exact.

| arm | whiting fill: year, min (month) | whiting realised | planktivores fill | water column fill |
|---|---|---|---|---|
| copy d | 0.94, 0.90 (Oct) | 0.381 | 0.88 | 0.96 |
| +shallow | 0.90, 0.79 (Sep) | 0.369 (Aug 0.25) | 0.88 | 0.96 |
| +shallow+energy | 0.90, 0.79 (Sep) | 0.373 | 0.89 | 0.96 |
| +shallow+energy+flat | 0.88, 0.69 (Sep) | 0.362 | 0.87 | 0.94 (winter 0.87-0.90) |
| field | min 0.30 (May-Jul), max 0.69-0.73 (Sep-Feb) | - | min 0.20 (Apr) | - |

- **Whiting** goes below need in July-August: realised 0.25 against ~0.37-0.42. Its fill falls from July to a minimum
  in September and refills by December. That is about 2 months after the field minimum and about 1/3 of its depth.
  The May-June deficit the multipliers predict does not appear. Fix105's whiting spring placement and its other
  prey (epifauna) carry May-June [J]. This is the frozen sated placement of 294.6: a hungry whiting would place itself
  differently.
- **Planktivores and the water column** hardly move. Their krill sits mostly on deeper beds, and the planktivores'
  winter is zooplankton-limited.
- **Diet gate and the booked M2.** Every arm keeps the annual means: the presence season averages to the library's
  0.30, E to 5000, and the flat stock season to 1. budget_gate is therefore unchanged by construction and was not
  re-run. The realised krill M2 under the arms was not computed.
