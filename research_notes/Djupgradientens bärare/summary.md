# What carries the water column's bed-depth gradient: krill and planktivore pairs (2026-11-22)

Research subagent with two literature subagents; the main agent saved this text.
- Nothing in the library, code, tests, manifest, AGENTS.md or resume was changed. Nothing was committed.
- Every variant ran on a library COPY in the session scratchpad.
- No value was set from or fitted to model output.
- The variants are bracketed alternatives, each tied to a named field source. They are not a recommendation.

Tags: [F] full text, [A] abstract, [S] secondary or analogue, [D] derived here, [J] judgement. Sources E1-E19 and model
facts M1-M5 are in `evidence.md`.

## Files

- Literature trails: `q1_evidence.md`, `q2_evidence.md`, `q3_evidence.md`; rows in `tables/q1_krill_bed_rows.csv` and
  `tables/q2_rows.csv`.
- `data/`: quote excerpts only. The PDFs were deleted for their licences (`data/README.md`).
- Run: `bash run_all.sh <repo> <scratch>` from `scripts/` (~6 min, CPU). It writes:
  - `tables/variants_stdout.txt` and `.json`: class tables and the reproduction gate;
  - `tables/gate/` and `tables/intake/`: one file per variant and scope (wc, all, wcshape);
  - `tables/summary_stdout.txt`: the combined table, with OFF sets diffed against the library;
  - `tables/per_bed_stdout.txt`.
- Scripts:
  - `variants.py` (with `variant_defs.py`): band_pairs.py logic at K_deep 0.09.
  - `make_lib.py`: writes the library copy.
  - `gate_copy.py`: budget_gate on the copy (the gate_pinned.py pattern).
  - `intake_by_class.py`: the caller's wc_class_decomp.py plus `library_path`.
  - `tabulate.py`, `per_bed.py`.
  - From the literature agents: `q1_gov_krill_by_bed.py`, `saithe_krill_by_hour.py`.

## Question

The trained water column stands over beds < 150 m, while GOV saithe are 0.78 over 150-250 m beds. Policy-free intake
per tonne is 2.0-2.8x higher over < 150 m. Which field evidence bears on the depth tables of WC -> krill and WC ->
planktivores? Three questions:
- the krill day layer's height above 150-250 m beds;
- saithe time in the krill layer;
- gadoid detection in near-darkness.

## Verdict

1. **The gradient is mostly vision, not geometry (M3, M4) [D].**
   - At K_deep 0.09 the krill G of 18 = geometry 5.2 x vision 3.5. The planktivore G of 6.9 = geometry 1.4 x
     vision 5.0.
   - With the krill near the bed over 190-245 m beds and no vision, the krill G is 1.97.
   - Over a 160 m bed the library already puts krill on the bed (gap 4 m). Daytime a there is still 1/6.5 of a
     100 m bed. Bed light is ~7e-4 lx (median of sun-up ticks over the year) against Ke 0.02 lx.
2. **Q1 (new data): the library layer survives [F/D, low-moderate].**
   - Over 150-175 m beds, in-region data put the layer near the bed. The library has done this since section 262.
   - Over 200-250 m beds the layer is 50-120 m above the bed. Sources: Kaartvedt et al. 2002, Oslofjord 200 m bed;
     Degnbol 1985, 250 m bed; 0 of 4 in-box near-bed day tows. That is the library's height or higher.
   - A bed-referenced layer 20-30 m up over 190-250 m beds comes only from analogues (Scotian basins; Cochrane et al.
     1994). In the box it is supported only in kind, by GOV Q3 presence: 3/16 hauls at 200-250 m, non-closing net.
   - Season: no consistent pattern.
   - Night: most of the layer goes to 10-50 m, and a deep remnant stays.
3. **Q2 (no new data on time in the layer) [F/D].**
   - No saithe tag gives height above the bed, and no diel stomach series with krill exists.
   - Saithe krill FO is day 0.52, night 0.47. Krill stay 24-42 h in the stomach, so FO cannot time feeding.
   - Bound: rapid vertical runs take 0.15 of the time (Iceland DSTs), not linked to prey.
   - Analogue: Barents cod rise 10-15 m off the bed with rising krill.
4. **Q3 (no measurement on krill) [F].**
   - The nearest is juvenile walleye pollock detecting Artemia in the dark at <= 1 cm (lateral line). The fish
     length in the source is unsettled, so this neither supports nor rules out R_nv 0.1 m.
   - The vision rows therefore use the fish-prey bracket of "Torskfiskars syn i morker", which is unresolved by a
     factor of 30 in Ke. It is the largest lever here.

## Variant table

Columns:
- Class a's are pre-296 library units for beds < 100 / 100-150 / 150-250 / >= 250 m. The loader renormalises each to
  an equal-cell mean of 1.
- G = a(< 150, cell-weighted) / a(150-250).
- R0 = a(< 100) / a(150-250). This is the caller's x27.
- Intake G = policy-free I(< 150) / I(150-250), Jan / Apr / Jul / Oct.
- Gate = field-spread budget_gate, WC encounter / booked share, krill / planktivores.

Scope:
- Scope wc writes all four WC pairs, a = live a x the variant's ratio.
- The wcshape runs keep the live a and change only the shape. They differ by <= 0.15 in intake G
  (`tables/summary_stdout.txt`).

| variant | source | WC -> krill classes | G (R0) | WC -> planktivores classes | G | intake G | gate |
|---|---|---|---|---|---|---|---|
| library | 262/269/280 | 0.0302 / 0.0129 / 0.0011 / 6.8e-05 | 18.2 (27.5) | 0.038 / 0.0167 / 0.00368 / 0.00284 | 6.9 | 2.76 / 2.35 / 2.02 / 2.40 | 1.02 / 0.87 |
| krill far | Kaartvedt et al. 2002 Stn D (gap 87 m at 200 m) + Degnbol 1985 [F/D] | 0.0302 / 0.0129 / 0.00102 / 6.8e-05 | 19.5 (29.5) | library | 6.9 | 2.81 / 2.37 / 2.05 / 2.45 | 1.00 / 0.88 |
| krill near | Cochrane et al. 1994, Herman et al. 1993 (Scotian analogue; gap 7 m, 190-245 m beds) [F/D] | 0.0302 / 0.0129 / 0.00139 / 6.8e-05 | 14.4 (21.8) | library | 6.9 | 2.61 / 2.27 / 1.93 / 2.26 | 1.07 / 0.85 |
| krill near, Jul-Sep only | GOV Q3 presence (DATRAS) [D] x the Scotian geometry | 0.0302 / 0.0129 / 0.00118 / 6.8e-05 | 16.9 (25.6) | library | 6.9 | 2.72 / 2.32 / 1.99 / 2.36 | 1.03 / 0.87 |
| saithe 0.15 of the day in the krill layer | Armannsson & Jonsson 2012 runs, upper bound [F/J] | 0.0346 / 0.0149 / 0.0024 / 0.00095 | 9.6 (14.4) | 0.0391 / 0.0162 / 0.00341 / 0.00259 | 7.5 | 2.32 / 2.11 / 1.74 / 1.97 | 1.31 / 0.72 |
| Ke 2e-3 lx | Anthony 1981 / Aksnes & Utne 1997, F 1 [D] | 0.0322 / 0.0151 / 0.0026 / 9.14e-05 | 8.5 (12.4) | 0.04 / 0.0224 / 0.00609 / 0.00289 | 4.9 | 1.91 / 1.73 / 1.51 / 1.70 | 1.10 / 0.86 |
| Ke 0.06 lx | same, F 30 [D] | 0.0283 / 0.0108 / 0.000669 / 5.77e-05 | 26.9 (42.3) | 0.036 / 0.0132 / 0.00327 / 0.00282 | 6.9 | 3.12 / 2.57 / 2.25 / 2.75 | 0.95 / 0.91 |
| Ke 2.4e-4 lx | scotopic branch, same data [D/J] | 0.0331 / 0.0158 / 0.00382 / 0.000118 | 6.0 (8.7) | 0.0409 / 0.0249 / 0.0101 / 0.00298 | 3.1 | 1.52 / 1.41 / 1.29 / 1.40 | 1.07 / 0.91 |
| R_nv fish 0.05 m | burbot (Michels 2020), length-scaled [D/J] | library | 18.2 | 0.0369 / 0.0151 / 0.00117 / 0.000348 | 20.6 | 4.48 / 4.08 / 3.02 / 3.54 | 1.25 / 0.71 |
| R_nv fish 0.5 m | same, prey-size-scaled [D/J] | library | 18.2 | 0.0401 / 0.0197 / 0.00828 / 0.00739 | 3.4 | 1.88 / 1.62 / 1.50 / 1.73 | 0.76 / 1.06 |
| krill near + Ke 2e-3 | combination | 0.0322 / 0.0151 / 0.00325 / 9.14e-05 | 6.8 (9.9) | as Ke 2e-3 | 4.9 | 1.81 / 1.67 / 1.46 / 1.61 | 1.18 / 0.82 |

**OFF pairs.**
- No row takes a WC pair OFF, in scopes wc and wcshape alike.
- In scope "all" (whiting, planktivore and mackerel krill pairs rewritten too), the Ke 2e-3, Ke 2.4e-4 and
  combination rows put whiting -> benthivores newly OFF at the field spread. Its ratio is ~0.32, against 0.35 ok in
  the library. Whiting's krill a rises while its fish pairs are not rewritten, so this is a scope artefact as much as
  a result.

**Benthivore and whiting tables.** They move with the vision and saithe rows. G for benthivores / whiting:

| row | benthivores | whiting |
|---|---|---|
| library | 3.8 | 4.2 |
| Ke 2e-3 | 3.3 | 3.4 |
| Ke 2.4e-4 | 2.3 | 2.3 |
| R_nv 0.5 | 1.9 | 2.1 |
| R_nv 0.05 | 17 | 19 |

**Not in the table.**
- `sens_Rnv_krill0.25`: [J], no source. Krill G 11.3, intake G 1.81-2.37.
- `sens_dark` and `sens_dark_near`: decomposition only.

**Reading [J].**
- The krill-position rows move intake G by -0.15 to +0.05.
- Only the vision rows move it materially: 1.3-1.9 at the low-Ke or large-R_nv edge, 2.2-4.5 at the other.
- The saithe bound lands at 1.7-2.3.
- No field-tied row reaches intake G of about 1.

## Main caveats

1. **Vision bracket.**
   - It is the fish-prey bracket, from cod contrast thresholds with an unknown geometry factor F.
   - No krill-specific Ke exists.
   - Ke is the gadoid eye, so a consistent change moves the whiting, benthivore and planktivore pairs too. Only the
     WC pairs, plus the krill pairs in scope "all", are in the copies.
2. **Level against shape.**
   - The 270 capture p x schooling F, the ling decomposition and the WC stage_access are not re-derived at the
     variant's R_eff.
   - Scope wc scales the live a by the variant's ratio and keeps the 296 x0.16. Scope wcshape keeps the live level.
   - The fish pairs keep their live dark_ratio, so the monthly intake G of the vision rows is approximate.
3. **Light.** One light profile is used for all beds; there is no bed-dependent K.
4. **Season.** The engine holds one constant depth table per pair. The Q3-only row is averaged into an annual table.
   A seasonal table needs engine work (flagged, not designed).
5. **Thin krill-position evidence.**
   - About 10 in-region tows, one Oslofjord station and one hypoxic basin.
   - The fjords are darker than the open Skagerrak.
   - GOV presence comes from a non-closing net.
   - Species are mixed.
6. **Policy-free intake.** Intake is computed with fixed attack, T = 1, start densities and no depletion. The gate is
   budget_gate at 16 x 16.

## Gaps

1. A gadoid Ke or reactive distance on krill-sized prey at 1e-4 to 1e-2 lx. This decides the tables more than
   anything else.
2. In-box daytime 120/200 kHz profiles over 150-250 m beds, resolving the bottom 1-5 m, in winter and summer.
3. Saithe height above the bed by time of day: bed-referenced DSTs or bottom-mounted echosounders.
4. Diel saithe stomachs with digestion stages over 150-250 m beds.

## Notes

- `q1_evidence.md` verdict item 6 compares the field with the pre-262 krill_day_layer. The library's 262 layer is
  124-157 m over 157-166 m beds and fits.
- `data/README.md`, Q2 section: `q1_kaartvedt1996_meps136/paper.txt` was deleted. Its passages are in `quotes.txt`
  in the same folder.
