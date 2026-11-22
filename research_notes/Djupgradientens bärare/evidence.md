# Evidence: what carries the water column's bed-depth gradient (2026-11-22)

The full literature trails, with quotes, are in three files:
- `q1_evidence.md`: krill day position against the bed.
- `q2_evidence.md`: saithe and hake in the krill layer.
- `q3_evidence.md`: dark detection of krill.

Rows are in `tables/q1_krill_bed_rows.csv` and `tables/q2_rows.csv`. This file lists the items the variants rest on and
the model facts. Tags: [F] full text, [A] abstract, [S] secondary or analogue, [D] derived here, [J] judgement.
Downloads are quote excerpts only; every PDF was deleted for its licence (`data/README.md`).

## Model facts (M)

**M1. Reproduction gate.** `scripts/variants.py` execs research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py
up to its section 0, as class_terms.py does. It runs at K_deep 0.09 (section 280, as Djupljus i synparen's
s1_band.py). Its "library" variant gives, at g3 (`tables/variants_stdout.txt`, "gate PASSED"):

| pair | classes | a | night/day |
|---|---|---|---|
| WC -> krill | 0.0302 / 0.0129 / 0.0011 / 6.8e-05 | 0.0107 | 0.0257 |
| WC -> planktivores | 0.038 / 0.0167 / 0.00368 / 0.00284 | 0.0148 (pre-270) | - |
| WC -> benthivores | 0.0676 / 0.0426 / 0.0141 / 0.0127 | - | - |
| WC -> whiting | 0.0588 / 0.0379 / 0.0112 / 0.00966 | - | - |

- The whiting, planktivore and mackerel krill tables reproduce too.
- Benthivores -> krill is derived by benth_profile.py, not by this run. It is not gated and never written to a copy.
- A library copy written from the library variant is identical to the live file (yaml compare), and budget_gate on it
  prints the plain run's text.
- The section 270 factor (capture p x schooling F) and the 296 x0.16 sit on the live a. They do not change the class
  shape: the loader renormalises each table to an equal-cell mean of 1.

**M2. The library krill day layer (section 262, `krill_day_layer` with ANCHOR_B/Z).** The gap is from the layer's
bottom edge to the bed:

| bed | layer | gap |
|---|---|---|
| 150 m | 80-110 m | 40 m |
| 160 m | 124-156 m | 4 m |
| 170 m | 123-157 m | 13 m |
| 200 m | 112-152 m | 47.5 m |
| 250 m | 95-145 m | 105 m |

Over 160-300 m beds the layer bottom stays at 145-157 m depth, so it is light-set, not bed-set.

**M3. Per-bed daytime a of WC -> krill relative to the 100-150 m class mean** (`scripts/per_bed.py`,
`tables/per_bed_stdout.txt`; not presence-weighted). The light is the median over all sun-up ticks of the year at
K_deep 0.09.

| bed | daytime a | layer gap | light at the bed | note |
|---|---|---|---|---|
| 100 m | 3.6 | - | 0.16 lx | |
| 160 m | 0.55 | 4 m | 7e-4 lx | krill on the bed; the drop from 100 m is vision only |
| 200 m | 0.03 | - | - | |
| 250 m | 0.0008 | - | - | |

Night a falls with the bed: 0.06-0.56 over beds < 100 m, 0.014-0.055 over 100-150 m, and about 0.01 from ~175 m.

**M4. Decomposition at K_deep 0.09.** `sens_dark` puts every visual member at R_nv; `sens_dark_near` does the same
with the krill_near_scotian layer.

- Krill G 18.2 = geometry 5.2 x vision 3.5. `sens_dark_near` gives the geometry left once the krill sit near the bed
  over 190-245 m beds (`tables/summary_stdout.txt`).
- Planktivores G 6.9 = geometry 1.38 x vision 5.0. Benthivores and whiting: geometry 1.00.
- K_deep 0.09 made vision the larger term for krill too. At 0.06 the "Sejens profil" note had geometry 5.2 x
  vision 1.6.

**M5. Intake by class, policy-free.** Library intake G is 2.76 / 2.35 / 2.02 / 2.40 (Jan / Apr / Jul / Oct).
`scripts/intake_by_class.py` is the caller's wc_class_decomp.py with one change, `library_path`.

## Q1. Krill day position over 150-250 m beds

| item | site, bed | day position | tag | source |
|---|---|---|---|---|
| E1 | in box, 166 m; 157 m; 167 m | catches at 131-154 and 138-148 m; 0 kg at 92-110 m over 167 m the same day | [D, cited] | ICES AT DB, RV Dana HERAS 2025 ("Krill över grunda bottnar"/distribution 1.1) |
| E2 | Skagerrak, ~204 m (median) | 0 of 4 day tows with footrope 0-3 m above the bed; 0 kg on the bed at 207-213 m over 213 m | [D, cited] | same; "Sejens profil" heras_deep_windows |
| E3 | box, 150-200 / 200-250 m | euphausiids in GOV day hauls Q3 9/34 and 3/16; Q1 1/4 and 0/16; 100-150 m Q1 16/45, Q3 4/35 | [D] | NS-IBTS DATRAS, `scripts/q1_gov_krill_by_bed.py` (see note) |
| E4 | Oslofjord, Drøbak Stn D, 200 m | positive day tows 100-158 m in May, 80-113 m in Aug, 100-107 m in Nov; 0 at 150 m in Nov; "remained below 70 m at day" | [F/D] | Kaartvedt, Larsen, Hjelmseth & Onsrud 2002, MEPS 228:193, doi:10.3354/meps228193, Table 3 (checked here in the local text) |
| E5 | Skagerrak NNE of Skagen, 250 m | 90-150 m, Aug 1984 | [F] | Degnbol & Munch-Petersen 1985, ICES CM 1985/H:46 |
| E6 | Masfjorden, ~195 m | 130-150 m on a Jan morning (45-65 m above the bed) | [F/D] analogue | Balino & Aksnes 1993, MEPS 102:35, doi:10.3354/meps102035 |
| E7 | Scotian Shelf, LaHave Basin, 192 m mooring | "maximum of the daytime euphausiid layer (220 m) ... maximizes at a comparable height (20 to 30 m) above bottom. This indicates the daytime layer depth is more strongly influenced by the proximity of the bottom"; layer "above the 7 m gap" | [F] analogue | Cochrane, Sameoto & Belliveau 1994, MEPS 107:55, doi:10.3354/meps107055 (quote checked here in "Krillens fläckighet"/data/cochrane1994_meps107) |
| E8 | Emerald/LaHave, ~270 m | M. norvegica "below 160 m depth during daytime" | [F] analogue | Herman, Cochrane & Sameoto 1993, MEPS 94:165, doi:10.3354/meps094165 |
| E9 | Bunnefjorden, 150-154 m (hypoxic) | day layer bottom ~24 m above the bed in Mar, ~0-10 m in Aug, bed avoided in Oct (O2) | [F] | Moksnes 2021 MSc, Univ. Oslo; Brun 2007 MSc |
| E10 | Clyde, 150-165 m | summer 50-85 m above the bed; winter "evenly spread between 60 and 150 m" to the net limit 15 m above the bed | [F] analogue | Lass et al. 2001, MEPS 214:177, doi:10.3354/meps214177; Tarling 2003 |
| E11 | Barents bank, 160-170 m | "krill were found close to the bottom" by day, among cod | [F] analogue | Skaret et al. 2020, MEPS 638:13, doi:10.3354/meps13254 |

Note on E3: the GOV does not close, so these are presence counts and an upper bound on near-bed presence. The
denominator is the strata that recorded euphausiids at least once.

Correction to `q1_evidence.md` verdict item 6. Its "function" (81-116 m over 157-166 m beds) is the pre-262
krill_day_layer. The library's section 262 layer is 124-157 m there (M2), so the in-box July tows already fit the
library.

## Q2. Saithe and hake in the krill layer

| item | content | tag | source |
|---|---|---|---|
| E12 | No tag gives saithe height above the bed. Rapid vertical runs take 0.15 of the time (0.002-0.33), with no day/night split | [F, via note] | Armannsson & Jonsson 2012 ("Krillintag och jaktsätt") |
| E13 | Saithe krill FO by sun phase: day 0.52 (463), twilight 0.39 (70), night 0.47 (38). Krill stays 24-42 h in the stomach, so FO cannot time feeding | [D] | ICES stomach DB, `scripts/saithe_krill_by_hour.py` |
| E14 | Barents cod on a 125-150 m bed rose "10-15 m from the bottom" when the krill rose. On 175-240 m beds euphausiids were 84 % of cod diet weight, with FO flat over 24 h (midnight sun) | [F] analogue | Dolgov, Yaragina, Ajiad, Mehl & Bogstad 1992, hdl:11250/107818 |
| E15 | "fish invaded the krill layer at night" (Oslofjord) | [A] | Onsrud & Kaartvedt 1998, MEPS 171:209, doi:10.3354/meps171209 |
| E16 | Pacific hake and krill "separated vertically by tens of meters during daylight hours". Box hake eat almost no krill: FO 0.024, n 126 | [A]; [D] | Mackas et al. 1997, doi:10.1139/f97-113; ICES DB |

## Q3. Dark detection (vision variants)

| item | content | tag | source |
|---|---|---|---|
| E17 | Cod contrast thresholds (Anthony 1981, J Fish Biol 19:87) refit (Aksnes & Utne 1997, Sarsia 82:137). Ke in downwelling lx 2e-3 at F 1 to 6e-2 at F 30. Scotopic branch 1.5-2.4e-4 lx, whose plateau R^2 is ~1/3 of the bright one | [D] | research_notes/"Torskfiskars syn i morker" E1 rows 1-4 |
| E18 | R_nv on fish prey: burbot in the dark, mean 5.6 cm, max 12 cm. Scaled to a 40-70 cm gadoid: 0.05-0.3 m by predator length, ~0.1-0.5 m by prey size | [F/D] | Michels 2020 MS thesis (TAFS doi:10.1002/tafs.10293), via "Torskfiskars syn i morker" E2 |
| E19 | Krill: no measured dark detection distance. Juvenile walleye pollock detected Artemia in darkness "only when they were very close (<= 1 cm)", by the lateral line | [F] analogue | Ryer & Olla 1999, MEPS 181:41, doi:10.3354/meps181041; `data/q3_ryer1999/` |

On E19 [J]: the fish length is unsettled. The kept quote reads "20 to 50 mm TL"; the Q3 agent read 42-114 mm. Scaled by
length to a 50 cm saithe, 1 cm gives 0.05-0.25 m. That neither supports nor rules out the judged 0.1 m on krill, so no
R_nv-on-krill variant is in the table; `sens_Rnv_krill0.25` is a sensitivity only. Krill sounds (Kühn et al. 2022) and
stimulated bioluminescence (Widder et al. 1992) exist as cues, but no fish use of them is measured.

## Not written

`summary.md` was NOT written: the harness refuses report files from a subagent. Its text was handed to the caller.
Where `make_lib.py` says "flagged in summary.md", the flag is in that hand-back text.
