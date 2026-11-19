# Light below 65 m in the box: measured by gliders (2026-11-19)

Written by the research subagent, saved by the main agent. Field evidence only. No edits were made to the library, code, tests, manifest, resume, AGENTS.md or other notes, and no model value is recommended. Tags: [F] field/primary source; [D] derived here; [J] judgement; [A] abstract only; [S] secondary.

Files:
- `scripts/`:
  - `voto_download.py` and `bed_lookup.py` fetch the data.
  - `voto_k.py` computes the dark floor, slab K, and daytime light by month and bed class.
  - `near_bed.py` gives K by height above the bed.
  - `consistency.py` checks sensor against sensor and the within-cast near-bed effect.
  - `spectral.py` gives the band, lx per umol and the rod-band K.
  - `deep_light.py` and `bed_light.py` apply the prior note's convention with the measured K.
  - `night_floor.py` reads the night values.
- `tables/` holds each script's stdout and json.
- `data/README.md` lists the sources and licences: VOTO CC BY 4.0 with an attribution line, and EMODnet Bathymetry.
- `papers/LICENCES.md` covers the PDFs fetched by the literature pass.

## Verdict

- **Light below 65 m in the box is measured.**
  - Voice of the Ocean (VOTO) SeaExplorer gliders worked the Swedish side of the box and the Trench edge (10.0-11.6 E, 57.5-58.8 N): 28 delayed-mode missions, 2020-2026 [F].
  - 16 missions (Nov 2024 to Feb 2026) carry a Biospherical MPE-PAR. Its night floor is 1-2.5e-5 umol m-2 s-1, ~10^4 below SMHI's QSP-2000, and it reads daytime PAR to a median 108-148 m by month (p90 138-178 m).
  - 12 missions (Aug 2020 to Jan 2022) carry a SeaBird OCR-504 measuring Ed490 and Ed532.
  - Every month is covered; July only by the OCR-504.
- **K_deep for PAR is about 0.09, not 0.06.** Over 150-250 m beds, K stays near its 50-65 m value [D]:

| quantity (sun > 5 deg, 15 m slabs, n = slab fits) | median | p25-p75 |
|---|---|---|
| K_PAR mid-water (>= 100 m above bed), 65-170 m, n 9 589 | **0.092** | 0.076-0.109 |
| K_PAR 75-100 / 50-75 / 25-50 / 0-25 m above the bed | 0.098 / 0.102 / **0.115 / 0.139** | 0.080-0.160 |
| K_PAR over >= 250 m beds (Trench), 65-170 m | 0.082 | 0.071-0.098 |
| rod band (Govardovskii A1 template, 496 nm), 100-250 m | K_PAR - 0.002 to 0.003 | - |
| 490 nm | K_PAR - ~0.01 | - |

- **K_deep bracket [D/J]: 0.075-0.11 for PAR over 150-250 m beds, centre 0.09; rod band ~0.08-0.09.**
  - The lowest ~50 m above the bed adds a nepheloid layer at K 0.11-0.14.
  - The old judged dark bound (0.09) is now the measured centre.
  - 0.06 appears only in clear-cast-selected 490 nm data (section 2).
- **Daytime light at the bed** (median over sun > 0, all months), on the prior note's convention with the measured K and the near-bed layer. lx at 54 per umol, umol in brackets [D]:

| bed | median | bracket (p25-p75 K) | mid-water K only | old K_deep 0.06 |
|---|---|---|---|---|
| 150 m | 1.8e-4 lx [3.3e-6] | 2.6e-5 to 8.5e-4 lx | 1.4e-3 lx | 0.022 lx |
| 200 m | 1.7e-6 lx [3.1e-8] | 1e-7 to 1.8e-5 lx | 1.4e-5 lx | 1.1e-3 lx |
| 250 m | 1.7e-8 lx [3.1e-10] | 4e-10 to 3.9e-7 lx | 1.4e-7 lx | 5.4e-5 lx |

  - Mid-water light at 100 / 150 / 200 / 250 m with the 150-250 m bed-class K: 0.13 / 1.0e-3 / 9.6e-6 / 8.8e-8 lx (p25-p75: 0.07-0.23 / 2.4e-4 to 4.4e-3 / 1e-6 to 1e-4 / 4e-9 to 2e-6).
  - Over >= 250 m beds: 0.20 / 3.9e-3 / 9.7e-5 / 2.5e-6 lx.
  - The near-bed layer is measured only over beds <= ~195 m. At 200-250 m beds it is assumed [J], so lead with the mid-water column there.
- **A 150-250 m bed gets ~1e-4 to 1e-6 lx by day, not 0.006 lx** [D].
  - That is 30x to 1000x darker, and 2-4 orders below the judged Ke (0.02 lx).
  - It holds in every variant except the selection-biased 490 nm one (0.019 lx at 150 m; section 2).
- **Spectrum** [D]:
  - The deep light peaks at 480-500 nm, which is the gadoid rod band.
  - lx per umol: 55-65 at 65 m, 45-49 at 150 m, 40-44 at 200-250 m.
  - Rod-effective light per umol is 2.5-3x its 20 m value.
- **Confidence:**
  - High that K_PAR stays ~0.09-0.10 to 140-170 m over 150-250 m beds: two sensor types agree after the selection check, as do up- and down-casts and all seasons.
  - Medium on the near-bed layer and on anything below ~170 m (extrapolated) [J].
  - Medium on the absolute level: the convention and the gliders differ by ~2.5x (section 3).

## 1. Glider data and checks [F data, D analysis]

- **Processing:**
  - ERDDAP server-side 5 m bin means per cast.
  - EMODnet bed depth at each cast position.
  - Day is sun > 5 deg.
  - Dark is the night median per mission and bin (sun < -12 deg; < -6 deg for bins >= 50 m in June).
  - A bin is kept if it is >= 3x dark and >= 10x noise.
- **Instrument checks:**
  - Over 150-250 m beds at 65-140 m, K by MPE serial is 0.094-0.100 (s/n 2984), 0.092-0.098 (s/n 3201) and 0.074-0.088 (s/n 3199, Jan-Feb 2026 only).
  - Up- and down-casts agree to within 0.002.
  - Glider K at 35-50 m is 0.106 against the SMHI CTD's 0.109, and at 50-65 m 0.100 against 0.091.
  - K correlates with backscatter (Spearman +0.45 to +0.67) and chlorophyll (+0.2 to +0.3), and anticorrelates with dive depth (-0.4 to -0.5).
- **Selection.** Deep slabs are measurable only in bright, clear casts. In the 1 562 casts usable to 140 m, K rises from 0.078 (65-80 m) to 0.091 (125-140 m). The pooled 125-170 m medians (0.083-0.094) are biased low by this.
- **Season** (150-250 m beds, 65-155 m):
  - Apr-Sep 0.089-0.098, Oct-Mar 0.090-0.104.
  - Clearest: April and June, 0.078-0.088.
  - Murkiest: Nov-Dec, 0.12-0.14 at 50-80 m.
- **Near-bed layer:**
  - By class: K rises from 0.092 (>= 100 m above the bed) to 0.10 (50-100 m), 0.115 (25-50 m) and 0.139 (0-25 m), and reaches 0.16 over beds < 150 m.
  - Within the same cast, the lowest usable slab against the slab 45-60 m higher: +0.032 (25-50 m above the bed, n 401) and +0.060 (0-25 m, n 39), with backscatter 2.2x and 4.7x higher.
  - This fits Jerlov 1953, who found particles accumulating near the bottom on the Kattegat-Skagerrak section [F], and the Skagerrak nepheloid layers of Karageorgis et al. 2003 [A].

## 2. Spectral band, and the 490 nm values [F OCR data, D analysis]

- **OCR-504 raw values:**
  - All beds, 65-95 m: K490 0.066-0.068, K532 0.080-0.087, PAR 0.067-0.075.
  - 150-250 m beds: K490 0.073 / 0.072 / 0.065 at 65-80 / 80-95 / 95-110 m.
  - >= 250 m beds: K490 0.058-0.061 at 65-125 m.
- **These values are clear-cast-biased (consistency.py):**
  - The OCR passes a 490 nm bin only where PAR is >= 0.029 umol (Ed490 at 10x noise).
  - Imposing that floor on the MPE casts lowers their K at 65-110 m over 150-250 m beds from 0.094-0.096 to 0.082 / 0.075 / 0.072. Over >= 250 m beds it falls from 0.082-0.083 to 0.077 / 0.071 / 0.065.
- **The two sensor types agree where both see well.** At 20-65 m, matched by month and bed class, MPE minus OCR is +0.004 (median of 56 cells; range -0.023 to +0.059). The years and instruments do not differ.
- **The rod-band K is therefore the unbiased MPE K_PAR minus the measured PAR-490 offset (0.007-0.01): ~0.08-0.09.**
- **Spectral fit:**
  - K(lambda) = Kw(lambda) + X exp(-S(lambda - 490)), solved per slab from K490 and K532, with Pope & Fry water.
  - K is lowest at 490 nm; K500 is 0.001-0.003 higher and K550 0.02-0.03 higher.
  - The model's K_PAR is within 0.01 of the measured OCR PAR.
  - K at the rod peak (496 nm template) is 0.002-0.003 below K_PAR.
- **Monochromatic lx per umol:** 470 nm 16, 490 nm 35, 500 nm 53, 530 nm 133, 550 nm 148.

## 3. Absolute level against the convention [D]

- **The convention:** clear-sky lux x 0.5 cloud, the Secchi top 20 m and SMHI K to 65 m.
- **The gliders' own daytime medians as a share of it:**
  - over 150-250 m beds: 0.41 at 50 m, 0.39 at 80 m, 0.36 at 100 m, 0.31 at 120 m;
  - over >= 250 m beds: 0.24-0.34.
- **By month at 80-100 m:** Apr x1.0; Sep x1.9-2.4; Jun and Aug x0.4-0.6; Mar x0.3-0.4; Feb, May and Oct x0.1-0.25; Nov-Dec x0.01-0.05. Dec 2024 had K 0.14-0.15 at 20-65 m, against Secchi 11.5 m in the convention.
- **Glider-anchored bed light**, starting from the measured 8.4e-5 umol (4.5e-3 lx) at 120-125 m over 150-250 m beds:
  - 150 m: 1.1e-4 lx (5.8e-5 to 1.9e-4);
  - 200 m: 4.8e-7 lx (8e-8 to 2e-6);
  - 250 m: 4e-9 lx.
- **Measured daytime medians, all beds:**
  - 100 m: 1e-4 umol (Feb) to 4.2e-3 umol (Sep).
  - 140 m: ~8e-5 umol in Apr and Sep; below the floor in other months.

## 4. Literature and portals

- **Aarup, Holt & Hojerslev 1996** (CSR 16, I-III: Kd(lambda), beam c, 1990-92) [A]: paywalled. Their Skagerrak-Kattegat K(475) is 0.243 +- 0.115 for 0-20 m only (via Aas et al. 2013, Table 1) [S].
- **Hojerslev**, Skagerrak, Mar-Apr 1979: K(475) 0.126, 0-20 m [F via Aas 2013]. Skagerrak water is a coastal Jerlov type; Norwegian coastal water is type III.
- **Norheim, Klevjer & Aksnes 2016** (MEPS 551:45), Norwegian Sea Atlantic water, RAMSES radiometer: K486 0.032-0.044 at 90-231 m [F]. That water is ~2-3x clearer than the box's deep water.
- **Bozman et al. 2017** (J. Plankton Res. 39:280), Lurefjorden: measured K(500) 0.117 at 5-80 m. 0.102 for 80-400 m is derived from absorption [F/D]. This is the closest analogue.
- **Rostad et al. 2016**, Masfjorden: K(470-500) 0.192 / 0.092 / 0.060 at 0-25 / 25-50 / >50 m [F].
- **Sornes et al. 2007**, western Norwegian fjords: absorption 0.020-0.077 m-1 at 100-300 m [S].
- **Kampa 1971** (via Kaartvedt et al. 2019): K 0.024-0.06 below 200 m, oceanic [S].
- **BGC-Argo:** one Norwegian Sea float (WMO 6902547, OCR-504; Organelli et al. 2017) publishes Kd only for the first optical depth [F].
- **Portals:** the ICES CTD mirror at EMODnet has no PAR or transmissometer data. EMODnet Physics' PAR hits in the box are these same VOTO gliders. The IMR Torungen-Hirtshals section has T and S only.

## 5. Bioluminescence (brief)

- **Glider night readings** (MPE-PAR, 5 m bin means, 100-260 m) [D]:
  - The excess over the floor (p99 - p10) is ~1e-5 umol m-2 s-1 (mission max 5.6e-5).
  - That is <= ~5e-4 lx nominal, or ~3e-4 lx at 490 nm.
  - The glider stimulates the flashes, so this is an upper bound. It still exceeds daylight at a 200-250 m bed.
- **Kongsfjorden polar night:** bioluminescence was > 98 % of photons below 60 m [S, Cronin et al. 2016].
- No ambient (unstimulated) measurement exists for 150-250 m in temperate shelf water.

## Gaps

1. **No light above the sensor floor below ~150-180 m.** Light at 200-250 m beds rests on extrapolated K [J]. A photon-counting or RAMSES cast to the bed at a 200-250 m station would settle it.
2. **The near-bed layer is unmeasured over 200-250 m beds**, and its thickness and variation along the Trench slope are unknown.
3. **The ~2.5x gap** between the convention and the gliders' absolute level at 50-120 m is unexplained. Candidates: the cloud factor, the top-layer K, glider attitude, or unequal months.
4. **Gadoid rod sensitivity at 1e-4 to 1e-6 lx, and ambient bioluminescence at the bed**, decide what this light means for vision; neither is covered here.
5. **Aarup et al. 1996 (I-III)** was not read (paywalled).

## 6. What the measured K does to the water column's bed-class gradient (main agent) [D]

`scripts/class_terms_var*.py` re-run research_notes/"Sejens profil over djupa bottnar" `class_terms.py` (saithe
profile exp15 unchanged) with other (Ke, K_deep) pairs; outputs in `tables/class_terms_var*_stdout.txt`. The
script has ONE K below 65 m: it omits the near-bed layer (K 0.115-0.139, 0.16 over < 150 m beds; voto_k gives K
0.13-0.16 at 65-125 m over 100-150 m beds) and the ~2.5x level gap of section 3. Both darken the beds further,
the 100-150 m class included. Ke values from research_notes/"Torskfiskars syn i morker" (bracket 2e-4 to 0.07 lx).

G = a(< 150 m beds) / a(150-250 m beds); r01 = a(< 100 m) / a(100-150 m); eq = equal-cell a vs the library.

| Ke (lx) | K_deep | WC -> benthivores G | r01 | eq | WC -> whiting G | WC -> krill G |
|---|---|---|---|---|---|---|
| 0.02 (library) | 0.06 (library) | 2.65 | 1.22 | 1.00 | 2.70 | 7.96 |
| 0.02 | 0.075 | 3.44 | 1.37 | 0.89 | 3.65 | 11.6 |
| 0.02 | 0.09 | 3.76 | 1.59 | 0.81 | 4.17 | 18.2 |
| 0.02 | 0.11 | 3.59 | 1.98 | 0.73 | 4.13 | 30.8 |
| 0.07 | 0.09 | 3.34 | 1.98 | 0.70 | 3.80 | 28.2 |
| 2e-3 | 0.075 | 2.46 | 1.14 | 1.11 | 2.45 | 6.62 |
| 2e-3 | 0.09 | 3.32 | 1.22 | 1.01 | 3.39 | 8.51 |
| 2e-4 | 0.075 | 1.68 | 1.09 | 1.31 | 1.66 | 5.45 |
| 2e-4 | 0.09 | 2.26 | 1.11 | 1.20 | 2.23 | 5.90 |
| 2e-4 | 0.11 | 3.30 | 1.17 | 1.08 | 3.30 | 7.39 |
| no vision (R_nv only) | - | 1.00 | 1.02 | 0.31 | 1.00 | 5.16 |

- At the measured centre (0.09) the bed-prey gradient is 2.3-3.8 across the Ke bracket, against 2.65 today. Only
  the corner of the clear tail (0.075) AND the most sensitive eye (2e-4) gives a flatter one (1.7).
- With the darker water the advantage shifts onto the shallowest beds (r01 up to 2.0), where the trained water
  column already stands.
- The measured light moves the rule further from the stomachs (field juvenile gadoids at 150-250 m 0.40-0.93 vs
  model 0.04-0.13; the fall beyond 250 m contradicted by 1-2 orders), and Svalbard cod eat fish in the polar night.
  That points at detection without daylight (R_nv, bioluminescence), which no gadoid measurement constrains. The
  no-vision row is the far bound: G 1.0 but every visual pair's a x0.3.
