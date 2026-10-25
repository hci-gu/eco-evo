# Evidence C - what the trawl CATCHES of the edible bottom community in the grid (t WW/km2/yr)

Scope: component (a) of the bottom-trawl impact on `benthic_community`: landings + dead discards of
Pandalus borealis and Nephrops norvegicus, landed crabs/lobster, and dead discards of non-target benthic
invertebrates, inside the 60 x 60 km grid (3600 km2), 2015-2023/24. The direct seabed kill of animals that
are NOT caught is component (b), in `evidence_D_trawl_kill.md`. Bottom line in `summary_trawl_removal.md`.

Tags: [F] full text / full data read, [A] abstract only, [S] search snippet, [D] my derivation.
All raw files are under `data/` (FDI zips and extracted CSV, Norwegian ERS zips and filtered CSVs, ICES
advice PDFs, literature PDFs, EMODnet GeoJSON). Scripts: `data/scripts/` (fdi_filter.py, ers_filter.py,
ers_analyse.py, ers_pra.py, ingrid_share.py, budget.py). Outputs: `data/derived/`.

## 1. EU landings by ICES rectangle (Sweden + Denmark + minor EU), STECF FDI spatial data 2013-2023

Source: STECF/JRC FDI spatial landings, EU27, data call 2024 (`2024_FDI_spatial_data.zip`,
https://jeodpp.jrc.ec.europa.eu/ftp/public/JRC-OpenData/FAD/fdi/), file
`spatial_landings_tableau_pts_<year>_EU27.csv` [F, data]. Landings in tonnes by year, quarter, metier,
species and ICES rectangle. Every row is given twice, once per 0.5-degree c-square half of the rectangle,
with the same value. The plain sum over rows matches the FDI Table A sub-region total: PRA 3.a.20 2022, spatial
sum 2856 t against Table A 2347 t + 44 confidential rows; NEP 3456 against 3128 + 179 C rows. So each
half-row is a half and the plain sum is the rectangle total [D]. No country field; Norway is not in it.

Mean EU landings, t/yr, 2015-2023 (2019-2023 in brackets) [F data, D means]:

| Rectangle | P. borealis (PRA) | Nephrops (NEP) | Edible crab (CRE) | Stone crab Lithodes (KCT) | Lobster (LBE) |
|---|---|---|---|---|---|
| 45G0 | 1137 (1185) | 332 (362) | 0.3 | 0.8 | 0.1 |
| 44G0 | 171 (163) | 1081 (1123) | 13.8 (18.1) | 0.2 | 0.1 |
| 45G1 | 27 | 282 | 33.6 | 0.2 | 6.4 |
| 44G1 | 1.5 | 678 | 92.0 | 0.1 | 7.1 |

- Year to year, PRA in 45G0 is 842-1548 t and NEP in 44G0 594-1523 t (2015-2023).
- Gear: in 45G0, PRA is OTB_CRU 771 t + OTT_CRU 362 t. NEP is OTB_MCD 255 t + OTT_MCD 41 t. In 44G0, NEP is
  OTB_MCD 1023 t (the Danish 90 mm mixed Nephrops trawl) and PRA is OTB_CRU 130 t [F data].
  So shrimp comes from the shrimp trawl with a 19 mm grid, and Nephrops from the mixed 90 mm trawl.
- **Boiled-weight correction.** ICES corrects Danish shrimp landings by x1.13 and Swedish by x1.11 for
  boiling, "Boiled landings from the Danish fleet continue to be corrected using the correction factor of
  1.13" [F, ICES pra.27.3a4a 2025].
  - FDI Table A 2022 for 3.a.20 + 4.a, DK + SE: 1376 + 972 + 488 + 84 = 2919 t. ICES has DK 2170 + SE 1173
    = 3343 t, ratio 1.15 [D]. FDI is therefore uncorrected.
  - I apply x1.12 (range 1.00-1.13) to FDI shrimp [D].

## 2. Norway: haul-by-haul logbooks (ERS DCA), 2011-2024

Source: Fiskeridirektoratet open data, "Elektronisk rapportering (ERS)" yearly zips, file
`...-fangstmelding-dca.csv`, haul start position + live weight (rundvekt) per species
(https://www.fiskeridir.no/Tall-og-analyse/AApne-data/elektronisk-rapportering-ers) [F, data]. Hauls are
placed in the grid by their start position, projected to SWEREF 99 TM with the repo's
`tools/make_fishing_maps.wgs84_to_sweref99tm`.

Norwegian ERS catch, t/yr, mean 2015-2024 [F data, D]:

| | P. borealis | Nephrops | Herring | Blue whiting | Saithe |
|---|---|---|---|---|---|
| inside grid | 5.2 (0-21.8 by year) | 0 | 125 | 18 | 3.0 |
| 45G0 | 32.9 (0.1-83) | 0 | 126 | 18 | 5.3 |
| 44G0 | 0.5 | 0.3 | 69 | 4.5 | 0.4 |

For 2011-2014, Norwegian ERS shrimp in 45G0 was 1.5 t/yr [F data].

**Coverage.**
- ERS covers vessels of 15 m and longer; the shortest PRA vessel found was 16.1 m.
- Norwegian ERS PRA south of 62 N was 2899 t (2018), 2646 t (2022) and 1711 t (2024). ICES Norwegian landings
  were 5492, 4900 and 3541 t, so ERS covers ~50 % [D].
- The rest is caught by small (<15 m) coastal trawlers. ERS puts the Norwegian shrimp effort in 46G0, 46F9
  and 44F5-F8, near the Norwegian coast and in the Norwegian Deep, not in 45G0 [F data].
- So Norwegian shrimp catch in the grid is ~5-8 t/yr. I assume <15 m vessels add up to x1.5 (central 7.8
  t/yr) and take 52 t/yr (2 x 33 t x 0.785) as high [D].

**Conflict with the EIA.**
- The Mareld EIA (`mkb.txt`) gives three-fleet landings in 45G0 of 9,892 t/yr (2010-2020), with Norway
  4,334.5 t/yr and northern shrimp 21.2 % (= 2,097 t/yr) [F, EIA].
- FDI EU shrimp in 45G0 was 1,122 t/yr in 2013-2020, which would leave ~975 t/yr for Norway [D]. Norwegian
  haul positions do not support that: 1.5-83 t/yr.
- The same mismatch holds for blue whiting: EIA 17.8 % = 1,761 t/yr, against FDI EU 28 t/yr and Norwegian
  ERS 18 t/yr.
- I conclude that the EIA's Norwegian rectangle numbers (appendix C11, not available) are not haul-located,
  perhaps allocated from Norwegian statistical areas. I do not use them for the central value [D].
- The EIA-based shrimp total would be 2,097 x 0.785 = 1,646 t/yr in the grid, about 1.5x my central [D]. That
  is an alternative high value, not within my envelope.

## 3. Share of each rectangle's catch that falls inside the grid (effort-weighted)

EMODnet bottom-otter-trawl effort (MW-h per c-square, `fishingbottomottertrawls` WFS,
`data/emodnet/fishingbottomottertrawls.json`) [F data]. Each c-square was sampled on a 20x20 lattice and
tested against the grid square in SWEREF 99 TM. The table gives the share of each rectangle's effort that
falls inside the grid [D]:

| Window | 45G0 | 44G0 | 45G1 | 44G1 |
|---|---|---|---|---|
| 2015-2018 | 0.762 | 0.531 | 0.154 | 0.067 |
| 2017-2020 | 0.771 | 0.559 | 0.167 | 0.083 |
| 2018-2021 | 0.766 | 0.587 | 0.173 | 0.091 |
| 2019-2022 | 0.810 | 0.812 | 0.071 | 0.020 |
| 2021-2024 | 0.818 | 0.835 | 0.094 | 0.026 |
| area share (grid km2 / rectangle km2) | 0.602 | 0.367 | 0.073 | 0.052 |

- **44G0 effort drops 86 % between the 2018-2021 and 2019-2022 windows** (8851 to 1254 MW-h), mostly south of
  57.8 N (2588 to 267 MW-h). FDI EU Nephrops landings in 44G0 did not drop (1005-1523 t/yr, 2018-2023).
  - Inference [D]: Danish VMS effort is largely missing from the recent EMODnet windows. The 44G0 shares from
    those windows (0.81-0.84) describe the remaining (mostly Swedish) effort.
  - Central for 44G0 is therefore 0.559, the mean of the three windows that include Danish effort.
- Central for 45G0 is 0.785 (mean of all five windows), for 45G1 0.132 and for 44G1 0.057. Low = area share
  (45G0 0.602, 44G0 0.367; 44G1 0.02, the lowest window). High = highest window [D].
- **Nephrops inside 45G0.** Swedish logbooks put Nephrops at 12.61 % of the shellfish catch in 45G0 but only
  0.84 % inside the Mareld wind-farm area, and shrimp at 86.7 % and 98.47 % (`mkb.txt` 4810-4813) [F, EIA].
  The EIA adds that a video survey showed "relativt låg täthet av bohålor" in Mareld [F, EIA]. So Nephrops in
  45G0 is fished mainly outside Mareld. For Nephrops in 45G0 I lower the low share to 0.30 [D].
- **Check against the Nephrops standing stock** in the sister note `evidence_A2_nephrops.md`: 4.5 g/m2 (2.4-8.1)
  grid mean. My central in-grid Nephrops removal (landings + dead discards 0.296 t/km2/yr) is 6.6 % of that
  per year. ICES FU 3-4 harvest rate is 5.4 % by number (2022-24) and the sister note gives ~7 % by weight [D].
  The effort-weighted Nephrops share is therefore consistent with the stock harvest rate.

## 4. Discards of the target species

- **P. borealis** (ICES pra.27.3a4a 2025, Table 8) [F]: estimated discards / landings, 2017-2024. DK 0.044,
  SE 0.129, all fleets 0.042 [D].
  - SLU observer trips in Swedish grid trawls 2011-2013 found 16.5-30.6 % of the shrimp catch discarded
    (Tables 13-15, MSC Swedish shrimp full assessment 2015) [F]. That was before the 2016+ estimates fell.
  - The grid fleet in 45G0 is mixed SE/DK (FDI 3.a.20 PRA 2022: DK 1376 t, SE 972 t). I use 0.08 (0.04-0.25)
    of landings and assume all discarded shrimp die. No survival study was found [D].
- **Nephrops** (ICES nep.fu.3-4 2025) [F]:
  - Discards / landings by weight 2020-2024: 5501 / 31183 = 0.176 [D].
  - "Discard survival rate (2026) 25 Percentage by number of the discards (ICES, 2016)"; Table 7: "75% dead /
    25% surviving".
  - Dead discards = 0.176 x 0.75 = 0.132 of landings. Range: 0.15-0.25 discard ratio, 75 % dead.

## 5. Other edible invertebrates

**Landed crabs and lobster** (CRE, KCT, LBE, CRA; FDI; all gears, pots included) [F data]. In the grid
(effort shares): 20 t/yr central (11-29), mostly edible crab from 44G0/44G1 [D].

**Discarded non-target invertebrates in the shrimp trawl** (SLU observer programme, Swedish trawl with grid,
Skagerrak). Source: MSC full assessment, Sweden Skagerrak/Kattegat/Norwegian Deep cold water prawn, DNV GL
2015, Tables 13-15 [F] (`data/lit/msc_se_pra.pdf`). Values are t raised to the Swedish fleet:
- 2011: P. borealis 451.2; white shrimp (Pasiphaea multidentata) 27.5; Nephrops 3.6; Pasiphaea spp. 1.75;
  "Other species" 10.1.
- 2012: P. borealis 668.1; glass/white shrimp 20.8; crimson shrimp (Pasiphaea tarda) 15.2; Nephrops 6.5;
  Atlantopandalus propinqvus 6.0; Norwegian shrimp (Pontophilus norvegicus) 5.8; pink shrimp (P. montagui)
  5.3; Pasiphaea spp. 1.7; other 14.7.
- 2013: P. borealis 652.4; Norway pout 26.2; glass/white 14.2; crimson 10.1; squat lobsters (Munida) 9.0;
  Nephrops 6.0; other 11.4.

As a share of the P. borealis catch [D]:
- Benthic non-target crustaceans (Munida, Atlantopandalus, Pontophilus, P. montagui; Nephrops excluded): 0 %
  (2011, none listed), 2.6 % (2012), 1.4 % (2013).
- "Other species" (fish and invertebrates mixed): 1.7-2.2 %.
- Pasiphaeid shrimps (bentho-pelagic, mesopelagic shrimps of the Norwegian Trench): 3.8-6.5 %. They are not
  bottom fauna in the model's sense. I report them as a separate, excluded line: ~5 % x PRA catch = ~0.02
  t/km2/yr [D].

Note the gap. WWF's comment in the same report: "Bycatch rates of invertebrates are unknown" [F]. Observers
record fish and commercial shellfish, not echinoderms or hermit crabs. Valentinsson et al. (ICES JMS 74:134,
2017) report that "few non-commercial invertebrate species have been regularly recorded by onboard observers"
in the Swedish Nephrops fisheries [S]. The 19 mm grid diverts large epifauna (starfish, large crabs, sea
pens) out of the trawl before the codend. Most of it is not "caught" but goes through the gear, which is
part of (b). I use 3 % (1.4-5 %) of the shrimp catch [D].

**Discarded non-target invertebrates in the Nephrops trawl.** No Skagerrak/Kattegat data were found. The
analogue is the Clyde Sea Nephrops fishery: 70 mm, no grid, 1998-1999. Bergmann et al. 2002, Fish. Res.
57:169 [F] (`data/lit/bergmann2002_discards.pdf`):
- "On an average, 9 kg of discards were produced per kilogram of Nephrops" (Table 3: north 10.4, south 6.2
  kg per kg Nephrops landed).
- Catches in the north "contained more invertebrates (60% cf. 15% in the south)".
- "In the north Clyde Sea area, invertebrates formed a high proportion (95%) of the discards. In comparison,
  trawls from the Farn Deeps and the Irish Sea yielded only 17 and 5% invertebrates".

Invertebrate discards are therefore ~0.5 kg per kg Nephrops in the south and ~10 in the north [D]. The
Skagerrak mixed fishery uses 90 mm with a SELTRA panel or a 70 mm grid. I take 0.3 t per t Nephrops landed
(0.1-1.5) [D, weak, analogue only].

**Survival of discarded invertebrates.**
- Bergmann & Moore 2001, ICES JMS 58:163 (decapods) [A/S]: "Short-term mortality ... ranged from 2-25%,
  with Pagurus bernhardus showing the lowest mortality"; longer-term survival of undamaged controls was
  72-83 %, of damaged animals ~30 %.
- Bergmann & Moore 2001, ICES JMS 58:531 (echinoderms) [A/S]: short-term mortality 0-31 %. A companion
  abstract reports 100 % post-trawling mortality of Ophiura ophiura at 14 days, and >90 % for Asterias with
  multiple arm loss [S].
- Skagerrak animals come up from 100-300 m. I take 50 % dead (25-100 %) [D].

**Scavenging.** Discarded and damaged animals feed scavengers. After a beam-trawl pass up to 13 % of the
annual macrobenthos secondary production becomes available (Groenewold & Fonds 2000, cited in Bergmann et al.
2002) [F, secondary]. In the model this is an energy subsidy to the benthivores, not a loss. I do not
credit it.

## 6. The grid budget (data/scripts/budget.py; output data/derived/catch_budget_output.txt) [D]

Formula, per rectangle r (45G0, 44G0, 45G1, 44G1), with share s_r from section 3:
- PRA_L = sum_r L_PRA,r x s_r x 1.12 + Norway
- PRA_D = PRA_L x 0.08
- NEP_L = sum_r L_NEP,r x s_r; NEP_dead discards = NEP_L x 0.176 x 0.75
- crabs = sum_r L_crab,r x s_r
- inv_shrimp = (PRA_L + PRA_D) x 0.03 x 0.5
- inv_neph = NEP_L x 0.3 x 0.5

Central arithmetic:
- PRA_L = (1137.2 x 0.785 + 170.7 x 0.559 + 26.7 x 0.132 + 1.5 x 0.057) x 1.12 + 7.8
  = (892.7 + 95.4 + 3.5 + 0.1) x 1.12 + 7.8 = 1118.5 t/yr
- NEP_L = 331.6 x 0.785 + 1081.4 x 0.559 + 281.5 x 0.132 + 678.3 x 0.057 = 260.3 + 604.5 + 37.2 + 38.7 = 940.6 t/yr

| Component, t/yr in grid | Low | Central | High | Central t/km2/yr | Central slope, t/km2/yr per MW-h/km2/yr |
|---|---|---|---|---|---|
| P. borealis landings (incl. Norway) | 754 | 1119 | 1270 | 0.311 | 0.103 |
| P. borealis discards (dead) | 30 | 90 | 317 | 0.025 | 0.008 |
| Nephrops landings | 531 | 941 | 1285 | 0.261 | 0.087 |
| Nephrops dead discards | 60 | 124 | 241 | 0.035 | 0.011 |
| Crabs + lobster landed | 11 | 20 | 29 | 0.006 | 0.002 |
| Non-target inverts, shrimp trawl (dead) | 3 | 18 | 79 | 0.005 | 0.002 |
| Non-target inverts, Nephrops trawl (dead) | 13 | 141 | 1927 | 0.039 | 0.013 |
| **Total** | **1402** | **2452** | **5148** | **0.681** | **0.226** |
| (excluded) pasiphaeid shrimps, shrimp trawl | | ~60 | | ~0.02 | |

- t/km2/yr = t/yr / 3600. Slope = t/km2/yr / 3.014, the grid mean of `maps/fishing_bottom_otter_trawls.npz`.
- Low and high set every factor at its end of the range together, so they are envelopes, not intervals.
  Most of the high comes from the Clyde-analogue Nephrops-trawl invertebrate discards. Without that line the
  high is 3221 t/yr = 0.89 t/km2/yr [D].

By taxon (central, t/km2/yr):
- P. borealis 0.336 (49 %)
- Nephrops 0.296 (43 %)
- other crustaceans + echinoderms + molluscs (dead discards) 0.044 (6 %)
- crabs/lobster landed 0.006 (1 %)

## 7. As a rate on the community biomass B

- Catch-based loss rate = C / B per year, with C = 0.68 t/km2/yr (0.39-1.43).
- With B = 100 t/km2: **0.0068 /yr (0.0039-0.0143)**. That is ~33 times smaller than today's d x SAR =
  0.226 /yr at the same mean effort [D].
- Per tick (6 h) at mean effort: 0.0068 / 1460 = 4.7e-6.
- Per unit effort: C/(B E) = 0.226 / B per MW-h/km2/yr per year = 2.26e-3 /yr per unit at B = 100, or
  1.55e-6 per tick per unit (biomass_factor at value 100: 1.55e-4) [D].

**The catch is concentrated in two sub-pools.**
- P. borealis: C_PRA / B_PRA = 0.336 / B_PRA. At a hypothetical B_PRA = 1 t/km2 this is 0.34 /yr. The stock
  F/FMSY was 1.1-1.8 in 2007-2023 (ICES Table 9) [F]; absolute F is not published.
- Nephrops: C_NEP / B_NEP = 0.296 / 4.5 = 0.066 /yr, with B_NEP from `evidence_A2_nephrops.md`.
- If the model's community B contains the shrimp and Nephrops as a fixed share, a lumped C/B charges the
  catch to all of B, while in reality it falls on a few %.

**Overlap with (b).**
- Hiddink's d comes from before/after sampling of the trawl path, with grabs, cores, sledges or trawls.
  Organisms that are caught and landed are gone from the path and count in d if the sampler catches them.
- P. borealis is hyperbenthic and migrates vertically, and grabs do not sample it. It is **outside d**.
  Fully additive.
- Nephrops sits in burrows and grabs undersample it. Mostly additive.
- Crabs, Munida and echinoderm discards are epifauna, and d is "primarily ... epi-fauna" (Pitcher et al.
  2022) [F, via depletion_and_recovery.md]. They are **largely inside d**.
- No-overlap (fully additive) bound: 0.68 t/km2/yr. Minimal-additive bound (shrimp + Nephrops only):
  0.63 t/km2/yr [D].

**Fleet basis.**
- The model's effort map and ICES SAR exclude Norwegian VMS (swept_area.md). Today's (b) therefore covers
  the EU fleets only.
- (a) includes Norway, but Norway's in-grid contribution is ~8 t/yr (0.7 % of the shrimp). (a) restricted
  to the VMS-visible fleets is 2444 t/yr = 0.679 t/km2/yr. Both components are effectively on the EU basis [D].
- Vessels under 12 m are not in VMS. In FDI 2015-2023 they land 0.1 % of the shrimp in 45G0 and 44G0, and
  5.6 % (45G0) and 4.6 % (44G0) of the Nephrops [F data, D]. So about 2 % of (a) comes from fleets that are
  not on the effort map.

**Separate finding: the effort map level.**
- EMODnet effort in 44G0 fell 86 % from the 2018-2021 window to the 2019-2022 window, 8851 to 1254 MW-h.
  South of 57.8 N it fell from 2588 to 267 MW-h.
- EU Nephrops landings in 44G0 did not fall: 1005-1523 t/yr in 2018-2023 (FDI).
- Inference [D]: Danish VMS is probably missing from the two recent windows.
- `maps/fishing_bottom_otter_trawls.npz` is scaled to the 2021-2024 grid mean. Its level (3.0 MW-h/km2/yr),
  and every library catchability calibrated per unit of that map, may therefore be low for the southern grid.

## Source status

| Source | Used for | Status |
|---|---|---|
| STECF FDI spatial landings EU27, data call 2024 (JRC FTP) | EU landings by rectangle, species, metier 2013-2023 | [F] data |
| STECF FDI 2025 Table A (JRC FTP) | boiled-weight check, 3.a.20 discards by country | [F] data |
| Fiskeridirektoratet ERS DCA 2011-2024 | Norwegian haul-located catch | [F] data |
| ICES Advice 2025 pra.27.3a4a (doi 10.17895/ices.advice.27202809; figshare file 55169207) | landings/discards by country 1970-2024, boiling factors, F/FMSY | [F] |
| ICES Advice 2025 nep.fu.3-4 (doi 10.17895/ices.advice.27202755) | landings, discards, 25 % survival, harvest rate | [F] |
| EMODnet HA WFS fishingbottomottertrawls | in-grid effort shares | [F] data |
| Mareld EIA (mkb.txt) | 45G0 landings by nation/species, shellfish split in Mareld | [F] |
| MSC full assessment Swedish cold water prawn, DNV GL 2015 (cert.msc.org) | SLU observer catch composition 2011-2013 | [F] |
| Bergmann et al. 2002 Fish. Res. 57:169 (epic.awi.de/10389) | Clyde Nephrops trawl invertebrate discards | [F] |
| Bergmann & Moore 2001a,b ICES JMS 58:163, 58:531 | invertebrate discard survival | [A]/[S] |
| Valentinsson et al. 2017 ICES JMS 74:134 | invertebrate observer gap | [S] |
| sister note evidence_A2_nephrops.md | Nephrops standing stock 4.5 g/m2 | read only |
