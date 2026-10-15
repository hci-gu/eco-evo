# Swept area ratio (SAR) of bottom otter trawling in the Mareld / Poseidon Nord grid (57.81-58.31 N, 10.07-11.15 E), and a conversion from the EMODnet effort map (MW-h/km2/yr) to SAR

Scope note: about 30 tool calls, roughly 20 of them web searches, page fetches or downloads of public files, plus local computation. Full texts read:
- Eigaard et al. 2016 (ICES JMS) and its 2016 corrigendum, as Ifremer PDFs.
- Eigaard et al. 2017 (ICES JMS, Ifremer PDF).
- The ICES Technical Service to OSPAR 2018 (sr.2018.14).
- The OSPAR QSR 2023 BH3 assessment PDF.
- Hiddink et al. 2017 (PNAS, PMC HTML).
- Rijnsdorp et al. 2020 (ICES JMS, WUR PDF).
- NAFO SCR Doc. 16/056 (Skagerrak Pandalus fleet).
- The WGSFD BENTHIS gear-width table and the ICES VMS data-call R scripts, both on GitHub.

Sciberras et al. 2018 rests on its abstract only. The ICES VMS web API (`taf.ices.dk/vms/api`) returned HTTP 500, and its SAR endpoints need a login.

Local files used:
- `/home/simon/eco-evo/maps/fishing_bottom_otter_trawls.npz`
- `/home/simon/eco-evo/tools/make_fishing_maps.py`. Its rasteriser and SWEREF 99 TM projection were reused unchanged.
- `/home/simon/eco-evo/coordinates.txt`
- The sister note `research_notes/Fiskeridödlighet i Skagerrak idag/local_fishery.md`.

I also downloaded the EMODnet Human Activities WFS layers `fishingsurface`, `fishingsubsurface`, `fishingbottomseines`, `fishingbeamtrawls` and `fishingdredges`, for 57.6-58.5 N, 9.8-11.4 E, into the session scratchpad (`emodnet/`), next to the cached `fishingbottomottertrawls.json`. Numbers computed from these are marked (Local).

Grid membership: "grid c-squares" are the 216 c-squares (0.05 x 0.05 deg, about 16.4 km2) whose centre lies inside the grid. That is the same convention as the sister note. Rasterised 1-km maps use `make_fishing_maps.rasterise`. Units throughout: SAR in yr^-1, effort density in MW-h km^-2 yr^-1, and k = SAR per (MW-h/km2), which is km2 swept per MW-h.

## Q1. Published SAR data for Skagerrak / 3a and for the grid

### Takeaway
The ICES WGSFD SAR product is openly available per c-square through EMODnet Human Activities, as two WFS layers:
- `fishingsurface`, attribute `srfc_sr`, titled "Average Surface Swept Area Ratio".
- `fishingsubsurface`, attribute `sbsrfc_sr`, titled "Average Subsurface Swept Area Ratio".

Both use the same c-squares and the same five 4-year windows as the `mw_fshn` effort layers. EMODnet holds no other SAR layers, and no layer per métier (no OT_CRU / OT_DMF / OT_MIX split). The surface and subsurface layers are the sum over all mobile bottom-contacting gears: otter trawls, seines, beam trawls and dredges.

In the 2021-2024 window the grid c-squares average:
- surface SAR 2.74 yr^-1 and subsurface SAR 0.50 yr^-1;
- c-square maxima of 9.8 (surface) and 2.3 (subsurface).

The surface maximum is an anchor-seine (SDN) c-square on the southern edge. The highest otter-trawl c-squares reach surface SAR 7.5-7.6 and subsurface 2.2-2.3. The older 2015-2021 windows are about twice as high (surface about 6, subsurface about 1.3), but that step coincides with a likely data-coverage change (see Inferences).

ICES "surface" SAR is the total gear footprint. Subsurface SAR (penetration of 2 cm or more) is a subset of it, not an addition to it.

### Cited Findings
- **EMODnet HA WFS capabilities.** The fishing layers are `fishingbeamtrawls`, `fishingbottomottertrawls`, `fishingbottomseines`, `fishingdredges`, `fishingpelagic`, `fishingstaticgears`, `fishingsubsurface` and `fishingsurface`.
  - The `fishingsurface` and `fishingsubsurface` abstracts list: "ii) the average annual subsurface (top) and surface (bottom) mobile bottom contacting fishing gear (i.e. bottom otter trawls, bottom seines, dredges, beam trawls) disturbance by ecoregion ... expressed as average swept-area ratios (SAR)".
  - On data and coverage: "VMS/logbook data are anonymized and aggregated in a 0.05×0.05 degree grid prior to submission to ICES" and "The last data loaded into the database, from the ICES 2025 Fishing Overview update or revision (where available), report the 2021-2024 averages."
  - Effort is "mW fishing hours" and is "only shown for vessels >12 m having vessel monitoring systems (VMS)".
  - (Fetched full text, GetCapabilities XML) - [EMODnet HA WFS GetCapabilities](https://ows.emodnet-humanactivities.eu/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetCapabilities)
- **ICES SAR definition.** "ICES (2016a) defines the swept area as the cumulative area contacted by a fishing gear within a grid cell over one year. The swept area ratio (SAR, also defined as fishing intensity) is the swept area divided by the surface area of the grid cell."
  - Surface is "< 2 cm penetration depth of the gear components", and subsurface is "≥2 cm penetration depth".
  - SAR was computed "by métier ... following the approach of Eigaard et al. (2016)" for the 14 BENTHIS métiers.
  - "SDN_DMF Danish seine for plaice and cod (note: there is no subsurface component for this gear)".
  - Caveat: "at shorter, yearly time-scales the proportion of the seafloor trawled will be overestimated because trawling is randomly distributed at small spatial scales".
  - (Fetched full text) - [ICES Technical Service sr.2018.14 to OSPAR](https://odims.ospar.org/public/submissions/bottom_f_inten/bottom_f_intensur/2017-01/001/ospar_bottom_f_intensur_2017_01_001-other-OSPAR_request_on_the_production_of_spatial_data_layers_of_fishing_intensity_pressure.pdf)
- **Surface includes subsurface (OSPAR QSR 2023 BH3).** "Subsurface abrasion is a component of surface abrasion and therefore, only occurs in the presence of surface abrasion, due to specific gear components that penetrate ≥ 2 cm below the seafloor surface (e.g., otter trawls doors)."
  - Swept area "is a multiplication of the width of the gear in contact with the seabed by the average vessel speed and the time fished per unit area (c-square) per year".
  - Coverage caveat: "VMS data from Portugal, Iceland and Norway were not included in assessments as the submitted data did not pass ICES quality checks".
  - OSPAR intensity classes: "High (4) >1,00- ≤3,00; Very High (5) > 3,00", and "areas that were fished more than three times per year did not show any further levels of degradation (van Loon et al., 2018)".
  - (Fetched full text) - [OSPAR QSR 2023 BH3a](https://oap.ospar.org/media/filer_public/0d/50/0d50ce44-f6ad-4a19-ac2a-dde5e1dffc87/p00948_bh3_benthic_habitats_damage_fisheries_qsr2023.pdf)
- **Eigaard et al. 2017, the European footprint for 2010-2012, vessels > 15 m.** "In addition to the total width of the gear used to estimate the surface impact, the subsurface impact was estimated based on information on the dimensions of the gear components that penetrate into the seabed".
  - Table 2, Skagerrak-Kattegat (IIIa): shallow zone (0-200 m) seabed 41 x 10^3 km2, swept 147 x 10^3 km2, "Intensity 3.63" yr^-1, coverage 100%. Deep zone (201-1000 m): 14 x 10^3 km2, swept 36 x 10^3 km2, intensity 2.56 yr^-1.
  - "Highest intensities were recorded in the Skagerrak-Kattegat, Iberian Portuguese area, Tyrrhenian Sea and ...". For IIIa, "the high subsurface footprint is mainly a result of high fishing intensities with bottom trawls targeting crustaceans and mixed fish".
  - (Fetched full text) - [Eigaard et al. 2017, ICES JMS 74:847](https://archimer.ifremer.fr/doc/00362/47277/47238.pdf)
- **Grid SAR from the EMODnet/ICES layers (Local, computed).** Means over the 216 grid c-squares:

  | Window | OT effort MW-h/km2/yr | Seine effort | Surface SAR | Subsurface SAR |
  |---|---|---|---|---|
  | 2015-2018 | 5.01 | 0.055 | 5.95 | 1.26 |
  | 2017-2020 | 5.86 | 0.033 | 6.24 | 1.37 |
  | 2018-2021 | 6.13 | 0.028 | 6.21 | 1.38 |
  | 2019-2022 | 2.26 | 0.038 | 2.00 | 0.36 |
  | 2021-2024 | 3.07 | 0.078 | 2.74 | 0.50 |

  - Beam-trawl and dredge effort in the grid is zero in all windows. The only dredge record in the box is one 2015-2018 entry outside the grid.
  - 2021-2024 c-square maxima in the grid: surface 9.82, at 57.825 N 10.875 E, a c-square with 20.7 MW-h of seine effort and 1.5 MW-h of otter-trawl effort. Subsurface maximum 2.31.
  - The top otter-trawl c-squares are 58.125 N 10.625 E (surface 7.63, subsurface 2.30, 232 MW-h) and 58.175 N 10.175-10.225 E (7.5-7.6, 2.2-2.3, 212-218 MW-h).
- **ICES SAR rasterised onto the 1-km grid (Local, computed).** I used the same "today" recipe as the effort map: the mean of window-normalised patterns, scaled to the 2021-2024 grid mean.
  - Surface: grid mean 2.97, 99th percentile 9.5, max 19.6. The max is at row 59, col 20. It comes from seine c-squares just south of the grid (57.775 N, 10.425-10.475 E: surface SAR 30-37 and only 0.14-0.17 subsurface in 2021-2024, with 79-99 MW-h of seine effort).
  - Subsurface: grid mean 0.49, max 1.41. Pattern averaging smooths the 2.3 c-square peak.
  - The cell-wise correlation with the otter-trawl effort map is 0.96 for subsurface SAR but only 0.51 for surface SAR, because of seines.

### Inferences
- The SAR the project needs exists openly at c-square resolution. It is all-gear, and in this grid it is effectively "otter trawl + Danish/anchor seine": no beam trawl, no dredge.
- **The 2018-2021 to 2019-2022 step is probably a coverage artefact, not a real halving of trawling.** Grid effort falls 63% between two windows that share three years. The sister note found the same 60% drop for the whole Skagerrak and judged it a data-submission change.
  - The SAR-per-MW-h ratio also changes (Q3), while the subsurface/surface ratio stays at 0.21-0.23. A metier mix that stays the same while swept area per MW-h falls points to a change in which vessels are in the data (more large, low-k vessels), not in gear.
  - I found no ICES statement explaining the drop. Norway's VMS is excluded from the ICES abrasion products (OSPAR QSR 2023), so Norwegian shrimp trawlers are probably missing from every window.
  - **Treat the absolute SAR as a lower bound**, as the sister note does for effort.

## Q2. Gear footprint: converting MW-h to swept area (Eigaard et al. 2016)

### Takeaway
For otter trawls, ICES uses the BENTHIS door-spread models. Swept area per hour = door spread (a·kW^b) x towing speed. Per MW-h:

k = a·kW^(b-1)·v[kn]·1.852 (km2 per MW-h, i.e. SAR per MW-h/km2).

Because b ≈ 0.3-0.5, k falls with vessel power. For OT_CRU (Nephrops/shrimp, 2.5 kn) it ranges from 1.65 at 150 kW to 1.14 at 300 kW, 1.06 at the BENTHIS average of 345 kW, and 0.87 at 500 kW. OT_MIX gives 1.59 / 0.98 / 0.68 at 150 / 300 / 500 kW. These k values are **total footprint = surface SAR**.

The subsurface share is 32.1% for OT_CRU, 7.8% for OT_DMF and 14.7% for OT_MIX in the WGSFD table. That gives a subsurface k of about 0.53 / 0.37 / 0.28 for OT_CRU at 150 / 300 / 500 kW.

### Cited Findings
- **Eigaard et al. 2016, Table 4** ("Parameter estimates for the relationships between vessel size (in kW) or overall length ... and overall footprint size"). Path type "OT door spread", door spread in metres = a(kW^b):
  - OT_CRU: a = 5.1039, b = 0.4690 (SE 1.8153, 0.0598; n = 124);
  - OT_DMF: 9.6054, 0.4337 (n = 39);
  - OT_MIX: 10.6608, 0.2921 (n = 94);
  - OT_MIX_CRU: 37.5272, 0.1490 (n = 271);
  - OT_MIX_CRU_DMF: aLOA+b with a = 3.9273, b = 35.8254 (n = 53);
  - SDN_DMF seine rope length 1948.8347·kW^0.2363.
  - The text gives "door spread for OTs, which equals the total width of this gear" and "penetration depth of the individual components was indexed as either surface (<2 cm) or subsurface (≥2 cm)".
  - (Fetched full text) - [Eigaard et al. 2016, ICES JMS 73:i27](https://archimer.ifremer.fr/doc/00310/42137/41505.pdf)
- **Corrigendum, Table 5** (component shares of the footprint, speed, vessel size), for OT_CRU "Nephrops or shrimp":
  - doors/clumps/weights 2.7%, sweeps and bridles 67.9%, ground gear 29.4%;
  - towing speed 2.5 (±0.3) knots; engine power 345.5 kW (±210.0).
  - OT_DMF: 1.6 / 86.0 / 12.4%, 3.1 kn, 441.7 kW. OT_MIX: 1.7 / 80.9 / 17.4%, 2.8 kn, 400.6 kW. OT_MIX_CRU: 3.0 kn, 681.0 kW.
  - Corrected text: "This is ≈100% more than the total swept area estimate of otter trawling for demersal fish (≈0.8 km2), for which impact at the subsurface level is estimated to be ≈0.1 km2".
  - The original paper says OT_CRU ground gear on soft bottom "was assigned to have impacts entirely at the subsurface level".
  - (Fetched full text) - [Eigaard et al. 2016 corrigendum, ICES JMS doi:10.1093/icesjms/fsw116](https://archimer.ifremer.fr/doc/00310/42137/45569.pdf)
- **WGSFD BENTHIS table used for the ICES products** (`tblAux_benthisGearWidthsForWGSFD17.csv`), columns `av_kw, av_fspeed, subsurface_prop, gearWidth(km)`:
  - OT_CRU 345.5 kW, 2.48 kn, 32.1%, 0.0789 km;
  - OT_DMF 441.7, 3.06, 7.8%, 0.105;
  - OT_MIX 400.6, 2.79, 14.7%, 0.0614;
  - OT_MIX_CRU 681, 3.01, 29.2%, 0.105;
  - OT_MIX_CRU_DMF 473.1, 2.63, 22.9%, 0.114;
  - SDN_DMF subsurface 0%.
  - (Fetched full text) - [ices-eg/wg_WGSFD on GitHub](https://raw.githubusercontent.com/ices-eg/wg_WGSFD/master/data/data_products_2018/tblAux_benthisGearWidthsForWGSFD17.csv)
- **Current ICES data-call workflow.** Gear width per VMS ping comes from a reported `LE_GEARWIDTH` if present, otherwise "the modeled gear width using benthis methods in ICES SFDSAR Package", otherwise "the default average gear width by metier provided in benthis". "Trawls: SA = gear_width * time * speed * 1.852". "Danish seine (SDN): Uses rope loop geometry, SA = (time / 2.591234) * gear_width^2 / pi / 4".
  - (Fetched full text, script comments) - [ICES-VMS-and-Logbook-Data-Call, 2_eflalo_tacsat_analysis.R](https://raw.githubusercontent.com/ices-eg/ICES-VMS-and-Logbook-Data-Call/HEAD/2_eflalo_tacsat_analysis.R)
- **Skagerrak shrimp fleet size (Swedish logbooks, 2015).** "The size of the vessels ranges between 10-34 m with an average of 19 m ... The average engine effect is around 368 kW (92-738 kW)". "During the last 7 years the twin trawlers have increased their landings from 7 to around 50% of total Swedish Pandalus landings."
  - Danish Pandalus vessels fell "from 138 vessels in 1987 to only 9 vessels in 2015 ... fleet size seems to have stabilized at around 10-12 larger vessels", and twin trawl "today is the most common gear deployed".
  - (Fetched full text) - [NAFO SCR Doc. 16/056](https://www.nafo.int/Portals/0/PDFs/sc/2016/scr16-056.pdf)
- **Computed k per métier and kW (Local).** Total footprint (surface SAR) in km2 per MW-h, from Table 4 and the corrigendum's Table 5 speeds:

  | kW | OT_CRU (2.5 kn) | OT_DMF (3.1 kn) | OT_MIX (2.8 kn) | OT_MIX_CRU (3.0 kn) |
  |---|---|---|---|---|
  | 150 | 1.65 | 3.23 | 1.59 | 2.93 |
    | 300 | 1.14 | 2.18 | 0.98 | 1.63 |
  | 345.5 | 1.06 | 2.01 | 0.88 | 1.44 |
  | 400 | 0.98 | 1.85 | 0.80 | 1.27 |
  | 500 | 0.87 | 1.63 | 0.68 | 1.05 |
  | 700 | 0.73 | 1.35 | 0.54 | 0.79 |

  - Door spread for OT_CRU is 54 m at 150 kW, 74 m at 300 kW and 94 m at 500 kW, giving 0.25 / 0.34 / 0.44 km2 per fishing hour.
  - Subsurface k = surface k x subsurface share: OT_CRU x 0.321 gives 0.53 / 0.37 / 0.28 at 150 / 300 / 500 kW, and OT_MIX x 0.147 gives 0.23 / 0.14 / 0.10.

### Inferences
- For a 150-500 kW Skagerrak shrimp/Nephrops trawler (OT_CRU), the gear model gives **k_surface ≈ 0.87-1.65, about 1.1 at the fleet's typical 350-370 kW**, and k_subsurface ≈ 0.28-0.53.
  - The fleet's effort is weighted toward the larger vessels, which fish more hours at higher kW. The effort-weighted k is therefore nearer the 400-500 kW end (0.9-1.0) than the vessel-count mean suggests.
- k is a ratio of sums. A fleet's k is Σ(width·speed·hours)/Σ(kW·hours), so a map in MW-h can be converted linearly only if the vessel and métier mix is the same across cells. The 0.96 cell-wise correlation between ICES subsurface SAR and the effort map (Q1) says this holds well enough in this grid.

## Q3. Cross-check: the ICES SAR against the converted effort map

### Takeaway
ICES SAR is itself built from the Eigaard models, so this is **not an independent validation**. It shows which effective kW and métier mix the ICES numbers imply for this grid.

The ratio is computed in grid c-squares dominated by otter trawling (seine effort < 5% of otter effort; these carry 94-99% of the grid's otter effort). It is **0.74 surface and 0.16 subsurface per MW-h/km2 in both 2019-2022 and 2021-2024**, and 0.98-1.09 surface and 0.22-0.24 subsurface in 2015-2021. The subsurface/surface ratio is 0.21-0.23 in every window.

Applied to the model's effort map (mean 3.01, max 9.62 MW-h/km2/yr), k_surf = 0.74 gives an otter-trawl **surface SAR with grid mean 2.2 yr^-1 and cell max 7.1 yr^-1**. k_sub = 0.16 gives a **subsurface SAR with grid mean 0.48 and cell max 1.5**. The ICES all-gear grid means for 2021-2024 are 2.7-3.0 surface and 0.49-0.50 subsurface. The surface difference is the seine footprint.

### Cited Findings
- **Effort-weighted SAR per MW-h in otter-dominated grid c-squares (Local, computed;** sum(SAR)/sum(effort density) over c-squares with seine effort < 5% of otter effort and otter effort > 0.5 MW-h/km2):

  | Window | Surface k | Subsurface k | Sub/surface | Share of grid OT effort |
  |---|---|---|---|---|
  | 2015-2018 | 1.088 | 0.242 | 0.223 | 0.92 |
  | 2017-2020 | 1.028 | 0.233 | 0.227 | 0.98 |
  | 2018-2021 | 0.983 | 0.224 | 0.228 | 0.99 |
  | 2019-2022 | 0.744 | 0.158 | 0.212 | 0.94 |
  | 2021-2024 | 0.738 | 0.159 | 0.215 | 0.96 |

  - The c-square medians are similar: surface 1.28-1.42 (2015-2021) and 0.67-0.74 (2019-2024); subsurface 0.21-0.22 and 0.16.
  - Least-squares fit over all grid c-squares, 2021-2024: surface ≈ 0.644·OT + 5.83·SDN, subsurface ≈ 0.161·OT + 0.08·SDN.
  - Seines therefore carry about 5.8 surface SAR per MW-h/km2, roughly 8x the otter trawl per MW-h, and essentially no subsurface SAR. In 2021-2024 that is about a quarter of the grid's surface SAR (0.71 of 2.99) on 4% of the effort.
- **Converted effort map (Local, computed** on `maps/fishing_bottom_otter_trawls.npz`: mean 3.01, median 2.25, 90th percentile 6.58, max 9.62 MW-h/km2/yr, 91 cells < 0.1):

  | k_surf | Grid mean | Cell max | Cells with SAR > 1 | Cells with SAR > 3 |
  |---|---|---|---|---|
  | 0.60 | 1.81 | 5.8 | 68% | 23% |
  | 0.74 | 2.23 | 7.1 | 75% | 29% |
  | 1.00 | 3.01 | 9.6 | 80% | 37% |
  | 1.10 | 3.32 | 10.6 | 81% | 41% |

  - Subsurface: k_sub 0.12 / 0.16 / 0.24 gives grid means 0.36 / 0.48 / 0.72 and cell maxima 1.15 / 1.54 / 2.31.
  - For comparison, the ICES subsurface layer rasterised with the same recipe has mean 0.49 and max 1.41, and the c-square max is 2.31.
- **Implied fleet (Local, computed from the Q2 formulas).** k_surf = 0.74 corresponds to OT_CRU at about 680 kW or OT_MIX at about 440 kW. k_surf ≈ 1.0-1.1 corresponds to OT_CRU at 300-400 kW, which matches the Swedish fleet average of 368 kW.
  - The observed subsurface/surface ratio of 0.21-0.23 lies between OT_CRU (0.321; 0.304 in Rijnsdorp et al. 2020) and OT_MIX (0.147). It is closest to OT_MIX_CRU_DMF (0.229).

### Inferences
- The 2015-2021 ICES ratio (about 1.0-1.1) is what the Eigaard OT_CRU model gives for the documented Skagerrak shrimp fleet, so it reads as the "complete" fleet.
- The 2019-2024 ratio (0.74) implies a much larger effective vessel, or more OT_MIX. It coincides with the effort halving, which fits smaller (higher-k) vessels or one country's data leaving the submission. It does not fit a change of gear, because the subsurface share did not move.
- Either way, **the effort map and k must come from the same window**. The map is scaled to the 2021-2024 level, which pairs with k_surf ≈ 0.74. Scenario maps with "more" effort, built on the 2015-2021 level, pair with about 1.0.
  - The product k x effort comes out nearly the same either way: 2021-2024 surface SAR is 0.74 x 3.07 ≈ 2.3, and 2018-2021 is 0.98 x 6.13 ≈ 6.0. The two windows still disagree by a factor of about 2.6, and only one of them can be "today".
- The rasterised SAR is uniform within each 16 km2 c-square, and so is the effort map. True 1-km SAR is more clustered. ICES warns that annual footprints at c-square scale overstate the area trawled and understate intensity inside it. A linear d·SAR loss is insensitive to this at the cell mean, but not to where the trawled patches are.

## Q4. Recommended conversion factor and which SAR pairs with d (inference)

### Takeaway
**Recommendation (inference):** use the surface (total-footprint) SAR with Hiddink et al. 2017 / Sciberras et al. 2018 depletion, and

k_surf = 0.75 SAR per MW-h/km2 (central, consistent with the 2021-2024 map), range 0.6-1.1.

On the model's map this gives a grid-mean SAR of about 2.2 yr^-1 and a cell max of about 7 yr^-1.

Hiddink's d is "the depletion of biota caused by each trawl pass", estimated over the whole gear's path. The penetration depth that predicts d is averaged over doors, sweeps and bridles, weighted by component width, and Hiddink's F is the swept-area ratio of the area trawled. The ICES/WGFBIT PD method (Rijnsdorp et al. 2020) applies métier-specific d to the trawling intensity of the total footprint, not to subsurface SAR. This is inferred, because the paper never literally says "surface". It rests on two pieces of evidence. First, its Table 2 lists the PD driver as plain "Trawling frequency", while the L2 method is marked "(subsurface)". Second, the subsurface ratio enters through d itself (d_m = 0.06·SSR_m/0.18), which would be double counting if T were already subsurface.

If subsurface SAR is used instead (k_sub = 0.16, range 0.12-0.25), d must be rescaled to per-subsurface-pass (d/SSR, about 0.06/0.18 ≈ 0.33 for a "standard" otter trawl). Otherwise the loss is understated by a factor of about 4-6.

### Cited Findings
- **Hiddink et al. 2017.** "B/K = 1 − F d / r ... where F is trawling frequency, d is the depletion of biota caused by each trawl pass (expressed as a proportion), and r is rate of increase".
  - "Trawling frequency F is defined as the swept area ratio, which is the area trawled annually divided by the studied area ...; it should ideally be calculated for small cells (∼1 km2), because trawling tends to be spatially clustered".
  - "Otter trawls caused the least depletion, removing 6% of biota per pass and penetrating the seabed on average down to 2.4 cm, whereas hydraulic dredges caused the most depletion, removing 41%".
  - "Predicted penetration depth of each gear type into the seabed was estimated from values in the literature by averaging the reported penetration depths of the individual components of the gear (e.g., doors, sweeps, and bridles of an OT) weighted by the width of these components".
  - (Fetched full text) - [Hiddink et al. 2017, PNAS 114:8301 (PMC5547586)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5547586/)
- **Rijnsdorp et al. 2020, the PD approach.** I_PD = 1 − Σ K_i (1 − Σ_m T_m d_m / r_i), "where ... T_m is the trawling intensity, and d_m is the depletion rate of métier m".
  - Table 2 lists the PD drivers as "Trawling frequency" and "Depletion rate". L2 instead uses "Trawling frequency (subsurface)".
  - The depletion of each otter métier was "estimated using the width of gear elements that penetrate into the seafloor relative to the total gear width (termed subsurface ratio SSR ...). The subsurface ratio of the standard otter trawl was set equal to the mean subsurface ratio of all otter trawl métiers weighted over their swept area (ratio = 0.18) ... estimated by 0.06 × SSR_m/0.18".
  - Table 1: "OT_CRU ... Nephrops, Pandalus, mixed fish 0.304 0.100"; OT_DMF 0.078 / 0.026; OT_MIX_1 0.229 / 0.075; SDN 0.000 / 0.009 ("Set equal to lowest depletion rate of any otter trawl metiers").
  - Hiddink 2017 ranges as quoted there: otter trawls "median: 0.06; 5–95% range: 0.02–0.16".
  - (Fetched full text) - [Rijnsdorp et al. 2020, ICES JMS 77:1772](https://edepot.wur.nl/535091)
- **Sciberras et al. 2018.** "A gear pass reduced benthic invertebrate abundance by 26% and species richness by 19%". Gears that penetrate deeper had larger effects. (Fetched abstract) - [Sciberras et al. 2018, Fish Fish 19:698](https://research.bangor.ac.uk/portal/cy/researchoutputs/response-of-benthic-fauna-to-experimental-bottom-fishing(824e2219-7719-4db5-b3d9-d83ff46ced4c)/export.html)
- **Eigaard et al. 2017** argue that for soft-sediment infauna "the subsurface level contact will be most relevant", but this concerns their seabed-integrity indicator, not a per-pass d. (Fetched full text) - [Eigaard et al. 2017](https://archimer.ifremer.fr/doc/00362/47277/47238.pdf)

### Inferences
- **Which SAR goes with d.** Hiddink's d comes from experimental trawl passes sampled across the trawled strip, with the penetration predictor weighted over the whole gear width. F = SAR is then the area trawled, i.e. the total footprint.
  - Rijnsdorp et al. 2020 confirm the convention in an ICES context: their otter-trawl d values (0.06 for the reference trawl; 0.10 for OT_CRU) multiply the total trawling intensity. The SSR enters only through d.
  - So `loss per tick = d x SAR_surface / 1460` with SAR_surface = k_surf x effort is the consistent form.
  - If the sister note picks the OT_CRU-specific d ≈ 0.10, it already contains the penetration weighting. Do not multiply it by the subsurface share again.
- **Central k = 0.75, range 0.6-1.1:**
  - 0.75 is the ICES ratio for the same window (2021-2024) as the map's level (0.74; the LSQ gives 0.64 with seines separated, which sets the low end at about 0.6).
  - 1.1 is the upper end, from the 2015-2021 ICES ratio and the Eigaard OT_CRU model at the documented 350-370 kW fleet average.
  - The pure-gear-model span for 150-500 kW shrimp trawlers is wider (0.87-1.65), but small vessels carry little of the effort.
  - Norwegian vessels missing from both the map and SAR affects the level of the map, not k, unless their vessels differ systematically in size.
- **Seines are outside this factor.** Danish/anchor seining adds surface SAR at about 5.8 per MW-h/km2. That is about 25% of the grid's 2021-2024 surface SAR and much more in the southern edge rows (57.78-57.83 N). Its d is far lower (SDN 0.009 in Rijnsdorp et al. 2020, against 0.06-0.10 for otter trawls), so leaving seines out changes the benthic loss by only a few percent. Adding them would need the `fishingbottomseines` map with its own k and d.

## Implications for the model

1. **Table form.** The benthos loss from bottom otter trawling per tick is d x k_surf x E / 1460, with E the `bottom_trawling` map (MW-h/km2/yr) and 1460 ticks per year at the library's 6-h tick. If the term is put in an impact table as `biomass_factor`, `lib/world/tick_time.py` already rescales it as a "loss" rate for other `--tick-length` values (`IMPACT_ROW_RESCALE_RULES`). A separate parameter would need its own rescale rule.
   - For a table linear in E, the per-tick loss fraction per MW-h/km2/yr is d x 0.75 / 1460.
   - With d = 0.06 (Hiddink OT median) this is 3.1e-5 per tick per MW-h/km2/yr. With d = 0.10 (OT_CRU, Rijnsdorp 2020) it is 5.1e-5.
2. **Expected magnitudes on today's map** (surface SAR mean 2.2, max 7.1 yr^-1). With d = 0.06, the annual removal rate is about 13% of benthic biomass on the grid mean and about 43% in the hottest cell. With d = 0.10, it is about 22% and 71%. These are gross removal rates before recovery, not equilibrium depletions.
   - About 75% of cells have SAR > 1 and 29% have SAR > 3. These are OSPAR's "High" and "Very High" classes.
   - Bratten-closure cells and the 91 near-zero cells get essentially no trawling loss.
3. **Use surface (total-footprint) SAR, not subsurface**, with the Hiddink/Sciberras d. The alternative pairing is k_sub = 0.16 with d/SSR (about 0.33). It is algebraically equivalent only if the SSR of the local fleet (0.21-0.23 from ICES) is the one used.
4. **Keep the window consistent.** The map is at the 2021-2024 level, and k = 0.75 belongs to that level. Do not combine it with the 2015-2021 SAR levels (about 6 yr^-1). If a "pre-2019" scenario map is made, use k ≈ 1.0.
5. **Optional refinement.** Instead of k x effort, read the ICES `fishingsubsurface` / `fishingsurface` layers straight into SAR maps (same rasteriser, attributes `sbsrfc_sr` / `srfc_sr`).
   - Those layers include seines in the surface SAR, so for the otter-trawl d use the effort-based surface SAR, or subsurface SAR / 0.22.
   - The effort-map route has the advantage that scenario maps in MW-h stay valid.
6. **Lower bound.** Vessels < 12 m and Norwegian vessels are missing, so all SAR here is probably a lower bound for the real pressure.

## Source status

| Source | Used for | Status |
|---|---|---|
| EMODnet HA WFS GetCapabilities and layers `fishingsurface`, `fishingsubsurface`, `fishingbottomottertrawls`, `fishingbottomseines`, `fishingbeamtrawls`, `fishingdredges` | Layer names, definitions; per-c-square SAR and effort in the grid | Fetched full text / data downloaded (Local computation) |
| ICES Technical Service sr.2018.14 (to OSPAR) | SAR definition, surface/subsurface, métiers, caveats | Fetched full text |
| OSPAR QSR 2023 BH3a | Subsurface is part of surface; Norway VMS excluded; SAR classes | Fetched full text |
| Eigaard et al. 2016 + corrigendum | Door-spread models, speeds, component shares, kW | Fetched full text |
| WGSFD `tblAux_benthisGearWidthsForWGSFD17.csv` | Subsurface proportions and average kW/speed used by ICES | Fetched full text |
| ICES VMS data-call scripts (GitHub) | Swept-area formula, gear-width hierarchy | Fetched full text (script comments) |
| icesVMS R package / `taf.ices.dk/vms/api` | Per-métier SAR, BENTHIS parameters | API returned HTTP 500; SAR endpoints need a login (Gap) |
| Eigaard et al. 2017 | Total width = surface; IIIa intensity 3.63 / 2.56 yr^-1 (2010-2012) | Fetched full text |
| Hiddink et al. 2017 (PNAS) | d per pass, F = SAR, penetration weighted over the whole gear | Fetched full text |
| Rijnsdorp et al. 2020 (ICES JMS) | PD model pairs d_m with total trawling intensity; OT_CRU d = 0.10, SSR 0.304 | Fetched full text |
| Sciberras et al. 2018 | 26% per pass (all gears) | Fetched abstract |
| NAFO SCR 16/056 | Swedish shrimp fleet 368 kW (92-738), twin trawls; Danish fleet 9-12 vessels | Fetched full text |
| Sister note `local_fishery.md` | 60% Skagerrak drop between windows; Norway coverage question | Local |
| Per-métier (OT_CRU/OT_DMF/OT_MIX) SAR for the grid's c-squares; ICES explanation of the 2019-2022 drop | Q1 / Q3 | Not found (Gap) |
