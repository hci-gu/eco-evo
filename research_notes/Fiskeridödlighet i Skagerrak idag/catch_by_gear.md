# Catch by gear for the Skagerrak gadoid and pelagic groups, and how to split each group's F between the bottom-otter-trawl and pelagic EMODnet maps

Scope note: about 55 tool calls. 6 were WebSearch and 0 WebFetch. The rest were shell downloads (curl) of primary files
and local processing, so the numbers come from primary tables rather than from search snippets:
- The 2026 ICES advice sheets for had.27.46a20, whg.27.47d, whg.27.3a, cod.27.46a7d20, pok.27.3a46 and hke.27.3a46-8abd,
  read as text from the PDFs the sister note `ices_gadoid_f.md` had already downloaded. I used the "Catch distribution
  by fleet in 2025" tables and the whiting 3.a catch history. (Fetched full text)
- The HAWG 2026 report (full PDF, already downloaded by `ices_pelagic_f.md`), used for:
  - herring fleet definitions;
  - 3.a.20 herring catch by sub-fleet and by country;
  - Table 11.2.1, the species composition of the Danish sprat fishery, which gives gadoid bycatch in the industrial
    fishery, North Sea and 3.a separately;
  - the NSAS catch by fleet.
  (Fetched full text)
- The STECF FDI 2025 data call, Table A ("FDI Catches by country", 2022-2024), downloaded as the JRC zip
  ([jeodpp FDI folder](https://jeodpp.jrc.ec.europa.eu/ftp/public/JRC-OpenData/FAD/fdi/),
  `2025_Effort-landings-catches-capacity-biological.zip`, 157 MB) and aggregated by sub-region, species, gear,
  target and mesh. (Fetched full text; the aggregation is mine)
- The ICES RCG métier list with its BENTHIS column (`RDB_ISSG_Metier_list.csv`, ICES VMS data-call repository on
  GitHub), the ICES Fisheries Overview R script that names the VMS gear layers (`report5_VMS.R`, ices-taf
  2022_NrS_FisheriesOverview), the EMODnet WFS GetCapabilities, and Eigaard et al. 2016 (full PDF). (Fetched full text)
- The ICES VMS web service's métier-to-gear-category lookup (`taf.ices.dk/vms/api/MetierLookup`) returned HTTP 500 on
  three tries. Section Q4 therefore infers the last link in the chain (BENTHIS OT_* -> FO category "Otter") and says so.

Everything marked "computed" is my own arithmetic on these files.

**FDI caveats, which apply to every FDI number below:**
- FDI is EU-only. Norway, which lands much of the herring and shrimp in 45G0, is missing. So is the UK after 2020.
- Cells with too few vessels are published as "C" (confidential). I dropped them and give the count of C rows where it
  matters.
- The readme warns: "Discard amounts in the catch data are scientific discard estimates ... these estimates might be
  uncertain and biased."

---

## Q1. Gadoid catches by gear in 3.a (and North Sea for context)

### Takeaway
- In the Skagerrak (3.a.20), gadoids are caught almost entirely by demersal gear:
  - bottom otter trawls, mainly the mixed Nephrops/demersal métier OTB_MCD: 59-97 % by species;
  - Danish and Scottish seines: 2-26 %;
  - gillnets: 1-18 %.
- Visible EU pelagic-trawl (OTM/PTM/PS) catch of cod, haddock, saithe and hake in 3.a.20 is 0.0 % in 2022-2024. For
  whiting it is 0.7 %.
- The only material non-demersal gadoid catch is industrial bycatch (IBC) of whiting. In 3.a it is 6-7 % of the whiting
  catch in 2021-2025 (24 % in 2020). In 2024-2025 nearly all of it came from the Danish small-mesh sprat fishery.
- At stock level, the non-demersal shares are:
  - haddock: IBC 1.7 % of catch;
  - North Sea whiting: IBC 6.1 %;
  - saithe: 5.7 %, mainly Norwegian purse seine far from the Skagerrak;
  - cod and hake: about 0 %.

### Cited Findings
- Haddock (Subarea 4, 6.a, Subdivision 20), Table 7 "Catch distribution by fleet in 2025":
  - catch 69 838 t, of which landings 53 344 t and discards 15 292 t;
  - landings: demersal trawl and seine >=100 mm 92 %, trawl 70-99 mm 1 %, other 7 %;
  - discards: demersal trawl and seine >=100 mm 61 %, trawl 70-99 mm 18 %, other 21 %;
  - "Industrial bycatch 1 201 tonnes".
  — [ICES Advice 2026 had.27.46a20](https://doi.org/10.17895/ices.advice.30932057) (Fetched full text)
- Whiting (Subarea 4 and 7.d), Table 7, 2025:
  - catch 33 331 t, landings 14 766 t, discards 16 544 t, industrial bycatch 2 021 t;
  - landings: demersal trawls/seine >=100 mm (North Sea) 60 %, 70-99 mm (North Sea) 15 %, 70-99 mm (Eastern English
    Channel) 9 %, other 16 %.
  — [ICES Advice 2026 whg.27.47d](https://doi.org/10.17895/ices.advice.30932456) (Fetched full text)
- Whiting in 3.a:
  - Table 6, 2025: catch 1119 t. Landings 395 t: human consumption bottom otter trawl 74 %, others 8 %, "Industrial
    bycatch (Denmark)" 18 %. Discards 724 t: bottom otter trawl 96 %, others 4 %.
  - Table 7, Danish industrial bycatch / ICES catch (t): 2020 437 / 1 830; 2021 85 / 1 212; 2022 53 / 709; 2023 44 / 726;
    2024 58 / 854; 2025 71 / 1 119.
  - "Whiting in Division 3.a are mainly caught as bycatch in demersal fisheries" (quoted in `ices_gadoid_f.md`).
  — [ICES Advice 2026 whg.27.3a](https://doi.org/10.17895/ices.advice.30932453) (Fetched full text)
- Cod (Subarea 4, 6.a, 7.d, Subdivision 20), Table 7, 2025: catch 24 764 t. Landings: demersal trawls and seines
  >=100 mm 79 %, gillnets 10 %, demersal trawls 70-99 mm 6 %, beam trawls 1 %, other gears 4 %. Discards: 56 % / 7 % /
  31 % / 1 % / 4 %. There is no pelagic or industrial category. — [ICES Advice 2026 cod.27.46a7d20](https://doi.org/10.17895/ices.advice.30932018) (Fetched full text)
- Saithe (4, 6, 3.a), Table 7, 2025: catch 73 382 t. Landings 67 683 t: bottom otter trawl 85.2 %, purse seine 6.1 %,
  gillnet 3.9 %, other 4.8 %. Discards 5 698 t: 98 % / 1.0 % / 0.6 % / 0.4 %. — [ICES Advice 2026 pok.27.3a46](https://doi.org/10.17895/ices.advice.30932291) (Fetched full text)
- Hake (Northern stock), Table 7, 2025: catch 51 834 t. Landings 48 053 t: gillnet 49.6 %, trawl 25.1 %, longline 17.1 %,
  unspecified 8.2 %. Discards 3 781 t. — [ICES Advice 2026 hke.27.3a46-8abd](https://doi.org/10.17895/ices.advice.30932108) (Fetched full text)
- Danish sprat fishery species composition, HAWG Table 11.2.1 (t), whiting / haddock / cod:
  - North Sea (27.4): 2022 488 / 124 / 0; 2023 1 802 / 1 120 / 0; 2024 1 414 / 916 / 0; 2025 1 789 / 1 044 / 0.
  - 3.a: 2020 249 / 5 / 1; 2021 13 / 1 / 0; 2022 1 / 0 / 0; 2023 0 / 0 / (blank); 2024 53 / 0 / 0; 2025 58 / (blank) / 0.
    The 2025 row is printed with area "37.3.a" [sic].
  - Sprat in the 3.a rows: 638 t (2021), 316 t (2022), 17 t (2023), 1 835 t (2024), 3 364 t (2025).
  — [HAWG 2026, Table 11.2.1](https://ices-library.figshare.com/articles/report/Report_of_the_Herring_Assessment_Working_Group_for_the_Area_South_of_62_N_HAWG_/31424315) (Fetched full text)
- HAWG 2026 Table 11.1.1, 2025 sprat landings: 27.3.a.20 1.0 kt, 27.3.a.21 2.5 kt. Most 3.a sprat was therefore taken
  in the Kattegat, not the Skagerrak (as cited in `ices_pelagic_f.md`). (Fetched full text)
- STECF FDI Table A, 27.3.a.20, EU landings + discards, 2022-2024 pooled (computed). Columns are bottom otter trawl
  (OTB/OTT/PTB), pelagic (OTM/PTM/PS), seine (SDN/SSC), static (GNS/GTR/LLS/...) and beam (TBB):
  - Cod, 6 912 t: 75.0 % / 0.0 % / 8.7 % / 15.8 % / 0.5 % (8 C rows in pelagic gear)
  - Haddock, 11 886 t: 74.3 % / 0.0 % / 24.8 % / 0.6 % / 0.3 % (33 C rows in pelagic gear, all OTM_DEF/PTM_DEF)
  - Whiting, 1 470 t: 95.0 % / 0.7 % / 4.0 % / 0.2 % / 0.1 % (35 C rows in pelagic gear)
  - Saithe, 3 630 t: 96.8 % / 0.0 % / 2.0 % / 1.1 % / 0.0 %
  - Hake, 1 968 t: 59.3 % / 0.0 % / 26.2 % / 13.8 % / 0.7 %
  - The leading single métier is OTB_MCD (mixed crustacean/demersal): 45-49 % of cod, 49-52 % of haddock, 75-87 % of
    whiting, 36-49 % of saithe and 45-53 % of hake.
  - Visible whiting in industrial small-mesh trawls (mesh <32 mm), landings: 2022 OTB_DEF_00D16 12.45 t + OTM_DEF
    6.78 t; 2023 4.08 t + 3.26 t; 2024 none visible (C rows only).
  — [STECF FDI 2025 data call, Table A](https://jeodpp.jrc.ec.europa.eu/ftp/public/JRC-OpenData/FAD/fdi/) (Fetched full text; aggregation computed)
- STECF FDI, North Sea (27.4), EU only: the pelagic-gear shares of the EU-visible haddock and whiting catch (37 % and
  28 %) cover only about 5 of the about 70 kt haddock stock catch, so I do not use them as stock-level shares. (Computed)

### Inferences
- Local (3.a.20) share of each species' catch taken by pelagic-map gear (OTM/PTM/PS):
  - Cod, haddock, saithe and hake: about 0 %. The HAWG table bounds only the sprat-fishery part of the 3.a industrial
    bycatch of cod and haddock (0-1 t a year in 2021-2025). The sandeel/pout part is unverified: all 33 OTM_DEF/PTM_DEF
    haddock rows in 3.a.20 are confidential, and for whiting that fishery gave most of the IBC in 2022-23 (ICES 53 and
    44 t against 1 and 0 t from the sprat fishery). Variant (c) in Q3 carries the stock-level 1.7 % for haddock.
  - Whiting:
    - 0.7 % visible in FDI;
    - about 5-6 % in 2024-2025, if the Danish sprat-fishery bycatch (53-58 t, Table 11.2.1) is pelagic-coded;
    - the advice's IBC of 71 t in 2025 (18 % of 395 t) is a separate estimate of the same quantity, close to the HAWG 58 t;
      I do not blend the two.
- The Danish 3.a sprat catch was 2.5 kt in the Kattegat against 1.0 kt in the Skagerrak in 2025. The Skagerrak part of
  that whiting bycatch is therefore likely smaller than the 3.a figure.
- Stock-level non-demersal shares (computed from the ICES Table 7 numbers above):
  - haddock 1 201 / 69 838 = 1.7 %;
  - North Sea whiting 2 021 / 33 331 = 6.1 %;
  - 3.a whiting 71 / 1 119 = 6.4 %;
  - saithe (67 683 x 0.061 + 5 698 x 0.010) / 73 382 = 5.7 % (purse seine);
  - cod and hake 0 %.
- ICES "IBC" means small-mesh industrial trawls of either gear type, not pelagic trawls only. In the North Sea in 2025,
  1 044 of the 1 201 t haddock IBC and 1 789 of the 2 021 t whiting IBC came from the Danish sprat fishery (HAWG
  Table 11.2.1), which FDI logs as OTM_SPF (Q4).
- The saithe purse-seine share is a Norwegian fishery outside the Skagerrak. It belongs in a stock-level variant only.

---

## Q2. Herring, sprat and mackerel catches by gear in 3.a

### Takeaway
- At stock level, pelagic species are caught almost only by pelagic gear. Shares on bottom-contacting gear: North Sea
  herring 0.2 %, sprat 0.7 % and mackerel 1.1 % of the EU catch (FDI 2022-2024). The NSAS industrial "B fleet", which
  is mostly the pelagic-coded sprat fishery, is 2.2 % of the NSAS catch.
- In 3.a.20 the herring fishery has been almost closed since 2022 (0.35-0.79 kt a year). Here the demersal share is no
  longer trivial: herring bycatch and discards in Nephrops/shrimp trawls plus herring in bottom-coded industrial trawls
  make up 0.2-8 % of the HAWG catch (landings only), or 4-21 % if FDI's discard estimates are added.
- Local mackerel is mostly hook-and-line and drift-net catch. Those gears are on neither map.

### Cited Findings
- Fleet definitions: "Fleet C: Directed fishery for herring in Kattegat and Skagerrak in which trawlers (with 32 mm
  minimum mesh size) and purse seiners participate. Fleet D: By-catch of herring in Kattegat and Skagerrak in the
  industrial fisheries. Covering all fisheries using towed nets with mesh sizes less than 32 mm, hence excl. purse
  seiners, e.g. the Sprat, Norway pout and Sandeel fishery". Table 3.1.2 maps 27.3.a.20 sub-fleets as follows: passive
  -> C, purse seine -> C, trawl < 32mm -> D, trawl >= 32mm -> C. — [HAWG 2026, section 3.1.3](https://ices-library.figshare.com/articles/report/Report_of_the_Herring_Assessment_Working_Group_for_the_Area_South_of_62_N_HAWG_/31424315) (Fetched full text)
- 3.a herring catch by fleet, C / D (kt), HAWG Table 3.1.3: 2017 24.99 / 1.08; 2018 22.84 / 0.42; 2019 14.32 / 0.60;
  2020 16.13 / 1.76; 2021 12.51 / 0.72; 2022 0.48 / 0.25; 2023 0.76 / 0.23; 2024 0.56 / 0.09. (Fetched full text)
- 3.a.20 herring catch by country, HAWG Table 3.1.1 (kt), total (Norway): 2021 10.18 (1.12); 2022 0.48 (0.25);
  2023 0.79 (0.59); 2024 0.35 (0.26). (Fetched full text)
- 3.a.20 herring 2025 by sub-fleet, summed from the HAWG catch-input rows (computed): purse seine 352.9 t, trawl <32 mm
  92.6 t (all Danish), passive 10.7 t, trawl >=32 mm 0.9 t, total 457.1 t. (Fetched full text; sums computed)
- Discards: "The overall amount of discards has always been assumed to be insignificant. In 2024, Denmark and Sweden
  provided estimates of discards to the Working Group and here the discard accounted for 2-4% of the total catches".
  (Fetched full text)
- NSAS 2025 SOP catch by fleet (HAWG Table 2.2.6, kt): A 418.5, B 9.4, C 0.3, D 0.4, total 428.64. "A fleet figure
  includes unsampled bycatch in the industrial fishery." (Fetched full text)
- The Danish sprat fishery took 8 157 t of herring in the North Sea and 269 t in 3.a in 2025, and "was conducted with a
  5% and 7% bycatch of herring in 2025 in the North Sea and Division 3.a". — [HAWG 2026, section 11.2](https://ices-library.figshare.com/articles/report/Report_of_the_Herring_Assessment_Working_Group_for_the_Area_South_of_62_N_HAWG_/31424315) (Fetched full text)
- Sprat: Table 7 of the 2026 advice lists the 2025 catch of 211 810 t as "Industrial trawl 100%", discards
  "Negligible". — [ICES Advice 2026 spr.27.3a4](https://doi.org/10.17895/ices.advice.32134120) (Fetched full text)
- STECF FDI Table A, 27.3.a.20 herring, EU only (computed):
  - Bottom-otter landings / discards (t): 2022 40.0 / 26.6; 2023 20.4 / 9.4; 2024 0.7 / 91.8.
  - In 2024 the OTB_MCD 80-99 mm rows have 0.22 t landed and 64.74 t discarded.
  - Bottom-otter landings are mostly OTB_DEF_00D16, the industrial small-mesh fishery: 36.9 t in 2022 and 20.0 t in 2023.
  — [STECF FDI Table A](https://jeodpp.jrc.ec.europa.eu/ftp/public/JRC-OpenData/FAD/fdi/) (Fetched full text; computed)
- STECF FDI, 27.3.a.20 mackerel, 2022-2024 pooled (503 t, EU): static and hook gears (GND, LTL, LHP, GNS) 82.0 %,
  bottom otter 13.4 %, pelagic 2.9 %. (Computed)
- STECF FDI, North Sea, EU, 2022-2024 pooled, bottom-otter share (computed):
  - herring 0.2 % of 529 kt;
  - sprat 0.7 % of 217 kt, nearly all of it OTB_SPF (1 402 t);
  - mackerel 1.1 % of 147 kt;
  - horse mackerel 9.9 % of 2.0 kt. (Computed)

### Inferences
- Local herring bottom-trawl share in 3.a.20 (computed). The denominator is the HAWG all-country catch, which includes
  Norway's purse-seine catch:
  - Landings only: 40.0 / 480 = 8.3 % (2022), 20.4 / 790 = 2.6 % (2023), 0.7 / 350 = 0.2 % (2024); 3.8 % pooled.
  - Adding FDI discards to both numerator and denominator: 13.1 %, 3.7 % and 20.9 %; 10.8 % pooled.
  - Measured against the EU-only FDI denominator, the share looks like 36 %. That figure is misleading because Norway is
    missing.
- HAWG treats herring discards as insignificant, while FDI estimates about 90 t of discarded herring in 3.a.20 Nephrops
  trawls in 2024, a quarter of the HAWG catch. The two do not agree. Since the totals are fixed, this only moves the split.
- The D fleet (trawl <32 mm, 92.6 t in 2025) is the industrial fishery. In FDI it shows up under both OTB and OTM codes
  (Q4), so it belongs partly on each map.

---

## Q3. Biomass-weighted split for the model (computed)

All numbers in this section are computed. share_p = sum_s w_s F_s p_s / sum_s w_s F_s.

### Gadoids
The weights and F come from `ices_gadoid_f.md`. Each species' share of the pooled F (w_s F_s / sum) under the W1
Skagerrak 2021-25 weights is cod 0.195, haddock 0.281, whiting 0.185, saithe 0.314 and hake 0.026 (pooled F 0.116).
Under the W3 box weights it is 0.149 / 0.161 / 0.233 / 0.433 / 0.024 (pooled F 0.133).

| Variant for p_s (pelagic-map share) | p_s cod / had / whg / pok / hke | Pelagic-map share, W1 | W3 |
|---|---|---|---|
| (a) Local, FDI 3.a.20 visible, 2022-24 | 0 / 0 / 0.007 / 0 / 0 | 0.13 % | 0.16 % |
| (b) Local + 3.a sprat-fishery whiting bycatch 2024-25 | 0 / 0 / 0.06 / 0 / 0 | 1.1 % | 1.4 % |
| (c) Stock-level, ICES Table 7 (IBC, purse seine) | 0 / 0.017 / 0.061 / 0.057 / 0 | 3.4 % | 4.2 % |

Worked example for (c), W1: 0.281 x 0.017 + 0.185 x 0.061 + 0.314 x 0.057 = 0.0048 + 0.0113 + 0.0179 = 0.034.
Saithe purse seine and whiting IBC are the only terms that matter.

There is also a demersal share that is on neither map. Locally, seines and static gears take cod 25.0 %, haddock
25.7 %, whiting 4.3 %, saithe 3.1 % and hake 40.7 % (FDI, beam trawl included). That is about 15 % of the W1-pooled
gadoid F (0.195 x 0.250 + 0.281 x 0.257 + 0.185 x 0.043 + 0.314 x 0.031 + 0.026 x 0.407 = 0.149).

### Pelagic fish
Weights come from `ices_pelagic_f.md`: scenario A (herring only), and B-mix with F contributions herring 0.262,
sprat 0.260, mackerel 0.370 and horse mackerel 0.022 (sum 0.914).

| Variant for p_s (bottom-map share) | Bottom-map share |
|---|---|
| Stock-level, A (herring only, FDI North Sea 0.2 %, up to 0.5 % if some B-fleet sandeel bycatch is bottom-coded) | 0.2-0.5 % |
| Stock-level, B-mix (herring 0.003, sprat 0.0065, mackerel 0.011, horse mackerel 0.10) | 1.0 % |
| Local 3.a.20 herring 2022-24, landings only / with FDI discards | 3.8 % / 10.8 % (single years 0.2-20.9 %) |
| Blend: 95 % of F fished at stock level, 5 % in-area (in-area rate 0.005 of 0.10, sister note) = 0.95 x 1.0 + 0.05 x 10.8 | 1.5 % |

A B-mix computed with local shares would give about 40 %. That number is an artefact: 3.a.20 mackerel is 82 %
hook-and-line and drift net, which is on neither map, so the bottom-otter part dominates what is left, and the local
horse mackerel tonnage is trivial. I do not use it.

### Inferences
- The gadoid F in the box is mostly realised locally. The sister note's swept-area estimate (u about 0.13) matches the
  pooled 0.12. So the local variants (a)-(b) deserve more weight than (c), but the per-species F values are stock-level.
  A central 2 % on the pelagic map lies between the local (0.1-1.4 %) and stock-level (3.4-4.2 %) results.
- The pelagic F is about 95 % imported: the in-area catch is 0.005 of the 0.10. Stock-level gear shares therefore
  dominate, which gives a central 1.5 % on the bottom map.

---

## Q4. Which map carries industrial small-mesh trawls (OTB_SPF)?

### Takeaway
- The split should follow the gear code, not the target species:
  - OTB/OTT/PTB codes, including OTB_SPF and OTB_DEF with mesh <16 mm, are BENTHIS bottom-contacting otter-trawl
    métiers. They belong on `fishingbottomottertrawls`.
  - OTM/PTM/PS have no BENTHIS métier and belong on `fishingpelagic` ("Pelagic trawls and seines").
- In practice the industrial fishery is rarely coded OTB_SPF:
  - Danish sprat trawling is logged as OTM_SPF, so it lands on the pelagic map.
  - The sandeel and Norway pout fisheries are logged as both OTB_DEF and OTM_DEF with small mesh. In 3.a.20 in 2022 that
    was about 61 % bottom-coded.
  - Industrial bycatch therefore straddles both maps.

### Cited Findings
- RCG métier list (BENTHIS column), North Sea RCG: OTB_SPF_16-31_0_0, OTB_SPF_<16_0_0 and the other OTB_SPF codes ->
  "OT_SPF"; OTB_DEF_<16_0_0 and OTB_DEF_16-31_0_0 -> "OT_DMF"; OTB_MCD -> "OT_MIX_CRU_DMF"; OTB_CRU -> "OT_CRU";
  OTT_SPF and PTB_SPF -> "OT_SPF"; SDN_DEF -> "SDN_DMF"; SSC_DEF -> "SSC_DMF". OTM_SPF, OTM_DEF, PTM_SPF and PS_SPF have
  an empty BENTHIS field. — [ICES VMS data call, RDB_ISSG_Metier_list.csv](https://github.com/ices-eg/ICES-VMS-and-Logbook-Data-Call/blob/main/resources/RDB_ISSG_Metier_list.csv) (Fetched full text)
- Eigaard et al. 2016 class "OTs for small pelagic species (herring, sprat, and sandeel)" as bottom-contacting otter
  trawls, with "very little subsurface impact (2.8%)". OT_SPF is one of the OT BENTHIS métiers in their gear-width
  models. — [Eigaard et al. 2016, ICES JMS 73: i27-i43](https://archimer.ifremer.fr/doc/00310/42137/41505.pdf) (Fetched full text)
- ICES Fisheries Overview script: `gears <- c("Static", "Midwater", "Otter", "Demersal seine","Dredge", "Beam")`,
  recoded with `Midwater = "Pelagic trawls and seines"`, `Otter = "Bottom otter trawls"` and
  `Demersal seine = "Bottom seines"`. — [ices-taf 2022_NrS_FisheriesOverview, report5_VMS.R](https://github.com/ices-taf/2022_NrS_FisheriesOverview/blob/master/report5_VMS.R) (Fetched full text)
- EMODnet layer titles: "Average MW Fishing hours: Bottom otter trawls" and "Average MW Fishing hours: Pelagic trawls
  and seines". Effort is "by gear type (Beam trawls, Bottom otter trawls, Bottom seines, Dredges, Pelagic trawls and
  seines, Static gears)" from the ICES Fisheries Overviews, VMS vessels >12 m only. — [EMODnet WFS capabilities](https://ows.emodnet-humanactivities.eu/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetCapabilities) (Fetched full text)
- FDI coding of the industrial fisheries (computed, EU landings + discards, t):
  - North Sea sprat: OTM_SPF 99 % (OTB_SPF 0.7 %).
  - 3.a.20 sandeel 2022: OTB_DEF 4 931 t, OTM_DEF 3 135 t.
  - North Sea sandeel 2023: OTM_DEF 100 725 t, OTB_DEF 25 124 t.
  - North Sea Norway pout 2022: OTM_DEF 9 534 t, OTB_DEF 4 061 t.
  - Gadoid catch in rows coded OTB_SPF is 0 t (only C rows) in 3.a.20 and under 0.7 % of EU catch in the North Sea.
  — [STECF FDI Table A](https://jeodpp.jrc.ec.europa.eu/ftp/public/JRC-OpenData/FAD/fdi/) (Fetched full text; computed)

### Inferences
- Inferred link: BENTHIS OT_* métiers -> FO category "Otter" -> EMODnet "Bottom otter trawls". I could not fetch the ICES
  lookup itself (HTTP 500). The chain is consistent: OT_SPF is a bottom-contacting otter-trawl métier, the bottom-contact
  (SAR) layers include all "bottom otter trawls", and OTM/PTM/PS have no bottom-contact métier.
- Practical rule:
  - Gadoid IBC from the sprat fishery goes on the pelagic map. IBC from sandeel and Norway pout trawls splits roughly
    60/40 between bottom and pelagic (one year of 3.a.20 data; North Sea data suggest 20-30 % bottom).
  - In 2024-2025 the 3.a whiting IBC came mostly from the sprat fishery, which supports counting IBC as pelagic-map
    effort in variants (b) and (c).

---

## Recommended split per map

| FG | Share on bottom-trawl map | Share on pelagic map | Range (pelagic map for gadoids, bottom map for pelagics) | Basis |
|---|---|---|---|---|
| gadoids | **98 %** | **2 %** | pelagic 0.1-4.5 % | Local FDI 3.a.20 (0.1 %) + sprat-fishery whiting bycatch (1.1-1.4 %) up to stock-level ICES IBC/purse seine (3.4-4.2 %), W1/W3 weights. The 15 % taken by seines and gillnets has no map of its own; it is demersal and stays on the bottom map. |
| pelagic_fish | **1.5 %** (round to 2 %) | **98.5 %** | bottom 0.2-11 % | Stock-level FDI North Sea / HAWG fleets (0.2-1.0 %) blended with the local 3.a.20 herring share (3.8-10.8 %) at the 5 % in-area fraction of F. |

Notes:
- Totals stay at F = 0.12 (gadoids) and 0.10 (pelagic_fish). Per 6-h tick the pelagic-map part of the gadoid F is
  0.12 x 0.02 / 1460 = 1.6e-6, and the bottom-map part of the pelagic F is 0.10 x 0.015 / 1460 = 1.0e-6 (computed).
- Both minority shares lie inside the uncertainty of F itself (gadoids 0.07-0.19). Keeping 100/0 is defensible. The
  98/2 and 98.5/1.5 splits mainly matter if the pelagic and bottom maps are spatially very different inside the grid,
  which they are: pelagic effort is patchy and moves between 4-year windows (`local_fishery.md`).
- If the model ever adds the `fishingbottomseines` layer, about 15 % of the gadoid F (haddock and hake seining) would
  move onto it.

---

## Source status

| Source | Used for | Status |
|---|---|---|
| [ICES Advice 2026 had.27.46a20](https://doi.org/10.17895/ices.advice.30932057) | Haddock catch by fleet, IBC | Fetched full text (PDF) |
| [ICES Advice 2026 whg.27.47d](https://doi.org/10.17895/ices.advice.30932456) | North Sea whiting catch by fleet, IBC | Fetched full text (PDF) |
| [ICES Advice 2026 whg.27.3a](https://doi.org/10.17895/ices.advice.30932453) | 3.a whiting fleet split, Danish IBC 1975-2025 | Fetched full text (PDF) |
| [ICES Advice 2026 cod.27.46a7d20](https://doi.org/10.17895/ices.advice.30932018) | Cod catch by fleet | Fetched full text (PDF) |
| [ICES Advice 2026 pok.27.3a46](https://doi.org/10.17895/ices.advice.30932291) | Saithe catch by fleet (purse seine) | Fetched full text (PDF) |
| [ICES Advice 2026 hke.27.3a46-8abd](https://doi.org/10.17895/ices.advice.30932108) | Hake catch by fleet | Fetched full text (PDF) |
| [ICES Advice 2026 spr.27.3a4](https://doi.org/10.17895/ices.advice.32134120) | Sprat "Industrial trawl 100%" | Fetched full text (PDF) |
| [HAWG 2026](https://ices-library.figshare.com/articles/report/Report_of_the_Herring_Assessment_Working_Group_for_the_Area_South_of_62_N_HAWG_/31424315) | Fleet definitions, 3.a.20 herring by sub-fleet/country, NSAS by fleet, Danish sprat-fishery bycatch (Table 11.2.1) | Fetched full text (PDF) |
| [STECF FDI 2025 data call, Table A](https://jeodpp.jrc.ec.europa.eu/ftp/public/JRC-OpenData/FAD/fdi/) | Catch by gear/métier, 3.a.20 and 27.4, 2022-24 | Fetched full data (zip); EU-only, C rows dropped |
| [ICES RDB_ISSG_Metier_list.csv](https://github.com/ices-eg/ICES-VMS-and-Logbook-Data-Call/blob/main/resources/RDB_ISSG_Metier_list.csv) | Métier -> BENTHIS mapping | Fetched full text |
| [ices-taf 2022_NrS_FisheriesOverview report5_VMS.R](https://github.com/ices-taf/2022_NrS_FisheriesOverview/blob/master/report5_VMS.R) | FO category -> layer names | Fetched full text |
| [EMODnet WFS capabilities](https://ows.emodnet-humanactivities.eu/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetCapabilities) | Layer definitions | Fetched full text |
| [Eigaard et al. 2016](https://archimer.ifremer.fr/doc/00310/42137/41505.pdf) | OT_SPF is a bottom-contacting otter trawl | Fetched full text (PDF) |
| ICES VMS MetierLookup service (`taf.ices.dk/vms/api/MetierLookup`) | BENTHIS -> FO category | Not available (HTTP 500); inferred |
| Swedish (HaV/SLU) and Danish (DTU/Fiskeristyrelsen) landings by gear for Skagerrak | National cross-check | Not searched; FDI covers the EU fleets, and Norwegian gear-level data were not found |
