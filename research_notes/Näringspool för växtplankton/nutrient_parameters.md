# Nutrient-phytoplankton parameters for the Mareld site (Skagerrak/Kattegat, ~58.1 N 10.8 E, ~100 m depth, mixed layer 6-17 m above the Baltic-outflow halocline)

Scope: about 45 tool calls (web searches, page fetches, PDFs run through pdftotext). Four items:
(1) nitrate/DIN half-saturation, (2) the fate of nitrogen grazed by copepods, (3) cross-pycnocline
exchange m and the regional vertical N flux, (4) phytoplankton C:wet weight and the Redfield C:N.

Read-status tags: **[F]** = full text read by me; **[A]** = abstract only; **[S]** = search snippet
or a secondary summary, unverified. "[F] of X" means I read citing source X in full but not the
original it cites. **[D]** = my own calculation. 1 mmol N = 14.007 mg N; 1 umol/l = 1 mmol m-3.

## Sources

| # | Source | Used for | Read |
|---|---|---|---|
| 1 | Fennel K, Wilkin J, Levin J, Moisan J, O'Reilly J, Haidvogel D (2006) Nitrogen cycling in the Middle Atlantic Bight. Global Biogeochem. Cycles 20:GB3007. doi:10.1029/2005GB002456 ([PDF](https://catz.ocean.dal.ca/memg/pubs/Fennel_Wilkin_et_al_GBC_06.pdf)) | Q1, Q2 NPZ values | [F] |
| 2 | Behrenfeld MJ, Boss ES (2014) Resurrecting the ecological underpinnings of ocean plankton blooms. Annu. Rev. Mar. Sci. 6:167-194, Supplement. doi:10.1146/annurev-marine-052913-021325 ([suppl. PDF](https://misclab.umeoce.maine.edu/documents/BehrenfeldBoss2014_supplemental.pdf)) | Evans & Parslow 1985 values (J, m, c3) | [F] of BB14; E&P 1985 not read |
| 3 | Evans & Parslow model web page, Univ. Victoria ([climate.uvic.ca/ep](https://climate.uvic.ca/ep/)) | E&P mixing rate default | [F] of page; E&P not read |
| 4 | Zheng Q, Viljoen JJ, Sun X, Kovac Z, Sathyendranath S, Brewin RJW (2025) Simulating vertical phytoplankton dynamics in a stratified ocean using a two-layered ecosystem model. Biogeosciences 22:3253-3278. doi:10.5194/bg-22-3253-2025 | Fasham 1990 m = 0.1 m/d; Ks 0.45-0.91 | [F] of text (Table 1 is an image, not read); Fasham not read |
| 5 | Levy M (2015) Exploration of the critical depth hypothesis with a simple NPZ model. ICES J. Mar. Sci. 72:1916-1925. doi:10.1093/icesjms/fsv016 | NPZ Kn, az, excretion | [F] |
| 6 | MacIsaac JJ, Dugdale RC (1972) Interactions of light and inorganic nitrogen in controlling nitrogen uptake in the sea. Deep-Sea Res. 19:209-232 ([PDF](https://www.waterboards.ca.gov/waterrights/water_issues/programs/bay_delta/docs/cmnt081712/sldmwa/macisaacanddugdale1972.pdf)) | field ks, eutrophic vs oligotrophic | [F] |
| 7 | Samuelsen A, Hansen C, Wehde H (2015) Tuning and assessment of the HYCOM-NORWECOM V2.1 biogeochemical modeling system. Geosci. Model Dev. 8:2187-2202. doi:10.5194/gmd-8-2187-2015 | NORWECOM (North Sea model) KN, AE | [F] |
| 8 | Steele JH (1998) Incorporating the microbial loop in a simple plankton model. Proc. R. Soc. Lond. B 265:1771-1777 ([WHOI PDF](https://seagrant.whoi.edu/wp-content/uploads/2015/01/WHOI-R-98-009-Steele-J.H.-Incorporating-the.pdf)) | Fasham 1990 growth/grazing 0.75 | [F] of Steele; Fasham not read |
| 9 | Kuijper LDJ, Anderson TR, Kooijman SALM (2004) C and N gross efficiencies of copepod egg production studied using a DEB model. J. Plankton Res. 26:213-226. doi:10.1093/plankt/fbh020 ([PDF](https://research.vu.nl/ws/files/2007525/172389.pdf)) | copepod N budget; quotes Kiorboe et al. 1985 | [F] of Kuijper; Kiorboe 1985 not read |
| 10 | Lundsgaard C, Olesen M (1997) The origin of sedimenting detrital matter in a coastal system. Limnol. Oceanogr. 42:1001-1005 ([record](https://marineinfo.org/doc/publication/145415)) | Kattegat pellet retention | [A] |
| 11 | Saba GK, Steinberg DK, Bronk DA (2011) The relative importance of sloppy feeding, excretion, and fecal pellet leaching in the release of dissolved carbon and nitrogen by Acartia tonsa. J. Exp. Mar. Biol. Ecol. 404:47-56. doi:10.1016/j.jembe.2011.04.013 | dissolved N release pathways | [S] |
| 12 | NPZ model, Wikipedia ([page](https://en.wikipedia.org/wiki/NPZ_model)), table of Franks, Wroblewski & Flierl (1986) | Franks gamma = 0.3 | [S] (Wikipedia, secondary) |
| 13 | Project note `research_notes/Zooplankton energibudget/copepod_energetics.md` (Straile 1997 meta-analysis) | copepod GGE | [F] of the project note (lines 147-163); Straile 1997 not read here |
| 14 | Hasager CB, Carstensen J, Ellermann T, Gustafsson BG, Hertel O, Johnsson M, Markager S, Skjoth CA (2003) On extreme atmospheric and marine nitrogen fluxes and chlorophyll-a levels in the Kattegat Strait. Atmos. Chem. Phys. 3:797-812. doi:10.5194/acp-3-797-2003 | regional entrainment N flux | [F] |
| 15 | Reissmann JH, Burchard H, Feistel R, et al. (IOW, ~2009) State-of-the-art review on vertical mixing in the Baltic Sea and consequences for eutrophication ([PDF](https://balticsea2020.se/english/images/Bilagor/vertical_mixing.pdf)) | halocline Kz (Baltic Proper) | [F] (grep-read of the relevant sections) |
| 16 | SMHI cruise reports (e.g. [exp_0513eng](https://www.smhi.se/download/18.38e7941719209b36a1fbd53/1728368634963/exp_0513eng.pdf)) | Kattegat winter DIN 5-6 umol/l | [S] |
| 17 | Mackinson S, Daskalov G (2007) An ecosystem model of the North Sea to support an ecosystem approach to fisheries management. Cefas Sci. Ser. Tech. Rep. 142 ([PDF](https://www.cefas.co.uk/publications/techrep/tech142.pdf)) | phytoplankton C:WW 1:10 | [F] (section 7 and Appendix 2) |
| 18 | Lund-Hansen LC, Pejrup M, Floderus S (2004) Pelagic and seabed fluxes of particulate matter and carbon, and C:N ratios resolved by sediment traps during a spring bloom, southwest Kattegat. J. Sea Res. 52:87-98. doi:10.1016/j.seares.2003.11.003 | regional C:N | [A] |
| 19 | Lomas MW, Baer SE, Acton S, Krause JW (2019) Pumped up by the cold: elemental quotas and stoichiometry of cold-water diatoms. Front. Mar. Sci. 6:286. doi:10.3389/fmars.2019.00286 | Redfield C:N 6.6 in diatoms | [F] (text; tables are images) |
| 20 | Menden-Deuer S, Lessard EJ (2000) Carbon to volume relationships for dinoflagellates, diatoms, and other protist plankton. Limnol. Oceanogr. 45:569-579. doi:10.4319/lo.2000.45.3.0569 | cell C per volume | [S] (exponents 0.81/0.94 in a secondary snippet; coefficients from memory, unverified) |

## Q1. Half-saturation constant for nitrate / DIN uptake

### Literature
- **NPZ convention, 0.5 mmol N m-3.** Fennel et al. 2006, Table 1: "kNO3 half-saturation concentration for uptake of NO3 0.5 mmol N m-3, range 0.007-1.5"; kNH4 also 0.5. The range is quoted from Lima & Doney 2004 (second-hand). [F]
- **Evans & Parslow 1985.** Behrenfeld & Boss 2014 suppl. table: "J - nitrogen uptake half saturation 0.5 [mmole m-3] Evans & Parslow, 1985"; "N0 - nitrogen at depth 10 [mmole m-3]". [F] of BB14; the 1985 paper was not read.
- **Fasham, Ducklow & McKelvie 1990 (k1).** NOT retrieved. All copies were blocked (see Failed fetches). Fennel 2006 descends directly from the Fasham model and is the fallback: 0.5. I have not filled this gap from memory.
- **Field kinetics.** MacIsaac & Dugdale 1972: "measurements in eutrophic waters ... indicate a ks of about 1.0 ug-atoms/l can be assumed for both nitrate and ammonium"; four coastal Peru NH4 values "0.66-2.01, with a mean of 1.11"; in oligotrophic waters "A ks of 0.1 ug-atoms/l was used as a best approximation". [F]
- **Regional model (North Sea / Skagerrak lineage).** In NORWECOM (Samuelsen et al. 2015), the half-saturation constants are "smaller for flagellates (KN = 1.5 mmol m-3 ...) than for diatoms (KN = 2.0 mmol m-3 ...)". NORWECOM V2.0 "was primarily applied to the North Sea". [F]
- **Other NPZ values.** Levy 2015: Kn = 0.7 mmol m-3 [F]. Zheng et al. 2025 adopt Ks = 0.7, citing "the range of 0.45-0.91 mmol N m-3" from Hurtt & Armstrong 1999 [F of Zheng]. Franks et al. 1986: ks = 1 umol N/l [S, Wikipedia]. A secondary compilation puts marine-diatom nitrate K at 0.45-1.87 umol/l [S].
- **Regional nutrient context.** Kattegat winter DIN is about 5-6 umol/l, and surface nitrate falls below 0.1 umol/l in summer (SMHI cruise reports) [S].

### Inference
- Published values span 0.1 (oligotrophic) to 2.0 (diatoms in the regional model). Generic NPZ models use 0.5. Eutrophic field kinetics give about 1.0.
- Winter DIN is only ~5-6 umol/l, so K = 0.5-2 sets how sharply the bloom stops, not whether it starts. At K = 1 and N = 5, limitation is N/(K+N) = 0.83; at K = 2 it is 0.71 [D].
- **Recommended: 1.0 umol N/l (range 0.5-2.0).** The NPZ value 0.5 is the lower end. 2.0 is the North Sea model's diatom value.

## Q2. Fate of nitrogen grazed by copepods

### Literature
- **N budget of a feeding copepod.** Kuijper et al. 2004, the DEB model fitted to Kiorboe's Acartia data:
  - The fraction of ingested N "allocated to pellets is constant at 0.312".
  - Excretion compares "favourably with fractions of 0.15 respired and 0.25 excreted observed by Kiorboe et al. (1985) for A. tonsa feeding on Rhodomonas baltica ... (at high food concentration)".
  - On egg production: "the low KN of 0.4 that is seen in egg production experiments".
  - "zooplankton are unable to utilize dietary N efficiently for egg production ... because of the N demands for maintenance". So the N excreted share rises at low ration.
  - [F] of Kuijper; Kiorboe 1985 not read.
- **Carbon GGE.** Straile 1997: mean 26 %, median 22 % (project note, source 13). Kuijper 2004 says copepod carbon GGE is "typically low, between 0.2 and 0.3" [F].
- **Dissolved release pathways (Acartia).** "excretion and sloppy feeding were the dominant modes of DOC and NH4+ release", and "a large proportion of ingested PON was lost as dissolved NH4+ and urea". I did not obtain the percentages. [S] (Saba 2011 and the thesis behind it)
- **Kattegat pellet fate.** Lundsgaard & Olesen 1997, southern Kattegat 1989: "Copepod defecation nearly equaled the sedimentation of fecal pellets". So there was little retention of pellets in the pelagic. Pellets were a minor share of sedimenting matter, about 82 % of which was other detritus. [A]
- **NPZ formulations.**
  - Fennel 2006 [F]:
    - Assimilation efficiency beta = 0.75; the unassimilated (1 - beta) goes to small detritus.
    - Excretion to NH4: basal lBM = 0.1 d-1 plus assimilation-related lE = 0.1 d-1.
    - Detritus remineralises at rSD = 0.03 d-1 and sinks at 0.1 m/d.
  - Fasham 1990, as quoted by Steele 1998: "The ratio of growth of Z to grazing on P ... can be high (0.75 in Fasham et al. (1990)) ... or low for oligotrophic situations (0.2 in Steele & Henderson (1993))". [F] of Steele.
  - Evans & Parslow (via BB14): "c3 - ingestion efficiency 0.5". In the BB14 equations the non-assimilated grazing does not return to N; it leaves the system. [F] of BB14.
  - Levy 2015: assimilated fraction az = 0.7, excretion tau_z = 5.8e-7 s-1 (0.050 d-1 [D]), export ratio fe = 0.2. [F]
  - NORWECOM: assimilation efficiency 0.75 for both zooplankton classes (Patsch et al. 2009). [F]
    - Its "90 % ... detritus ... 10 % returned to nitrate" applies to zooplankton *loss (mortality)* terms, not to excretion.
  - Franks 1986: growth efficiency gamma = 0.3, with the remaining 0.7 of the ration returned to nutrients immediately [S, Wikipedia].

### Calculation [D]
- Acartia at high food, ingested N, assembled from Kuijper 2004 (not one measured budget):

  | Fate | Share | Basis |
  |---|---|---|
  | Eggs (adult females) | ~0.4 | egg-production KN, Kiorboe data; not somatic growth |
  | Pellets | ~0.31 | DEB model output (constant), not a measurement |
  | Excreted, dissolved | ~0.25 | observed, Kiorboe et al. 1985 |
  | Total | ~0.96 | mixed bases, indicative only |

- Pellet residence in the mixed layer:
  - Assumption, not sourced here: copepod pellets sink at tens of m/d.
  - At that speed a pellet leaves a 6-17 m mixed layer within hours to a day, which matches the Kattegat observation (defecation ~ pellet sedimentation).
  - So on the time scale of the mixed layer, the pellet share counts as export, not recycling.

### Inference
- Fraction of grazed N returned quickly to the dissolved pool: **central 0.3, range 0.25-0.45.**
  - The low end is the measured excretion at high food.
  - The upper end allows for sloppy-feeding DON/NH4 (Saba 2011, unquantified here) and the higher excretion share at low ration (Kuijper).
- The rest splits roughly equally:
  - pellets that sink out (~0.3), which the model should treat as export below the halocline;
  - copepod growth (~0.3-0.4).
- **Two NPZ conventions, contrasted.**
  - Fasham/Fennel: 0.75 assimilated, 0.25 to detritus; excretion runs as separate per-day rates.
  - Franks: 0.7 returned instantly.
  - A one-fraction model of the Franks type that sets 0.7 overstates fast recycling about twofold relative to the copepod budget, given the Kattegat pellet evidence.

## Q3. Cross-pycnocline exchange m and the regional vertical N flux

### Literature
- **Evans & Parslow 1985 m.**
  - BB14 suppl.: "m - diffusion velocity 3.0 [m day-1] Evans & Parslow, 1985". The same run uses "MLD = 75 + 50 x cos(...)", i.e. mixed layers of 25-125 m. [F] of BB14.
  - The UVic E&P page default is "Vertical Mixing Rate (m/day): 3.0 m/day". [F] of page.
- **Fasham 1990 m.** Zheng et al. 2025: "previously used fixed coefficients, such as 0.1 m d-1 (Fasham et al., 1990) and 0.5 m d-1 (Macias et al., 2007)". They chose a fraction of MLD giving "a time-mean mu_m * z_m of 0.3 m d-1". [F] of Zheng.
- **Regional entrainment flux (Kattegat).** Hasager et al. 2003 [F]:
  - Fluxes come from a "basin-scale hydrodynamic model" with bulk mixed-layer physics (Gustafsson 2000) and a tracer-weighted salinity-nutrient relation. The authors call the result "an order of magnitude estimate of the entrainment fluxes".
  - Table 3, summer (May-August) means 1989-1999, in mg N m-2 d-1:

    | Basin | Mean | Interannual range |
    |---|---|---|
    | north | 8.3 | 0.2-22.5 |
    | central | 1.6 | 0.1-4.5 |
    | south | 1.7 | |

  - "The Northern Kattegat has much higher fluxes due to the influx of nutrient-rich water from the Skagerrak."
  - One-year return level: "around 30 mg N m-2 day-1 in the Central and Southern part ... (170 mg N m-2 day-1) ... in the Northern part". The 10-year central value is ~110.
  - The summer input from atmosphere plus entrainment is "efficiently digested", leaving the surface depleted.
- **Halocline diffusivity (analogue: Baltic Proper, not Kattegat).** Reissmann et al. [F]:
  - Turbulence measurements give "eddy diffusivities of the order of 10-5 m2s-1", and salt budgets give 10-4.
  - In the halocline, "the absolute minimum ... diapycnal diffusivity ... is 2 x 10-6 m2/s"; models give "2-3 x 10-6 m2/s in the halocline".
  - Stigebrandt's deep-water parameterisation is "Kv = 0.87*10-7 m2/s2 N-1".
  - The Arkona/Drogden wE values in that review are for dense bottom plumes, which is the wrong setting for this site.
- No Kattegat/Skagerrak microstructure Kz was found (Richardson et al. 2000, JPR 22:1663, is paywalled).

### Calculation [D]
- Hasager fluxes in mmol:
  - Summer means: north 0.59, central 0.11, south 0.12 mmol N m-2 d-1.
  - North interannual range: 0.014-1.6.
  - One-year return: ~2.1 (central/south), 12.4 (north). 10-year return (central): 7.9.
- **m = F/dN** (deep minus surface DIN). dN is assumed at 5 and 10 umol/l; no below-halocline summer DIN was retrieved.
  - North: m = 0.06-0.12 m/d.
  - Central: m = 0.011-0.023 m/d.
- **m = Kz/dz** for a halocline 5-10 m thick:
  - Kz 2e-6: m = 0.017-0.035 m/d.
  - Kz 1e-5: m = 0.09-0.17 m/d.
  - Stigebrandt's Kv applied to a Kattegat halocline (drho ~10 kg/m3 over 5-10 m, N = 0.10-0.14 s-1) gives Kv = 6-9e-7 m2/s and m = 0.008-0.011 m/d. That formula was derived for deep water, so it is only a lower bound.
- Implied exchange rates m/h for h = 6-17 m:

  | Model | m | m/h |
  |---|---|---|
  | Evans & Parslow | 3.0 m/d | 0.18-0.5 d-1 |
  | Fasham | 0.1 m/d | 0.006-0.017 d-1 |

- E&P's m with dN = 5 umol/l would supply 15 mmol N m-2 d-1 (210 mg N), about 25 times Hasager's northern summer mean. It would prevent the observed summer depletion.

### Inference
- The flux route and the diffusivity route agree on **m ~ 0.01-0.2 m/d**. The E&P value of 3 m/d belongs to a deep open-ocean mixed layer (25-125 m), not a 6-17 m layer capped by a halocline.
- **Recommended: m = 0.05 m/d (range 0.01-0.2).** The site sits on the Skagerrak side of the Kattegat-Skagerrak front, so the northern-basin flux is the closest analogue: ~0.6 mmol N m-2 d-1 in summer, with storm events of 2-12 mmol m-2 d-1.
- Episodic wind-driven entrainment dominates the real flux (the extremes rise steeply with return period). A constant m reproduces the mean, not the pulses.

## Q4. Phytoplankton carbon : wet weight and Redfield C:N

### Literature
- **North Sea Ecopath convention.** Mackinson & Daskalov 2007 [F]:
  - Appendix 2: "Carbon (g) to Wet weight (g) (Phytoplankton) 1 to 10 based on 0.1gC=0.2g dry weight = 1 g wet weight (Mathews and Heimdal 1980 - from Mackinson 1995)".
  - Section 7 applies it: "750 mgC m-2 (7.5 g wet wt m-2 using the above conversion)".
  - An alternative: "conversion factors used by Christensen (1995) (1 g C = 15 kcal; 1 g wet wt = 1.3 kcal ...)". The text calls the conversion "an important source of uncertainty".
  - The same appendix also lists "Carbon to dry organic matter (Phytoplankton) 1 to 5.4". That contradicts the 0.1 gC = 0.2 g DW (1:2) behind the 1:10.
- **Cell-level carbon density.** Menden-Deuer & Lessard 2000: diatoms differ from other protists, with exponents ~0.81 (diatoms) and ~0.94 (others) [S]. Lomas et al. 2019 report that cold-water diatoms carry more C per biovolume than the M-D&L fits [F].
- **Redfield C:N.** Lomas et al. 2019: POC:PON in their diatoms was "not statistically different than the canonical Redfield Ratio (C:N = 6.6)" [F].
- **Kattegat field C:N.** Lund-Hansen et al. 2004 (SW Kattegat, spring bloom): "C:N ratio of material collected in the 8 m trap was on average 6.9, close to the Redfield ratio (6.7)", with seabed traps at 8.8 [A].

### Calculation [D]
- Christensen route: 15/1.3 = 11.5 g WW per g C (C:WW = 0.087).
- Menden-Deuer & Lessard fits, taking cell density as 1 pg/um3. The coefficients 0.288 and 0.216 are unverified (from memory):

  | Cells | 100-10^5 um3 | 1000 um3 |
  |---|---|---|
  | Diatoms | C:WW 0.12-0.033 | 0.078 |
  | Other protists | C:WW 0.16-0.11 | 0.14 |

  This agrees with the project note's ~0.03-0.16.
- Redfield: 106:16 = 6.625 mol C/mol N = 5.68 g C/g N. So 1 mmol N = 79.6 mg C = 0.80 g WW at 1:10.
- In model units, 1 umol N/l over a 10 m mixed layer = 10 mmol N m-2 = 8.0 g WW m-2 = 8.0 t WW/km2.

### Inference
- **Recommended: C:WW = 0.10 g C per g WW (10 g WW/g C), range 0.05-0.15.** Use 0.05-0.08 for large spring diatoms and up to ~0.15 for small flagellates. The 1:10 matches the North Sea Ecopath model and the convention already used in `Zooplankton energibudget/copepod_energetics.md`.
- **Redfield: 6.625 mol/mol (5.68 g C/g N).** Kattegat sinking material (6.9) is consistent with it.

## Failed fetches and sources not read
- Fasham, Ducklow & McKelvie 1990, J. Mar. Res. 48:591-639:
  - eLischolar PDF was Cloudflare-blocked by curl and returned 403 to WebFetch, at two URL forms.
  - The GLOBEC and US-GLOBEC pages carry no parameter values.
  - The myroms.org bioFasham.in wiki returned 403.
  - CSDMS and NOAA bio_Fasham.in returned 404.
  - The OceanRep copy of Oschlies & Garcon 1999 hit a bot challenge.
- Evans & Parslow 1985 (Biol. Oceanogr. 3:327-347): original not read. Smith et al. 2015 (JGR, Wiley) returned 403. Jones, Parslow & Murray 2010: abstract record only.
- Eppley, Rogers & McCarthy 1969 (L&O 14:912): Wiley returned 403. Sarthou et al. 2005: HAL Anubis block.
- Saba et al. 2011 and the VIMS thesis: scholarworks.wm.edu returned 403 for both.
- core.ac.uk (Kattegat entrainment model record) returned 403. int-res.com returned 429. Richardson, Visser & Pedersen 2000: paywalled, abstract only.
- Menden-Deuer & Lessard 2000: not read (coefficients unverified). Pauly & Christensen 1995: the fishbase page has no C:WW conversion.
- No summer below-halocline DIN for the Kattegat or Skagerrak was retrieved, so dN in Q3 is an assumption.
