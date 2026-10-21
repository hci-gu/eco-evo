# Density-dependent natural mortality for the Mareld `pelagic_fish` pool: how strong should dM1/d ln B be?

Scope note: about 75 tool calls (searches, PDF downloads converted with pdftotext, Crossref abstract lookups, page-summarising fetches) plus local reads of `fgconfig/fg_library.yaml`, `mareld2.yaml` and `lib/environments/ecosystem_env/{predation,interactions,state}.py`, all read-only. No code or config was edited.

**How each source was read** (also in the table at the end):
- **fulltext**: PDF downloaded and searched or read.
- **abstract**: publisher or Crossref abstract.
- **snippet**: search-engine summary only. Numbers from snippets are *not verified*.
- **secondary**: a number quoted in another paper that I did read.

"(computed)" marks my own arithmetic; those numbers are not in any source. Sections use Literature / Calculation / Inference sub-headings.

**Definitions used throughout** (from the brief):
- beta_M = dM1/d ln B at B_ref, per year.
- gamma = beta_M / M1(B_ref), dimensionless. Example form M1 = M1_ref (B/B_ref)^gamma, or linear M0 + kB with gamma = k B_ref / (M0 + k B_ref).
- M1_ref = 0.58 /yr, B_ref = 12000 t / 3600 km2 = 3.33 t/km2. Z_pool = M1 + explicit M2 + F = 0.58 + 0.30 (gadoids) + 0.012 (porpoises) + 0.10 = 0.99 /yr. The M2 values are the `predation_mortality` entries in `fg_library.yaml` lines 443-480.

**Not verified / not reached:**
- **Myers, Bowen & Barrowman 1999.** The full text was not reached: the SEDAR mirror returned 404. Species alphas come second-hand from Froese & Luna 2004 Table 1, and the definitions from the 1998 NAFO working-paper version, read in full.
- **Liermann & Hilborn 1997.** PDF fetch failed (file not created). Its result is cited via Hilborn et al. 2014, read in full.
- **Nash, Dickey-Collas & Kell 2009.** Only a search snippet of the abstract plus the summary in Dickey-Collas et al. 2010 (fulltext); Crossref had no abstract.
- **Casini et al. 2006 (Oikos).** Bibliographic record only.
- **Andersen et al. 2017 (Fish Fish).** The DOI I tried resolved to another paper; not used.
- **Melvin & Stephenson 2007.** Not searched; cited only through Dickey-Collas et al. 2010.
- **Patterson 1996 (Ichthyophonus).** Not reached.
- **Kotterba et al. 2014 (stickleback egg predation).** The AFS abstract page failed (DNS error); snippet-level only.
- **Thorson 2020 (FishLife).** Clupeidae- and Ammodytidae-specific steepness was not in the retrieved text; only Scombridae (0.69) and the Actinopterygii mean (0.74) are given.
- **Unread confirmations.** No source gives a sandeel steepness. I found no herring-specific abundance-area (basin model) estimate.

---

## Q0. Conversion machinery

### Calculation

**(a) From gamma to the model's parameters.**
- Power form M1 = 0.58 (B/B_ref)^gamma: beta_M = 0.58 gamma.
- Linear form M0 + kB: k = gamma M1_ref / B_ref and M0 = M1_ref (1 - gamma). **(computed)**
  - gamma 0.25: k = 0.044 /yr per t/km2, M0 = 0.435.
  - gamma 0.5: k = 0.087, M0 = 0.29.
  - gamma 1.0: k = 0.174, M0 = 0. This is the largest gamma the linear form allows with a non-negative floor.
- Per tick (6 h, 1460 ticks/yr), divide k and M0 by 1460.

**(b) Recruitment compensation to an equivalent loss on a biomass pool (the inference step the brief asks for).**

*Step 1, Beverton-Holt in steepness form.* R(S) = 4 h R0 S / (S0 (1 - h) + S (5h - 1)), i.e. R = alpha S / (1 + beta S) with beta S0 = (5h - 1)/(1 - h).

*Step 2, per-capita recruitment elasticity at depletion d = S/S0.* e(d, h) = d ln(R/S) / d ln S = -beta S / (1 + beta S), with beta S = d (5h - 1)/(1 - h). e ranges over (-1, 0):
- e -> -1 is full compensation (R flat in S);
- e = 0 is no compensation (R proportional to S).

*Step 3, link to the pool.* In a pool without age structure, gross per-capita production g equals per-capita loss Z at equilibrium. If production scales with recruits (fixed growth and mortality schedule), then:
- d ln g / d ln B = e(d, h);
- d g / d ln B = e Z_eff.

Expressed as a density-dependent loss with production held at its B_ref value: beta_M,equiv = Z_eff |e(d, h)|.

*Step 4, what Z_eff is.* Write ln B = ln BPR(F) + ln R(B), with BPR the biomass per recruit. Then d ln B / dF = (d ln BPR / dF) / (-e), so -dF / d ln B = |e| / |d ln BPR / dF|. Hence Z_eff = 1 / |d ln BPR / dF|.
- **Upper bound.** For a pool where recruits enter at full weight with no delay, BPR is proportional to 1/Z, so Z_eff = Z = 0.99 /yr.
- **Real stocks.** Growth and maturation delays make BPR fall faster with F, so Z_eff < Z.

*Step 5, calibration by the MSY identity.* For any single pool with per-capita net production p(B), maximising F·B requires d(B p)/dB = 0, i.e. dp/d ln B = -p = -F_MSY at B_MSY. This is exact and model-free. So **beta_total(B_MSY) = F_MSY**, which gives Z_eff = F_MSY / |e(d_MSY, h)|. **(computed)**
- herring (h 0.67, d_MSY 0.26 from Thorson 2012 Clupeiformes, F_MSY 0.23-0.31): Z_eff = 0.35-0.48;
- mackerel (h 0.85, d_MSY 0.30 assumed, F_MSY 0.19): 0.22;
- horse mackerel (h 0.75, F_MSY 0.15 *assumed, not verified*): 0.20.

**(c) Functional-response elasticities, for the sign of predation mortality on the prey.** Per-capita predation mortality on prey B, at fixed predator biomass P:
- **Type II:** m = aP / (1 + ahB + I), so d ln m / d ln B = -ahB / (1 + ahB + I), always in (-1, 0). This is depensatory.
- **Type III:** m = aPB / (1 + ahB^2 + I), so d ln m / d ln B = 1 - 2ahB^2 / (1 + ahB^2 + I). This is compensatory below B* = sqrt((1+I)/(ah)) and depensatory above it, tending to -1.

These are the forms in `interactions.holling_a_eff` (lines 284-310), where I = w B_pred is the Beddington-DeAngelis interference term.

---

## Q1. Stock-recruitment and compensation, species by species

### Takeaway
Every component shows compensation in some analyses. Its strength at *today's* biomass, however, depends almost entirely on where each stock sits relative to its S-R breakpoint, and today the components sit on different sides.

ICES fits segmented ("hockey-stick") S-R relations, and that form gives a binary answer:
- **Above the breakpoint, e = -1** (recruitment flat in S): NSAS herring at 1.8x Blim, sprat, and marginally mackerel.
- **Below the breakpoint, e = 0** (R proportional to S): WBSS herring at 0.63x Blim and sandeel area 3r.

The Myers et al. reproductive rates convert to steepness of about 0.61-0.74 for herring, 0.49-0.57 for sprat, 0.81-0.89 for mackerel and 0.70-0.82 for horse mackerel **(computed)**. Clupeiformes have the lowest B_MSY/B0 of any order (0.26; Thorson et al. 2012), i.e. strong compensation at the taxon level.

Sandeel recruitment is regulated mainly by the *age-1* cohort rather than by SSB: an over-compensatory cohort interaction.

Across all stocks, recruitment is unrelated to SSB over the observed range in 61 % of 224 stocks (Szuwalski et al. 2015). For a pool model this is *strong* compensation (e near -1), not an argument for gamma = 0.

### Literature

**Myers et al. 1999 and steepness**
- Myers et al. 1999 (CJFAS 56:2404): the maximum annual reproductive rate "is relatively constant within species and ... there is relatively little variation among species"; it "can be interpreted as a standardized slope at the origin of a spawner-recruitment function". (abstract, Crossref) - [doi:10.1139/f99-201](https://doi.org/10.1139/f99-201)
- Myers et al. 1998, NAFO SCR 98/5, the working-paper version:
  - Definitions: alpha_hat = alpha · SPR(F=0), "the number of spawners produced by each spawner over its lifetime at very low spawner abundance". The annual rate is alpha_tilde = alpha_hat (1 - p_s), with p_s the adult survival (Eq. 4).
  - "The maximum annual reproductive rate for herring appears to be slightly less" than the 3-4 found for cod.
  - "herring has a smaller maximum reproductive rate than many species. The lower mean is due to a few stocks in the northern north Atlantic which have been reduced to very low levels (the Iceland stocks, the Norway ... stock, and the Georges Bank stocks)".
  
  (fulltext) - [NAFO SCR Doc. 98/5](https://www.nafo.int/Portals/0/PDFs/sc/1998/scr-98-005.pdf)
- Froese & Luna 2004, Table 1, reproducing the Myers et al. 1999 annual alphas: *Clupea harengus* 2.08, *Sprattus sprattus* 2.39, *Scomber scombrus* 3.03, *Trachurus trachurus* 1.68. *Ammodytes* is not listed. (fulltext of Froese & Luna; the alphas are **secondary** for Myers 1999) - [Acta Ichthyol Piscat 34:11-20](https://www.fishbase.de/rfroese/NoFecundRel.pdf)
- A SEDAR document gives the Beverton-Holt conversion h = alpha_hat/(4 + alpha_hat). (snippet, **secondary**; it is the standard identity, which I re-derived: h = R(0.2 S0)/R0 with slope at origin alpha_hat R0/S0) - [SEDAR S8RD21 record](https://www.fisheries.noaa.gov/inport/item/60041)
- Thorson 2020 (FishLife): the predictive mode of steepness for Scombridae is 0.69, and the bony-fish (Actinopterygii) mean is 0.74. The text also says Scombridae have "elevated values of natural mortality and steepness". These two statements sit uneasily together: the 0.69 mode is below Sebastidae's 0.72. No Clupeidae or Ammodytidae value was found in the retrieved text. (fulltext, NOAA repository copy) - [doi:10.1111/faf.12427](https://repository.library.noaa.gov/view/noaa/54871)

**Thorson et al. 2012 (taxon-level surplus production)**
- Clupeiformes SB_MSY/SB0 = 0.261 (SD 0.097, SE 0.040), against 0.404 pooled over 147 stocks.
- The Pella-Tomlinson shape for Clupeiformes is 0.599 (SE 0.131), "lower than the Fox model". The authors call this "unsurprising for Clupeiformes because species from this taxonomic order often have high variability in abundance and appear to be capable of rapid population recovery".
- The authors warn that this parameterisation "has positive surplus production even when total abundance is at zero when [shape] < 1 (i.e., SB_MSY/SB0 < 0.37)".

(fulltext) - [doi:10.1139/f2012-077](https://jensen.limnology.wisc.edu/wp-content/uploads/sites/147/2021/11/Thorson_etal_2012b_CJFAS.pdf)

**North Sea autumn-spawning herring (NSAS)**
- Dickey-Collas et al. 2010: "Nash et al. (2009a) did not find any evidence of depensatory effects ... However, compensatory mechanisms (i.e. a negative correlation between number of recruits per spawner and stock size ...) were evident. Compensation was stronger during the period of recovery than during the period of collapse." Year-class strength is set "during the overwintering phase of the larvae (between the early and late stages: 10-30 mm)". "Stock-recruitment relationships are no longer being considered as stationary". (fulltext) - [doi:10.1093/icesjms/fsq033](https://www.vliz.be/imisdocs/publications/ocrd/256670.pdf)
- Nash, Dickey-Collas & Kell 2009: compensation "was stronger after the collapse of the stock", and arose from "both increased production of larvae per spawner and increased survival to the juvenile stage". There is "only slight evidence for depensation", with the zero-recruitment point "close to the origin". (snippet only, **not verified**) - [doi:10.1016/j.fishres.2008.08.003](https://doi.org/10.1016/j.fishres.2008.08.003)
- Nash & Dickey-Collas 2005: "In 80% of the years, year class strength reflected SSB"; aberrant year classes are determined "between the pelagic larval and the juvenile stages". (abstract, Crossref) - [doi:10.1111/j.1365-2419.2005.00336.x](https://doi.org/10.1111/j.1365-2419.2005.00336.x)
- Payne et al. 2009: "an unprecedented 6 sequential years of poor juvenile production" (2002-2007) "despite high biomass levels"; "the poor recruitment arises during the larval overwintering phase". (fulltext) - [doi:10.1093/icesjms/fsn211](https://oceanrep.geomar.de/id/eprint/2933/1/538_Payne_2009_RecruitmentInAChangingEnvironment_Artzeit_pubid11565.pdf)
- ICES 2025 advice her.27.3a47d:
  - Blim = 828 874 t, the "Breakpoint in the segmented regression of the stock-recruitment time-series (1947-2023, excluding the recovery period 1979-1990)".
  - SSB 2024 about 1.51 Mt.
  - F_MSY revised "from 0.32 to 0.23" after the MSE.
  - "The stock has been in a low productivity regime since 2002".
  
  (fulltext) - [ICES advice, doi:10.17895/ices.advice.27202626](https://ices-library.figshare.com/articles/report/Herring_i_Clupea_harengus_i_in_Subarea_4_and_divisions_3_a_and_7_d_autumn_spawners_North_Sea_Skagerrak_and_Kattegat_eastern_English_Channel_/27202626)

**Western Baltic spring spawners (WBSS)**
- ICES 2025 advice her.27.20-24:
  - Blim = 120 000 t, "the mean of the two lowest SSB (1998, 1999) values with above-average recruitment".
  - F_MSY 0.31 from EqSim "with Beverton-Holt, Ricker, and segmented regression".
  - SSB 2026 = 75 781 t with zero catch; "spawning-stock size is below MSY Btrigger, BPA, and Blim"; "Climate change-related effects have resulted in lower early life-stage survival and reduced productivity of the Rugen spawning component".
  
  (fulltext) - [ICES advice, doi:10.17895/ices.advice.27202614](https://ices-library.figshare.com/articles/report/Herring_i_Clupea_harengus_i_in_subdivisions_20_24_spring_spawners_Skagerrak_Kattegat_and_western_Baltic_/27202614/1)

**Sandeel**
- Arnott & Ruxton 2002, North Sea 1983-1999, Ricker-type fits of ln(N0/S):
  - With SSB alone, a = -0.0010 per kt (p < 0.05).
  - With the age-1 term, ln(N0/S) = -0.0016 S - 0.7175 ln(N1 + 1) + 5.008. All terms are significant, and the age-1 coefficient is negative in all three regions (p = 0.007, 0.026, 0.025).
  - Adding the NAO gives a = -0.0019 and b = -0.8002.
  - "Relationships between recruitment and spawning stock biomass were weak, but Age-1 sandeels (non-spawners) were related negatively with recruitment"; "the adults are non-migratory".
  
  (fulltext) - [doi:10.3354/meps238199](https://www.int-res.com/articles/meps2002/238/m238p199.pdf)
- van Deurs et al. 2009:
  - "A significant positive linear relationship (r = 0.794, p = 0.019) existed between N0 and SSB when age-1 abundance was low ... while at high numbers ... no relationship was found."
  - The SSB x N1 interaction model explains 30 % (65 % with *C. finmarchicus*), against 12 % for a Ricker.
  - "Intra-specific predation on eggs and larvae by lesser sandeels may be insignificant"; they favour local food competition by age-1 fish.
  
  (fulltext) - [doi:10.3354/meps07960](https://int-res.com/articles/meps_oa/m381p249.pdf)
- ICES 2026 advice san.sa.3r (Skagerrak, northern and central North Sea): "SSB is below BPA, Blim, and MSY Bescapement"; Blim = 72 713 t; zero catch advised. (fulltext) - [ICES advice](https://ndownloader.figshare.com/files/62226832)
- Stock annex: SMS uses "a hockey stick stock recruitment function, which is scaled to have little influence on the likelihood"; the forecast uses "None. Instead the long-term geometric mean recruitment". (fulltext) - [ICES stock annex san.sa.3r](https://ndownloader.figshare.com/files/56798417)

**Sprat**
- ICES 2026 advice spr.27.3a4: empirical Blim = 77 157 t, "mean of the three lowest spawning-stock biomass (SSB) values ... with above[-average recruitment]"; SSB "above MSY Bescapement, BPA, and Blim"; Fcap 0.71. "A high proportion of the predicted SSB consists of recruits from the previous year". (fulltext) - [ICES advice](https://ndownloader.figshare.com/files/64136503)

**Mackerel**
- ICES 2026 advice mac.27.nea: SSB "below MSY Btrigger and BPA and above Blim"; Blim = 3 067 017 t, based on the SSB that "in the past has resulted in good recruitment"; F_MSY = 0.191. (fulltext) - [ICES advice](https://ndownloader.figshare.com/files/69489192)

**Cross-stock**
- Szuwalski et al. 2015: "recruitment and spawning biomass are not positively related over the observed range of stock sizes for 61% of 224 stocks"; "85% of stocks for which spawning biomass may not drive recruitment ... exhibit shifts in average recruitment". (abstract, Crossref) - [doi:10.1111/faf.12083](https://doi.org/10.1111/faf.12083)
- Vert-pre et al. 2013 (230 stocks, Fox surplus production):
  - Production is best explained by abundance in 18.3 %, regimes in 38.6 %, mixed in 30.5 % and random in 12.6 %.
  - Bias-corrected values are given inconsistently between the table and the text (24/27 % and 27/24 %).
  - No taxon breakdown is given.
  
  (fulltext via page-summarising fetch; weaker) - [doi:10.1073/pnas.1214879110](https://pmc.ncbi.nlm.nih.gov/articles/PMC3562848)
- Zimmermann et al. 2018: density-dependent recruitment was detected in 70 % of 70 NE Atlantic stocks; "97% of populations with statistically significant parameters for growth and recruitment showed a larger impact of density-dependent recruitment on population biomass". (fulltext, as read for the sibling gadoid note `../Täthetsberoende och interferens hos torskfiskar/gadoid_density_dependent_growth.md`; **secondary here**) - [doi:10.1111/1365-2656.12800](https://doi.org/10.1111/1365-2656.12800)

### Calculation

**Steepness from Myers alphas**, with alpha_hat = alpha_tilde/(1 - e^-M_adult) and h = alpha_hat/(4 + alpha_hat) **(computed)**. The adult M values are my assumptions, not taken from sources.

| Species | alpha_tilde (Myers) | Adult M assumed | alpha_hat | h |
|---|---|---|---|---|
| Herring | 2.08 | 0.2 / 0.3 / 0.4 | 11.5 / 8.0 / 6.3 | 0.74 / 0.67 / 0.61 |
| Sprat | 2.39 | 0.6 / 0.8 / 1.0 | 5.3 / 4.3 / 3.8 | 0.57 / 0.52 / 0.49 |
| Mackerel | 3.03 | 0.10 / 0.15 / 0.20 | 31.8 / 21.8 / 16.7 | 0.89 / 0.85 / 0.81 |
| Horse mackerel | 1.68 | 0.10 / 0.15 / 0.20 | 17.7 / 12.1 / 9.3 | 0.82 / 0.75 / 0.70 |

Myers' own caveat applies to herring: the low alpha is pulled down by collapsed northern stocks, so 0.67 is a cautious central value. Myers et al. estimated alpha from *Ricker* fits; converting it with the Beverton-Holt identity h = alpha_hat/(4 + alpha_hat) is my choice and approximate. ICES F_MSY and Fcap are fishing mortalities over an age range, not catch over total biomass, so the MSY-identity calibrations here are approximate.

**Per-capita recruitment elasticity e(d, h)** **(computed)**:

| h \ d | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|---|
| 0.50 | -0.23 | -0.38 | -0.47 | -0.60 | -0.71 |
| 0.67 | -0.42 | -0.59 | -0.68 | -0.78 | -0.85 |
| 0.75 | -0.52 | -0.69 | -0.77 | -0.85 | -0.90 |
| 0.85 | -0.68 | -0.81 | -0.87 | -0.92 | -0.95 |

**Sandeel, Arnott & Ruxton.** ln N0 = ln S + aS + b ln N1 + k. If S and N1 both scale with the pool, e = aS + b **(computed)**:
- at S = 300 / 500 / 800 / 1000 kt: e = -1.20 / -1.52 / -2.00 / -2.32 (a = -0.0016, b = -0.72);
- with the NAO model: -1.37 / -1.75 / -2.32 / -2.70.

e < -1 means recruitment *falls* as the stock rises (over-compensation). The 1983-1999 North Sea SSB range was not retrieved, so which S applies is open.

**Segmented regression** (the ICES S-R form): e = -1 above the breakpoint and e = 0 below it.

### Inference
- **Steepness picture.** The BH-steepness picture (h 0.5-0.85, e -0.4 to -0.9 at plausible depletions) and the hockey-stick picture agree that compensation is strong *above* the breakpoint. They disagree only for depleted components: the hockey stick says none, BH says moderate.
- **Which components are depleted.** In the Skagerrak, the two depleted components (WBSS herring, sandeel 3r) carry a large share of the pool.
- **Over-compensation.** It is real in sandeel, but it is a cohort-interaction effect (age-1 versus age-0), not a function of total biomass. A pool term can only approximate it, capped at |e| = 1.
- **Environment versus SSB.** "Recruitment unrelated to SSB" (Szuwalski; Payne's 2002-2007 failure at a 40-year SSB high) does *not* support gamma = 0 in a pool model. If mean recruitment is independent of S, per-capita recruitment is proportional to 1/S, which is e = -1. What environmental forcing argues against is a *stationary* curve, i.e. a fixed gamma with a fixed B_ref.

---

## Q2. Density-dependent growth and condition (reported so it is not counted twice)

### Takeaway
Density-dependent *somatic growth* is clear in mackerel and in Baltic sprat and herring condition. In herring it is mixed: present in some North Sea components historically, and not significant in a 15-stock meta-analysis at the within-population level.

In Mareld, growth comes from intake, so this channel belongs to the food term, not to M1. At mean density, however, the model's food term barely depends on herring density: herring are a type III specialist on zooplankton and saturated at mean zooplankton biomass **(computed)**. So the explicit food density dependence is currently weak, unless zooplankton is depleted locally.

### Literature
- Brunel & Dickey-Collas 2010 (15 North Atlantic herring stocks):
  - "Mean population biomass density varied between 0.23 t km-2 in the Irish Sea and 5.6 t km-2 for the Norwegian spring spawning population", and up to 12.1 t/km2 in single years.
  - Across stocks, winf rises with density (confounded with temperature: r = -0.61).
  - At the population level, "biomass and recruit density did not have a significant effect on cohort-specific growth parameters".
  
  (fulltext) - [doi:10.3354/meps08491](https://www.int-res.com/articles/meps2010/405/m405p015.pdf)
- Dickey-Collas et al. 2010:
  - Length-at-age rose about 2 cm at age 3 during the collapse in the Banks and Downs components, "negatively correlated with the changes in stock biomass".
  - Weight-at-age fell during the recovery, inversely correlated with stock size in Downs herring (Shin & Rochet 1998).
  - "strong herring year classes have grown poorly in recent years".
  - Melvin & Stephenson 2007 is cited for reduced competition at small stock size.
  
  (fulltext; Melvin & Stephenson **secondary**) - [doi:10.1093/icesjms/fsq033](https://www.vliz.be/imisdocs/publications/ocrd/256670.pdf)
- Olafsdottir et al. 2016, NEA mackerel 1984-2013:
  - "Growth of the latest cohort was 34% of the maximum cohort growth recorded."
  - "Both weight-at-length and cohort growth were negatively affected by mackerel stock size and Norwegian spring-spawning herring stock size (weight-at-length: r2 = 0.89 ...)".
  
  (abstract, Crossref) - [doi:10.1093/icesjms/fsv142](https://doi.org/10.1093/icesjms/fsv142)
- Casini et al. 2011: "fluctuations in sprat density have been the main drivers of the spatio-temporal changes of both sprat and herring condition"; "Herring density was an additional albeit weak significant predictor"; "density-dependent mechanisms also operate at the spatial scale within stock units". (abstract, Crossref) - [doi:10.1007/s10144-011-0269-2](https://doi.org/10.1007/s10144-011-0269-2)
- van Deurs et al. 2009 propose that age-1 sandeels depleting food locally reduce age-0 growth, forcing riskier feeding and so higher predation. This is density dependence that acts *through* growth on mortality. (fulltext, see Q1)

### Calculation (model, read-only)
- **Herring are a type III specialist on zooplankton.** `state.py:148` sets the type III mask for predators with exactly one prey, and `pelagic_fish` has `preys_on: true` only for zooplankton.
- **Parameters.** a = 0.04, h = 25, so ah = 1.0 km2/t. Zooplankton at the inference start is 35000 t / 3600 km2 = 9.7 t/km2. Herring interference I = 0.7 x 3.33 = 2.3.
- **Elasticity of per-capita intake with respect to own biomass** = -I/(1 + ahZ^2 + I) **(computed, uniform density, zooplankton held fixed)**:
  - -0.024 at 3.3 t/km2 per cell;
  - -0.07 at 10 t/km2;
  - -0.18 at 30 t/km2.
  - For comparison, a type II response would give -0.18 at the mean.
- Exploitative competition (zooplankton drawn down by herring) comes on top of this and has to be measured in a run.
- **Visibility caveat.** The herring -> zooplankton pair has `visibility_floor: 0.2`, so hiding zooplankton lowers the *visible* Z. The elasticity rises as visible zooplankton falls **(computed)**:
  - -0.08 at Z_vis = 5 t/km2;
  - -0.19 at Z_vis = 3 t/km2.
  
  The "weak" result therefore holds only if visible zooplankton stays near its total.

### Inference
- If visible zooplankton is near its total, the model has very little explicit growth density dependence for herring near B_ref. A recruitment-equivalent term on M1 would then not double count the food term at mean density. This is conditional on both zooplankton visibility and zooplankton depletion.
- It may partly double count in dense cells, or wherever zooplankton is depleted.
- The mackerel and sprat growth density dependence in the literature is competition for plankton. It should come from the food term and should not be added to M1.

---

## Q3. Density-dependent mortality proper

### Takeaway
Direct density-dependent mortality on the post-recruit pool has little evidence for small pelagics:
- **Egg predation** is strong but acts at the egg stage, and its density form is ambiguous (snippet-level).
- **Cannibalism** is minor in sandeel, by van Deurs' account.
- **Schooling** dilutes per-capita risk locally.
- **The Ichthyophonus epizootic** (1991-1993, about 10 % or more of the stock in the Skagerrak and Kattegat) was episodic, with no demonstrated density link.

What exists is early-life density dependence, already summarised as recruitment compensation in Q1.

### Literature
- Richardson et al. 2011, Georges Bank herring: "Estimated egg survival rates ranged from <2-70% from 1971 to 2005"; haddock egg predation "can cause alternate stable population levels"; "when the haddock population is high, seemingly conservative levels of fishing can still precipitate a severe decline". (abstract, Crossref) - [doi:10.1073/pnas.1015400108](https://doi.org/10.1073/pnas.1015400108)
- Kotterba et al. 2014, WBSS spawning lagoon:
  - Stickleback predation on herring eggs; 20 % of eggs consumed in the exclusion experiment, extrapolated to 42 % between spawning and hatch.
  - A "positive link between egg concentration per area and predation intensity", and nearly all eggs consumed at 25 eggs/cm2 or more.
  
  (snippet only; the AFS abstract page failed to resolve; **not verified**, kept qualitative) - [AFS 2014 abstract (failed)](https://afs.confex.com/afs/2014/webprogram/Paper16986.html)
- Arnott & Ruxton 2002 discuss possible cannibalism by age-1 sandeels on hatching larvae. van Deurs et al. 2009 argue it "may be insignificant" because the adults are buried in winter. (both fulltext)
- Beare, Reid & Petitgas 2002, NW North Sea herring: "herring school size is regulated mainly by location and time-of-day and ... 'measured' school size is not influenced by the local 'school count per EDSU'". (fulltext) - [doi:10.1006/jmsc.2002.1201](https://archimer.ifremer.fr/doc/00000/10113/7347.pdf)
- Mellergaard & Spanggaard 1997: Ichthyophonus prevalence fell "from 10.6 to 2.0% in the North Sea, from 8.0 to 5.6% in the Skagerrak"; "Annual mortality ... was roughly estimated at 12.8 to 36% in 1991, decreasing to a few percent in 1993". (abstract, publisher page) - [doi:10.3354/dao028191](https://doi.org/10.3354/dao028191)
- Rahimian & Thulin 1996: Skagerrak-Kattegat 1991, "more than 300 million, i.e. more than 10% of the population"; "the spring spawning population was assumed to be the most heavily infected". (abstract, publisher page) - [doi:10.3354/dao027187](https://doi.org/10.3354/dao027187)
- Dickey-Collas et al. 2010, Fig. 4 text: SMS predation mortality on 2+ herring "has been very stable" across a 100-fold change in herring biomass. (fulltext. This is a model output whose functional-response assumptions I did not check, not an observation.)

### Calculation
- **Egg stage.** If egg survival varies from <2 % to 70 % and is set by predator abundance (Richardson), most of the variation is predator-driven, not prey-driven. No beta can be extracted.
- **Disease.** About 0.13-0.36 /yr in 1991 falling to "a few percent" by 1993 is a pulse of roughly 0.1-0.3 /yr lasting 1-2 years. There is no density covariate, so no beta.
- **SMS M2 at stock scale.** An M2 that stayed stable while herring biomass changed about 100-fold implies an elasticity near 0 in that model **(computed: |d ln M2 / d ln B| << 0.1)**.

### Inference
There is no literature basis for a separate post-recruitment density-dependent mortality term in small pelagics beyond the recruitment-equivalent one. The egg-predation and disease evidence argues for *environmental and predator-driven variance*, not a density slope.

---

## Q4. Sign of the unmodelled-predation part of M1 (0.19 /yr)

### Takeaway
The theory for satiating predators gives a *negative* (depensatory) slope: specific predation mortality falls as prey rise above the satiation point (Bakun 2006). Generalist predators that switch prey give type III, which is compensatory at low prey: harbour porpoises in the southern North Sea are fitted best by type III (Ransijn et al. 2021).

The meta-analyses find little depensation in pelagic stocks (Hilborn et al. 2014). The net sign of the 0.19 component is therefore uncertain, with a magnitude between about -0.19 and +0.05 /yr. I take -0.05 as central.

The model's *explicit* gadoid predation on herring is computed to be in its saturated, depensatory branch at mean density.

### Literature
- Bakun 2006: "a refuge from predation when abundance is very low, very destructive predation between an abundance level sufficient to attract interest from predators and an abundance level sufficient to satiate available predators, and, as abundance increases beyond this satiation point, decreasing specific predation mortality". Above satiation, "the specific predation mortality rate ... decreases toward an asymptote of zero". (fulltext) - [doi:10.1016/j.pocean.2006.02.004](https://www.mafmc.org/s/Bakun_PredatorPit.pdf)
- Ransijn et al. 2021: "Prey switching behavior (change in preference dependent on prey abundance) was confirmed by the favored type III functional response model"; "a preference for sandeels". (abstract, Crossref) - [doi:10.1002/ece3.8380](https://doi.org/10.1002/ece3.8380)
- Hilborn et al. 2014:
  - "Hierarchical meta-analysis showed that the majority of the evidence was for no depensatory behaviour but could not rule out depensation at very low stock sizes".
  - For pelagic stocks "there is almost no weight for q > 0.04".
  - "some taxa (particularly clupeids) recovered about as frequently as they declined".
  
  (fulltext) - [doi:10.1093/icesjms/fsu035](https://www.arlis.org/docs/vol2/hydropower/Hilborn%202014.pdf)
- Liermann & Hilborn 1997, via Hilborn et al. 2014: "the most probable values ... close to or within the range of no depensation". (**secondary**; my own fetch failed) - [doi:10.1139/f97-105](https://doi.org/10.1139/f97-105)

### Calculation
- **Unmodelled predators.**
  - If they are type II and moderately saturated (ahB of about 0.3-1), the elasticity is -0.23 to -0.5, giving beta = 0.19 x (-0.23 to -0.5) = -0.04 to -0.10 /yr **(computed)**.
  - Fully saturated gives -0.19. Type III below the inflection gives up to +0.19.
- **Explicit gadoid predation in Mareld** (type III forced, a = 0.04, h = 217, gadoid I = 1.0 x 2500/3600 = 0.69). At herring 3.3 t/km2, all visible: ahB^2 = 94.5, elasticity = -0.965, so beta_M2,gadoid = 0.30 x (-0.965) = **-0.29 /yr** **(computed)**.
  - With 60 % or 35 % visible: -0.27 or -0.22.
  - The compensatory refuge applies only below B* = sqrt(1.69/8.68) = 0.44 t/km2 of visible herring.
- **Porpoise (type II).** ahB = 1.42, so beta = 0.0117 x (-0.59) = -0.007 /yr **(computed)**.
- These assume uniform density, fixed predator biomass and appetite, and no prey switching. Gadoids also eat zooplankton, benthos and each other, which softens saturation.

### Inference
- The non-modelled predators should not add a positive slope to M1. If anything they argue for a slightly lower gamma.
- In the model, explicit predation at mean density is depensatory by about -0.3 /yr. Any positive gamma on M1 is fighting that, not adding to an existing compensatory term.

---

## Q5. Converting each finding to beta_M and gamma at B_ref

### Calculation
The component weights are 30 / 36 / 20 / 7 / 7 % (herring / sandeel / mackerel / sprat / horse mackerel).

| Line of evidence | Formula | beta (per yr) | gamma = beta/0.58 |
|---|---|---|---|
| Schaefer with model r_max 0.648, B_ref at d = 0.3 / 0.5 | beta = r d | 0.19 / 0.32 | 0.33 / 0.56 |
| Same, if today's F = 0.10 is in equilibrium at B_ref (d = 1 - F/r = 0.85) | beta = r - F | 0.55 | 0.95 |
| MSY identity, weighted ICES F_MSY/Fcap (herring 0.23-0.31, sandeel 0.40, mackerel 0.19, sprat 0.71, horse mackerel ~0.15 *unverified*) | beta(B_MSY) = F_MSY | about 0.33 | about 0.57 |
| Segmented S-R, Z_eff from MSY (herring 0.40, sandeel 0.50, mackerel 0.22, sprat 0.70, horse mackerel 0.20) | Z_eff \|e\| with \|e\|: herring 0.5 (50/50 NSAS/WBSS, *assumed*), sandeel 0, mackerel 1, sprat 1, horse mackerel 0.77 | 0.16 | 0.28 |
| Same, Z_eff = Z_pool = 0.99 (upper bound) | | 0.47 | 0.81 |
| BH steepness/regression, Z_eff from MSY | \|e\|: herring 0.63, sandeel 1.0 (capped), mackerel 0.87, sprat 0.60, horse mackerel 0.77 | 0.33 | 0.58 |
| Same, Z_eff = 0.99 | | 0.81 | 1.40 |
| Sandeel alone, Arnott & Ruxton e = -1.2 to -2.0, capped at 1, Z_eff 0.5-0.99 | | 0.5-0.99 (sandeel only) | 0.9-1.7 |
| Unmodelled predation sign (Q4) | | -0.19 to +0.05, central -0.05 | -0.33 to +0.09 |
| Explicit model M2 at mean density (Q4) | | -0.29 | (already in model) |
| Explicit model food term at mean density (Q2) | | small (per-capita intake elasticity -0.02) | (already in model) |

All entries are **(computed)**. The per-component products for the BH line with MSY Z_eff are herring 0.25, sandeel 0.50, mackerel 0.19, sprat 0.42 and horse mackerel 0.15 /yr.

### Inference
- The *total* compensation the literature implies for this pool at B_ref is about **beta_total ≈ 0.3 /yr** (range 0.16-0.55). This is consistent across three independent routes: the model's own r_max read as a Schaefer pool, the MSY identity with ICES F_MSY, and steepness converted with an MSY-calibrated Z_eff.
- The simple-pool conversion (Z_eff = Z) roughly doubles this. It is an upper bound because it ignores the delay between recruitment and biomass.

---

## Q6. Spatial scale

### Takeaway
The recruitment-type density dependence in Q1 operates at the *stock* scale, on spawning and nursery grounds that mostly lie outside the 60 x 60 km domain:
- **WBSS:** the Rugen / western Baltic spawning component (ICES advice).
- **NSAS:** spawning grounds in the North Sea and Channel.

Herring and mackerel are migratory schoolers. School size does not respond to local school count (Beare et al. 2002), and density generally changes faster than occupied area (Thorson et al. 2016, demersal species). A per-1-km2-cell power law on herring biomass is therefore not a proxy for stock density. Sandeel is the exception: adults are non-migratory and bank-bound (Arnott & Ruxton), so density dependence could be local, at bank scale.

### Literature
- Thorson et al. 2016, 92 bottom-associated species: "0.6% increase in area for a 10% increase in abundance"; "population density generally changes faster than effective area occupied". Pelagic species were not included. (abstract, Crossref) - [doi:10.1098/rspb.2016.1853](https://doi.org/10.1098/rspb.2016.1853)
- Beare et al. 2002: school size is independent of local school count. (fulltext, Q3)
- Pacific herring (Puget Sound): age-0 "herring distributions do expand as a function of abundance". (snippet, **not verified**) - [AFS 2011 abstract](https://afs.confex.com/afs/2011/webprogram/Paper3937.html)
- Casini et al. 2011: density dependence in condition "also operate[s] at the spatial scale within stock units". (abstract, Q2)

### Inference
- **Use the domain mean.** If the term is adopted, let M1 depend on the *domain-mean* pool biomass, B_total/3600 km2 (or a field smoothed over tens of km), not on B in each cell.
- **Why not per cell.** Herring cell biomasses vary by orders of magnitude around 3.3 t/km2 through spawn colonies and schooling. A per-cell M1 proportional to B_cell^gamma would:
  - penalise schools and reward spreading thin, a learned behavioural artefact analogous to the `--local_reward_norm sum` problem noted in AGENTS.md;
  - contradict the local dilution effect of schooling.
- **Sandeel.** Only a bank-scale term would be defensible, and the pool does not resolve sandeel separately.

---

## Q7. Recommendation

### Recommended value
- **Literature-only target for M1** (assuming the model's explicit terms were density-neutral): recruitment-equivalent beta of about +0.3 /yr, plus the unmodelled-predation sign of about -0.05 /yr.
- **Central: gamma = 0.45, beta_M ≈ 0.26 /yr.**
  - Power form: M1 = 0.58 (B/B_ref)^0.45.
  - Linear form: M0 = 0.32, k = 0.078 /yr per t/km2.
- **Range: gamma 0.25-0.8, beta 0.15-0.46 /yr.** The upper end is the simple-pool conversion with segmented S-R (0.81). The Schaefer-equilibrium line (0.95) is left out as a bracket, not a reading: it assumes today's pool is in equilibrium under F = 0.10. Q1 (WBSS and sandeel below Blim, NSAS in a low-productivity regime) and the model's -15 %/yr herring decline both contradict that assumption. gamma = 1.0 is the largest value the linear form can take with a non-negative floor.
- **Contributions to the central beta, by component** **(computed: mean of the BH and segmented lines with MSY Z_eff, weighted, then scaled from 0.25 to 0.30 before the -0.05)**:
  - sandeel ≈ 0.11 (most uncertain: hockey stick says 0, regression says strong over-compensation);
  - herring ≈ 0.08 (WBSS below Blim pulls it down);
  - mackerel ≈ 0.05;
  - sprat ≈ 0.05;
  - horse mackerel ≈ 0.01.

### Case for gamma = 0 or small
1. **Depleted components.** WBSS and sandeel 3r are below their segmented-regression breakpoints, where the ICES S-R form has no compensation. If the Skagerrak pool is dominated by these, gamma -> about 0.1-0.2.
2. **No depensation.** Depensation is not supported for pelagics (Hilborn et al. 2014), so gamma < 0 has no support either. gamma = 0 is the neutral floor, not a conservative choice against collapse.
3. **Non-stationarity.** NSAS has been in a "low productivity regime since 2002", recruitment has shifted in 85 % of stocks, and environment-driven failures occur (Payne 2009). These argue that B_ref and the curve move. A fixed gamma tuned at today's B_ref encodes a carrying capacity the literature says is not fixed.
4. **Double counting.** Food-mediated density dependence would double count with gamma. In the current model it is computed to be weak near B_ref (Q2), but only if visible zooplankton is near its total. At low visible zooplankton (3 t/km2, elasticity -0.19), or where herring deplete zooplankton strongly, this argument gains weight.
5. **Herring year classes tracked SSB.** In 80 % of years in 1976-2000, NSAS year-class strength "reflected SSB" (Nash & Dickey-Collas 2005). This is the strongest herring-specific evidence for weak compensation (e near 0) at the stock scale. It sits in tension with the compensation found by Nash et al. 2009 and Dickey-Collas et al. 2010, which was strongest during the recovery.

What does **not** support gamma = 0 is the observation that "recruitment is unrelated to SSB". In a pool model that is strong compensation (e near -1).

### Interactions to flag (not solved here)
- **Explicit predation is depensatory.** At mean density the model's gadoid predation on herring is computed to be saturated, contributing beta of about -0.29 /yr. With gamma = 0.45 on M1, the model's *net* compensation at B_ref would be about 0.26 - 0.29 - 0.01 ≈ 0. The pool would still have no net regulation near B_ref. Reaching the literature total of about 0.3 /yr would need gamma near 1.0 or a less saturated gadoid response.
  - These numbers assume uniform density. **Before choosing gamma, measure the model's explicit beta**: perturb herring biomass by ±10 % with frozen policies and record per-capita growth, M2 and intake.
- **Normalisation at B_ref masks the decline.** Normalising M1 = M1_ref at B_ref means gamma > 0 *lowers* M1 whenever B < B_ref. That will partly mask the known herring underproduction decline (about -15 %/yr) without fixing its cause. Diagnose that decline separately, with gamma = 0, first.
- **Spatial form.** Use the domain-mean biomass, not per-cell biomass (Q6).

---

## Sources and how each was read

| Source | Link | Read as |
|---|---|---|
| Myers, Bowen & Barrowman 1999 CJFAS 56:2404 | https://doi.org/10.1139/f99-201 | abstract (Crossref); SEDAR PDF 404 |
| Myers et al. 1998 NAFO SCR 98/5 | https://www.nafo.int/Portals/0/PDFs/sc/1998/scr-98-005.pdf | fulltext |
| Froese & Luna 2004 Acta Ichthyol Piscat 34:11 | https://www.fishbase.de/rfroese/NoFecundRel.pdf | fulltext (Myers alphas secondary) |
| SEDAR S8RD21 record (h = alpha/(4+alpha)) | https://www.fisheries.noaa.gov/inport/item/60041 | snippet |
| Thorson et al. 2012 CJFAS 69:1556 | https://doi.org/10.1139/f2012-077 | fulltext |
| Thorson 2020 Fish Fish 21:237 | https://repository.library.noaa.gov/view/noaa/54871 | fulltext (grep; Clupeidae not found) |
| Dickey-Collas et al. 2010 ICES JMS 67:1875 | https://doi.org/10.1093/icesjms/fsq033 | fulltext |
| Nash, Dickey-Collas & Kell 2009 Fish Res 95:88 | https://doi.org/10.1016/j.fishres.2008.08.003 | snippet (+ secondary via Dickey-Collas 2010) |
| Nash & Dickey-Collas 2005 Fish Oceanogr 14:279 | https://doi.org/10.1111/j.1365-2419.2005.00336.x | abstract |
| Payne et al. 2009 ICES JMS 66:272 | https://doi.org/10.1093/icesjms/fsn211 | fulltext |
| ICES 2025 her.27.3a47d advice | https://doi.org/10.17895/ices.advice.27202626 | fulltext |
| ICES 2025 her.27.20-24 advice | https://doi.org/10.17895/ices.advice.27202614 | fulltext |
| ICES 2026 san.sa.3r advice and stock annex | https://ndownloader.figshare.com/files/62226832 ; https://ndownloader.figshare.com/files/56798417 | fulltext |
| ICES 2026 spr.27.3a4 advice | https://ndownloader.figshare.com/files/64136503 | fulltext |
| ICES 2026 mac.27.nea advice | https://ndownloader.figshare.com/files/69489192 | fulltext |
| Arnott & Ruxton 2002 MEPS 238:199 | https://doi.org/10.3354/meps238199 | fulltext |
| van Deurs et al. 2009 MEPS 381:249 | https://doi.org/10.3354/meps07960 | fulltext |
| Szuwalski et al. 2015 Fish Fish 16:633 | https://doi.org/10.1111/faf.12083 | abstract |
| Vert-pre et al. 2013 PNAS 110:1779 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3562848 | fulltext via summarising fetch (weaker) |
| Zimmermann et al. 2018 J Anim Ecol 87:672 | https://doi.org/10.1111/1365-2656.12800 | secondary (sibling gadoid note) |
| Brunel & Dickey-Collas 2010 MEPS 405:15 | https://doi.org/10.3354/meps08491 | fulltext |
| Olafsdottir et al. 2016 ICES JMS 73:1255 | https://doi.org/10.1093/icesjms/fsv142 | abstract |
| Casini et al. 2011 Popul Ecol 53:511 | https://doi.org/10.1007/s10144-011-0269-2 | abstract |
| Casini et al. 2006 Oikos 112:638 | (bibliographic only) | not read |
| Melvin & Stephenson 2007 | via Dickey-Collas 2010 | secondary |
| Richardson et al. 2011 PNAS 108:13606 | https://doi.org/10.1073/pnas.1015400108 | abstract |
| Kotterba et al. 2014 (stickleback, herring eggs) | https://afs.confex.com/afs/2014/webprogram/Paper16986.html | snippet; page fetch failed |
| Beare, Reid & Petitgas 2002 ICES JMS 59:469 | https://archimer.ifremer.fr/doc/00000/10113/7347.pdf | fulltext (abstract and grep) |
| Mellergaard & Spanggaard 1997 DAO 28:191 | https://doi.org/10.3354/dao028191 | abstract |
| Rahimian & Thulin 1996 DAO 27:187 | https://doi.org/10.3354/dao027187 | abstract |
| Bakun 2006 Prog Oceanogr 68:271 | https://doi.org/10.1016/j.pocean.2006.02.004 | fulltext |
| Ransijn et al. 2021 Ecol Evol 11:17458 | https://doi.org/10.1002/ece3.8380 | abstract |
| Hilborn et al. 2014 ICES JMS 71:2141 | https://doi.org/10.1093/icesjms/fsu035 | fulltext |
| Liermann & Hilborn 1997 CJFAS 54:1976 | https://doi.org/10.1139/f97-105 | secondary (fetch failed) |
| Thorson et al. 2016 Proc R Soc B 283:20161853 | https://doi.org/10.1098/rspb.2016.1853 | abstract |
| Pacific herring basin-model abstract (AFS 2011) | https://afs.confex.com/afs/2011/webprogram/Paper3937.html | snippet |
| Andersen et al. 2017; Patterson 1996 | - | not reached |
| Model facts: `fgconfig/fg_library.yaml` 144-200, 425-530; `mareld2.yaml` 70-96; `lib/environments/ecosystem_env/interactions.py` 284-310; `state.py` 148-158; `predation.py` | local repo | read |
