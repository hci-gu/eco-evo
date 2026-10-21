# Phytoplankton loss terms for the Mareld site (Skagerrak/Kattegat, ~58.1 N 10.8 E, ~100 m depth, mixed layer 6-17 m above the Baltic-outflow halocline, spring diatom bloom Feb-Mar)

Scope: about 45 tool calls (web searches, abstract pages, PDFs run through pdftotext, ERSEM parameter
file and source code from GitHub). Five items: (1) basal respiration, (2) lysis / viral / non-grazing
mortality, (3) sinking loss from a shallow mixed layer, (4) what the Stelfox-Widdicombe et al. 2004
dilution growth rates contain, (5) non-grazing losses in regional ecosystem models.

Read-status tags: **[F]** = full text read by me; **[A]** = abstract only; **[S]** = search snippet or
secondary summary, unverified. "[F] of X" = I read X in full but not the original it cites.
**[D]** = my own calculation. Rates are specific rates on phytoplankton biomass (d-1) unless stated.

## Sources

| # | Source | Used for | Read |
|---|---|---|---|
| 1 | Stelfox-Widdicombe CE, Archer SD, Burkill PH, Stefels J (2004) Microzooplankton grazing in Phaeocystis and diatom-dominated waters in the southern North Sea in spring. J. Sea Res. 51:37-51. doi:10.1016/j.seares.2003.04.004 ([VLIZ PDF](https://www.vliz.be/imisdocs/publications/55283.pdf)) | Q4 | [F] |
| 2 | Brussaard CPD, Riegman R, Noordeloos AAM, Cadee GC, Witte H, Kop AJ, Nieuwland G, van Duyl FC, Bak RPM (1995) Effects of grazing, sedimentation and phytoplankton cell lysis on the structure of a coastal pelagic food web. MEPS 123:259-271. doi:10.3354/meps123259 | Q2, Q3 | [F] |
| 3 | Brussaard CPD, Gast GJ, van Duyl FC, Riegman R (1996) Impact of phytoplankton bloom magnitude on a pelagic microbial food web. MEPS 144:211-221. doi:10.3354/meps144211 | Q2 | [F] |
| 4 | Baudoux A-C, Noordeloos AAM, Veldhuis MJW, Brussaard CPD (2006) Virally induced mortality of Phaeocystis globosa during two spring blooms in temperate coastal waters. Aquat. Microb. Ecol. 44:207-217. doi:10.3354/ame044207 | Q2, Q4 | [F] (abstract, methods, results grep-read) |
| 5 | Carstensen J, Conley D, Mueller-Karulis B (2003) Spatial and temporal resolution of carbon fluxes in a shallow coastal ecosystem, the Kattegat. MEPS 252:35-50. doi:10.3354/meps252035 | Q1, Q3 | [F] (methods, results); Cole 1991, Langdon 1993 not read |
| 6 | Mantikci M, Hansen JLS, Markager S (2017) Photosynthesis enhanced dark respiration in three marine phytoplankton species. J. Exp. Mar. Biol. Ecol. 497:188-196. doi:10.1016/j.jembe.2017.09.015 | Q1 | [A] |
| 7 | Behrenfeld MJ, Boss ES (2014) Annu. Rev. Mar. Sci. 6:167-194, Supplement (table of Evans & Parslow 1985 values) | Q1 | [F] of BB14; E&P 1985 not read |
| 8 | Fennel K et al. (2006) Nitrogen cycling in the Middle Atlantic Bight. Global Biogeochem. Cycles 20:GB3007. doi:10.1029/2005GB002456 | Q2, Q3, Q5 | [F] (Table 1) |
| 9 | Butenschoen M et al. (2016) ERSEM 15.06. Geosci. Model Dev. 9:1293-1339. doi:10.5194/gmd-9-1293-2016; parameters from [`testcases/ersem-15.06/fabm-ersem-15.06-L4-ben-docdyn-iop.yaml`](https://github.com/pmlmodelling/ersem) and `src/primary_producer.F90` | Q5 | [F] (Eqs 7, 12, 144, 239; P1-P4 blocks; code lines for lysis, respiration, sinking) |
| 10 | Samuelsen A, Hansen C, Wehde H (2015) HYCOM-NORWECOM V2.1. Geosci. Model Dev. 8:2187-2202. doi:10.5194/gmd-8-2187-2015, and its supplement Skogen & Soiland (1998) NORWECOM user guide v2.0 | Q5 | [F] (sect. 2.1, Table 2; user guide sect. 3.1.3 and parameter table) |
| 11 | Mackinson S, Daskalov G (2007) An ecosystem model of the North Sea. Cefas Sci. Ser. Tech. Rep. 142 ([PDF](https://www.cefas.co.uk/publications/techrep/tech142.pdf)) | Q3 (Cadee 1985), Q5 | [F] (phytoplankton section, basic-input table); Cadee 1985 not read |
| 12 | Kiorboe T, Hansen JLS, Alldredge AL, Jackson GA, Passow U, Dam HG, Drapeau DT, Waite A, Garcia CM (1996) Sedimentation of phytoplankton during a diatom bloom: rates and mechanisms. J. Mar. Res. 54 ([EliScholar](https://elischolar.library.yale.edu/journal_of_marine_research/2212)) | Q3 | [A] |
| 13 | Olesen M, Lundsgaard C (1995) Seasonal sedimentation of autochthonous material from the euphotic zone of a coastal system. Estuar. Coast. Shelf Sci. 41:475-490 (Kattegat, 1989 data) | Q3 | [A] |
| 14 | Olesen M (1995) Comparison of the sedimentation of a diatom spring bloom and of a subsurface chlorophyll maximum. Mar. Biol. 121:541-547 (southern Kattegat, 1990 data) | Q3 | [A] |
| 15 | Lundsgaard C, Olesen M (1997) The origin of sedimenting detrital matter in a coastal system. Limnol. Oceanogr. 42:1001-1005 | Q3 | [A] |
| 16 | Hoeglander H, Larsson U, Hajdu S (2004) Vertical distribution and settling of spring phytoplankton in the offshore NW Baltic Sea proper. MEPS 283:15-27. doi:10.3354/meps283015 | Q3 | [A] |
| 17 | Hansen JLS, Josefson AB (2003) Accumulation of algal pigments and live planktonic diatoms in aphotic sediments during the spring bloom in the transition zone of the North and Baltic Seas. MEPS 248:41-54. doi:10.3354/meps248041 | Q3 | [A] |
| 18 | Zhu J et al. (2026) A review of phytoplankton sinking rates. Biology (Basel) 15:130. doi:10.3390/biology15020130 | Q3 | [S] (WebFetch summary of first half of text) |
| 19 | Agusti S, Satta MP, Mura MP, Benavent E (1998) Dissolved esterase activity as a tracer of phytoplankton lysis. Limnol. Oceanogr. 43:1836-1849 | Q2 | [S] (no rates retrieved) |

## Q1. Basal (dark / maintenance) respiration

### Literature
- **Fraction of Pmax, field convention.** Carstensen et al. 2003 (Kattegat budget model): "It is generally believed that respiration is in the order of 5 to 10% of the light-saturated gross photosynthetic rate (Cole et al. 1991, Langdon 1993)". Their own model uses `Respiration = L*RL*Pmax + (24-L)*RD*Pmax` with "RL = 3% and RD = 1.5%", where RL, RD were *fitted* "so that respiration would account for approximately 50% of the 14C measurements during the darkest period of the year". Result: "Respiration accounted for on average 12% of the measured primary production"; "approximately 7 and 50% of primary production in summer ... and winter". [F] of Carstensen; Cole 1991 and Langdon 1993 not read.
- **Cultures.** Mantikci et al. 2017 (Danish group; a diatom, a dinoflagellate, a cryptophyte): maximal light-enhanced respiration Rmax:Pmax(g) "similar to 0.12" in all three; maintenance respiration 0.004-0.02 mol O2 (g Chl)-1 h-1; growth respiration 3.5 % h-1 of newly fixed C. [A]
- **NPZ convention.** Evans & Parslow 1985 via BB14 suppl.: "b - phytoplankton respiration 0.07 [day-1]". [F] of BB14.
- **Regional models** (details in Q5): ERSEM diatoms rest respiration `srs: 0.04` d-1 at 10 C (Q10 = 2) plus activity respiration `pu_ra: 0.2` of assimilation [F]; NORWECOM "metabolic loss rate at 0 C" 8.05E-7 s-1, temperature coefficient 0.07 C-1 [F].

### Calculation [D]
- Carstensen's RL/RD as a share of daily light-saturated gross production (Pmax held for L hours): (12*0.03 + 12*0.015)/12 = 4.5 % at L = 12 h; 5.1 % at L = 10 h.
- NORWECOM: 8.05E-7 s-1 = 0.070 d-1 at 0 C; x exp(0.07 T) -> 0.099 d-1 at 5 C, 0.114 at 7 C, 0.14 at 10 C. Ratio to diatom Pmax at 0 C (1.53E-5 s-1 = 1.32 d-1): 5.3 %.
- ERSEM: srs/sum = 0.04/1.375 = 2.9 % of max productivity (rest part only); at 5 C, 0.04*2^(-0.5) = 0.028 d-1.
- Mantikci maintenance to d-1, assuming 1 mol C per mol O2 and C:Chl 30-50 g/g: 0.004 -> 0.023-0.038 d-1; 0.02 -> 0.115-0.19 d-1 (culture temperature not known to me).

### Inference
- Basal respiration of ~0.03-0.1 d-1 at 3-8 C, central **0.05 d-1**, or ~**5 % of Pmax** (range 3-12 %).
- Whether to add it depends on whether the model's maximum growth rate is gross or net. A dilution-derived rate (Stelfox, Q4) is already net of 24 h respiration; subtracting 0.05 again double-counts. A 14C or Eppley-type gross Pmax needs it.

## Q2. Lysis / viral / non-grazing mortality

### Literature
- **Diatom spring bloom, Dutch coast.** Brussaard et al. 1996: "Specific algal cell lysis rates during the diatom bloom were always <0.1 d-1"; text: "always less than 10% d-1". [F]. Brussaard et al. 1995: "it is unlikely that autolysis of diatom cells was responsible for the decline of the spring bloom", and their sedimentation estimate assumes "daily losses of 5% due to cell lysis". [F]
- **Phaeocystis (after the diatoms).** Brussaard 1996: "The bloom of Phaeocystis declined through cell lysis (up to 0.3 d-1)"; Brussaard 1995: lysis peaked at "43% d-1" and was "the major loss factor for Phaeocystis, accounting for 75% of the decline of the bloom". [F]
- **Viral.** Baudoux et al. 2006, southern North Sea springs 2003-04: P. globosa single-cell viral lysis "low (0.01 to 0.03 d-1) until the collapse", then up to "0.35 d-1"; total cell lysis "<0.1 d-1 until Day 121", with "maximum rates of about 0.2 d-1"; viral lysis was 5-66 % of total mortality "assuming no loss due to sinking". [F]
- **Model values.** Fennel et al. 2006: "mP phytoplankton mortality 0.15 d-1", range 0.05-0.2 (Taylor et al. 1991). This is a lumped term; the model has no separate phytoplankton respiration. [F]. ERSEM lysis 0.045 d-1 at full nutrients, rising under nutrient stress (Q5) [F].
- Agusti et al. 1998 report "high phytoplankton lysis rates" in the summer NW Mediterranean (oligotrophic, not comparable). [S]

### Inference
- Diatoms during the bloom: **0.05 d-1** (range 0.02-0.1). After nutrient exhaustion and in Phaeocystis or flagellate phases: 0.1-0.3 d-1, viral peaks to 0.35-0.43 d-1. A nutrient-stress dependence, as in ERSEM, reproduces that shift.

## Q3. Sinking loss from a shallow mixed layer

### Literature
- **Velocities.** Zhu et al. 2026 review: diatoms "sinking rates of less than 1 m d-1 during active growth phases", up to "50 m d-1 during the apoptotic phase" (Passow); Skeletonema costatum 0.33-0.57 m d-1; cultures 0.01-1.06 m d-1; Si limitation raises sinking "1.5-4 times". [S]. Brussaard 1995, citing Riebesell 1989/1991 and Passow 1991: aggregate sinking "max. 4 m d-1" for spring-bloom Skeletonema/Thalassiosira [F] of Brussaard. NORWECOM diatoms 0.3-3 m/d (Q5) [F]. Fennel 2006: wPhy 0.1 m/d (range 0.009-25) [F].
- **Specific loss rates measured.** Kiorboe et al. 1996 (temperate fjord diatom bloom): "specific sedimentation rates of intact phytoplankton cells ... averaging ca. 0.1 d-1"; grazing was "a loss term ... comparable to sinking of intact cells"; no mass sinking ended the bloom. [A]
- **Southern Bight (well mixed).** Brussaard 1995: after the diatom peak, assuming 5 %/d lysis and no growth, "daily sedimentation losses varied between 4 and 15% of diatom standing stock per day", which is "low compared to values (up to 50% d-1) reported by Riebesell (1989) and Passow (1991)" in stratified water. The rate is a budget residual. [F]
- **Kattegat (pycnocline traps; same geometry as Mareld).**
  - Olesen & Lundsgaard 1995 (Mar-Oct 1989): viable cells were "18%" of a 63 g C m-2 flux. "The average daily vertical flux of carbon corresponded to 14% of the phytoplankton biomass." [A]
  - Lundsgaard & Olesen 1997: detritus was 82 % of the flux and faecal pellets a minor part. [A]
  - Olesen 1995 (1990): "The spring bloom period contributed 60% of the sedimentation of intact phytoplankton cells, but only 20% of the POC and PON sedimentation." [A]
  - Carstensen 2003 (0-10 m layer, 1989-97): "Annual sedimentation ... 55 g C m-2 yr-1, equivalent to 54% of net primary production". March-May was "55 to 65" % of annual sedimentation, and March alone 24-25 %. This is a budget closure term. [F]
- **North Sea / Baltic.**
  - Cadee 1985, via Mackinson & Daskalov: "20-35% of the primary production deposits to the sediment during spring in the northern North Sea". [F] of Cefas report.
  - Hoeglander et al. 2004 (Baltic, pycnocline ~10 m): intact cells were "16, 12 and 2% of phytoplankton primary production" at 25, 50 and 100 m, and bulk carbon "12%". [A]
  - Hansen & Josefson 2003 (Kattegat-Belt sediments): input was 24 % (chl a) and 64 % (fucoxanthin) of the potential spring production. The central Kattegat had "no detectable sedimentation", and the diatom-count figure of 202 % reflects advection and episodic events. [A]
- Heiskanen (Baltic spring bloom sedimentation) was not retrieved; see Failed fetches.

### Calculation [D]
Loss rate = w/H for non-motile cells in a well-mixed layer of depth H (an upper bound for weak mixing; trapped cells re-entrained by turbulence lower it):

| w (m/d) | H = 6 m | H = 10 m | H = 17 m |
|---|---|---|---|
| 0.1 | 0.017 | 0.010 | 0.006 |
| 0.3 | 0.050 | 0.030 | 0.018 |
| 1 (growing-diatom upper end) | 0.17 | 0.10 | 0.059 |
| 3 (aggregates, Si-stressed) | 0.50 | 0.30 | 0.18 |

- The healthy regime (0.1-1 m/d) and the senescent / aggregated regime (3-50 m/d) are different processes. They should not be averaged.
- Olesen & Lundsgaard: 0.14 d-1 total C flux per phytoplankton biomass x 18 % viable cells = **0.025 d-1** as intact cells, Mar-Oct mean.
- Share of production f to rate: f x mu. Cadee's 20-35 % at a bloom mu of 0.3-0.5 d-1 gives 0.06-0.18 d-1.

### Inference
- During active bloom growth: **0.03 d-1** (range 0.01-0.1; w ~0.3 m/d over H 6-17 m). This is consistent with the Kattegat 0.025 d-1 and below Kiorboe's 0.1.
- After nutrient (Si/N) exhaustion: 0.1-0.5 d-1 for days to weeks (Brussaard 0.04-0.15; w/H for 3 m/d). That is what removes the bloom. A nutrient-limitation-triggered term (as in ERSEM and NORWECOM) is the closest model analogue.

## Q4. Stelfox-Widdicombe et al. 2004: what the dilution growth rates contain

### Literature [F]
- **Nutrients: not added.** "Concurrent with previous studies (e.g. Brussaard et al., 1995) nutrients were not added to the incubation bottles since there is evidence to suggest that nutrient additions may have detrimental effects upon microzooplankton abundance (Gifford, 1988) as well as phytoplankton growth (Lessard and Murrell, 1998)."
- **Ambient nutrients (Table 1), all 7.1-9.5 C:**
  - Nearshore NO3 was 0.9-31.6 uM, with PO4 down to 0.01-0.02 uM after 13 April.
  - Offshore NO3 was 11.4 uM early and fell to 0.21-2.2 uM late. Offshore Si was 0.7-3.7 uM.
  - The offshore site still had the higher k: "0.35 to 0.67 d-1", versus 0.13-0.41 nearshore.
- **Incubation.**
  - Water came from 2-4 m and was screened through a 200 um mesh. Bottles were 2.3 l polycarbonate in a "flow-through deck incubator ... covered with a neutral density screen to mimic near-surface irradiance, for 24 h".
  - Apparent growth was computed from "Changes in the phytoplankton concentration over the 24 h incubation, as measured by fluorometry for chlorophyll-a", with mu = k - cg.
  - Two experiments with negative k were omitted.
- **Separate 14C production per unit C** was "1.5 +- 0.5 d-1 and 1.2 +- 0.3 d-1" nearshore and offshore. Mean dilution k was 0.29 and 0.48 d-1.
- **In general (Baudoux 2006, [F]).** A 0.2 um diluent (as in Stelfox) carries viruses at ambient concentration. Per-host viral lysis is therefore the same in every bottle and lowers the intercept k. Only a 30 kDa virus-free diluent makes viral lysis scale with dilution. That is why Baudoux ran paired series: "Specific viral lysis rates were calculated from the difference in P. globosa growth rates between the 2 dilution series".

### Inference
- **k is net of respiration.** The 24 h span includes the night in the dark, and the incubator light is screened.
- **k is net of density-independent losses** (autolysis, cell death) that occur in every bottle. Viral lysis at ambient virus levels is also in k. Only grazing, whose agents are diluted, goes into g.
- **k is gross of sinking.** The bottle is closed and its whole contents are sampled, so sinking cells are not lost.
- **k is under ambient, unamended nutrients.** It is a realised rate, not a nutrient-saturated maximum. The paper's offshore 0.67 d-1 is therefore a lower bound on nutrient-saturated net growth at ~7-8 C.
- **k is chlorophyll-based.** Chl:C drift during 24 h is uncorrected.
- **The 14C P/B vs k gap (factor 3-5) is not a respiration estimate.** 14C used whole water, including >200 um Phaeocystis colonies; biomass came from biovolume conversions; and 24 h 14C is already near-net. Treat it as a methods caveat.
- **Modelling consequence:**
  - If a model's maximum growth is calibrated to Stelfox's 0.67 d-1, it should not also subtract basal respiration or diatom autolysis.
  - It should still apply sinking and grazing explicitly.
  - Lysis can be applied only in its stress-triggered form (post-bloom, Phaeocystis).

## Q5. Non-grazing losses in regional ecosystem models

### Literature
- **NORWECOM V2.0** (user guide [F]): "The death rate of the algae (in the whole water column) is assumed to be constant", and "Since zooplankton is not included in the model, the death rate of algae also includes possible grazing".
  - `cc(3) death rate (0.1) 1.6E-6 (s-1)`.
  - Respiration `a(5) metabolic loss rate at 0 C 8.05E-7 (s-1)`, `a(6) ... 0.07 (C-1)`.
  - Sinking `SRDIA min 0.3`, `SRDIA max 3.0 (m d-1)` (maximum when Si < `SIB 1.0 uM`); `SRFLA 0.25 (m d-1)`.
- **HYCOM-NORWECOM V2.1 with zooplankton** (Samuelsen 2015 [F]): "phytoplankton independent of grazing is 0.035"; Table 2 "cc(3)= 4.0 x 10-7".
- **ERSEM 15.06** (yaml, code and paper [F]); temperature response lT = Q10^((T-10)/10).
  - Diatoms (P1): `srs: 0.04` "specific rest respiration at reference temperature (1/d)", `pu_ra: 0.2` "respired fraction of primary production", `pu_ea: 0.2` excreted fraction.
  - Lysis: `sdo: 0.05` "1.1 of minimal specific lysis rate (1/d)", applied as `sdo = (1/(MIN(iNs, iNI)+0.1))*self%sdo`, with no temperature factor.
  - Sinking: `SD = self%resm * MAX(0, self%esni - iNI) + self%rm`, with `resm: 5.0`, `esni: 0.7`, and `rm` unset (0) for P1.
  - Nano (P2): srs 0.04, sdo 0.05. Pico (P3): srs 0.045, sdo 0.055. Micro (P4): srs 0.035, sdo 0.045, resm 5.0.
- **Ecopath North Sea** (Mackinson & Daskalov 2007 [F]):
  - Phytoplankton B = 7.5 g WW m-2, P/B = 286 yr-1, EE = 0.212. Of a production of 2150, 455.8 is consumed and 1694 flows to detritus (t km-2 yr-1).
  - The report notes "very little is known of the lysis", and that the gC-to-wet-weight conversion gives 400-8000 g WW m-2 yr-1.
- **Fennel 2006** (US shelf NPZD, for comparison [F]): mP 0.15 d-1 (0.05-0.2), plus quadratic aggregation, and wPhy 0.1 m/d.
- **ECOHAM** (Paetsch & Kuehn 2008; Lorkowski et al. 2012): parameter tables not retrieved. Grosse et al. 2016 (BG 13:2511) defers to Lorkowski 2012.

### Calculation [D]
- NORWECOM V2.0 death: 1.6E-6 s-1 = 0.138 d-1, including grazing. The "(0.1)" annotation next to it is unexplained. V2.1 death: 4.0E-7 s-1 = 0.035 d-1, with grazing explicit.
- ERSEM diatom lysis by nutrient-limitation factor l:

  | l | Lysis (d-1) |
  |---|---|
  | 1 | 0.045 |
  | 0.3 | 0.125 |
  | 0 | 0.5 |

  Rest respiration is 0.028 d-1 at 5 C.
- ERSEM diatom sinking is 0 above l = 0.7 and at most 3.5 m/d at l = 0. That gives up to 0.2-0.6 d-1 over H 6-17 m.
- Ecopath "other mortality" = P/B (1 - EE) = 286 x 0.788 = 225 yr-1 = 0.62 d-1. This is a mass-balance residual covering lysis, sinking and the microbial loop, and is driven by an uncertain P/B.

### Inference
- Explicitly mechanistic models put linear non-grazing loss for a growing bloom at about **0.04-0.1 d-1** (lysis 0.035-0.05 plus rest respiration 0.03-0.07), with sinking added separately.
- All three nutrient-aware models raise the loss steeply after nutrient exhaustion: lysis in ERSEM, Si-triggered sinking in ERSEM and NORWECOM.
- Lumped single-term values (Fennel 0.15; NORWECOM V2.0 0.14 including grazing) should not be stacked on top of separate respiration, lysis and sinking terms.
- For a 6-17 m mixed layer, the sum of respiration (0.05, only if growth is gross), lysis (0.05) and sinking (0.03) is about **0.08-0.13 d-1** during the bloom. After nutrient exhaustion it is 0.2-0.6 d-1.

## Failed fetches / not retrieved
- Fasham, Ducklow & McKelvie 1990; Geider 1992 ("Respiration: taxation without representation"); Geider & Osborne 1989 (New Phytol. 112:327); Langdon 1988 (J. Plankton Res. 10:1291) and 1993 (ICES Mar. Sci. Symp. 197:69); Cole et al. 1991: not found or paywalled; no values taken from them except as quoted by Carstensen 2003.
- Smayda 1970 (Oceanogr. Mar. Biol. Annu. Rev. 8:353); Riebesell 1989; Passow 1991; Cadee 1985: not retrieved; quoted only second-hand (Brussaard 1995, Zhu 2026, Mackinson 2007).
- Kiorboe et al. 1994 and 1996 full texts (EliScholar, Cloudflare challenge); Olesen 1995 and Olesen & Lundsgaard 1995 full texts (Springer/Elsevier): abstracts only.
- Heiskanen 1998 and other Baltic sedimentation papers by Heiskanen: not retrieved.
- ECOHAM parameter tables (Paetsch & Kuehn 2008; Lorkowski et al. 2012, J. Mar. Syst.): not found open access.
- Calbet & Landry 2004 (L&O 49:51) and Agusti et al. 1998 (L&O 43:1836): rate tables not retrieved.
- marineinfo.org via WebFetch (HTTP 406) and int-res.com via WebFetch (HTTP 429): read via curl instead. CORE and the Southampton ePrints record for Stelfox were blocked by bot challenges; the VLIZ copy was used.
