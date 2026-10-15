# Bottom-trawl depletion per pass (d) and benthic recovery rate (r) - parameters for direct trawling mortality on benthic_community in the Mareld / Poseidon Nord grid (Skagerrak)

Scope note: about 40 tool calls. Full text was read for Hiddink et al. 2017 (PNAS, author copy from the UEA repository; the SI appendix with Table S4 was NOT retrievable), Pitcher et al. 2022 (PNAS, full text plus the SI PDF with Tables S1-S5, via Europe PMC), Rijnsdorp et al. 2020 (ICES J. Mar. Sci., WUR repository copy), Eigaard et al. 2016 (ICES J. Mar. Sci., Ifremer copy), Skold et al. 2018 (MEPS, accepted manuscript from DTU Orbit), the Swedish HaV/Havsmiljoinstitutet report on trawling and ecosystem services (2018), and the MarLIN sensitivity review for sea pens and burrowing megafauna. Sciberras et al. 2018 is from its abstract only (Crossref metadata); Wiley, Bangor and NOAA copies all returned 403. Hiddink et al. 2019 is known only as quoted by Rijnsdorp et al. 2020. Hansson et al. 2000, Pitcher et al. 2017 and Buhl-Mortensen et al. 2023 are snippets or abstracts. Each finding is tagged (Fetched full text), (Fetched abstract) or (Snippet). Confidence is lower for snippets.

Notation used below. d = proportion of benthic biota removed by one pass of the gear over a given patch of seabed. F or SAR = swept-area ratio, the number of times per year a cell's area is swept. r = intrinsic recovery rate of the logistic model, per year. B/K = equilibrium biomass relative to carrying capacity.

## Q1. Depletion per pass by gear, sediment, and biomass vs numbers

### Takeaway
For otter trawls the standard global value is d = 0.06 per pass (Hiddink et al. 2017). The 5-95% range is 0.02-0.16, and a symmetric 95% CL on the back-transformed estimate is -0.175 to 0.243, so the estimate is uncertain. Beam trawls remove 0.14, towed dredges 0.20 and hydraulic dredges 0.41. Depletion scales almost linearly with how deep the gear penetrates the sediment (Pearson r = 0.98 across gears). Hiddink found no difference in d between biomass and numbers. Sediment had no significant effect on how community biomass responded to trawling frequency; gravel mattered only for numbers. Pitcher et al. 2022 then predicted d by sediment from penetration depth. In that model, otter trawls on mud remove 0.115 per pass and on sand 0.047, with mud and gravel above sand. Rijnsdorp et al. 2020 scaled the 0.06 by each fishery's share of gear width that penetrates the seabed. That gives 0.100 for the Nephrops and Pandalus otter-trawl fishery (OT_CRU), the highest of all otter-trawl fisheries, and 0.026 for fish otter trawls (OT_DMF).

### Cited Findings
- **Hiddink et al. 2017, PNAS 114:8301-8306, gear means.** "OTs had the smallest impact, removing on average 6% of organisms per trawl pass and penetrating on average 2.4 cm into the sediment. Median penetration depths were 2.7 and 5.5 cm for BTs and TDs, respectively, and the corresponding median depletions per trawl pass were 14 and 20%, respectively. HDs had the largest impact, removing on average 41% of organisms per pass and penetrating 16.1 cm." Penetration depth and depletion: "very closely correlated (Fig. 2) (Pearson's r = 0.980, P = 0.020)". (Fetched full text) - [Hiddink et al. 2017, author copy](https://ueaeprints.uea.ac.uk/64321/1/Published_manuscript.pdf)
- **Hiddink et al. 2017, biomass vs numbers.** "Depletion rates estimated from the experimental studies for biomass and numbers were not significantly different. Thus, the pooled estimates of d (SI Appendix, Table S4) apply to both biomass and numbers." (Fetched full text) - [Hiddink et al. 2017](https://ueaeprints.uea.ac.uk/64321/1/Published_manuscript.pdf)
- **Hiddink et al. 2017, sediment.** In the comparative (fishing-ground) studies, each unit increase in SAR reduced community biomass by a mean 15.5%. "None of the other environmental variables significantly affected this response ... The effect of sediment composition on community biomass depletion was not significant". The interaction term for trawling frequency x gravel in the biomass model was -0.00067 (SE 0.0010, P = 0.5168). For community numbers the gravel effect was significant (-0.01206, SE 0.0035, P = 0.0009): "3.1% at 0% gravel content (typical for BT studies), 5.5% at 1% gravel content (typical for OT studies), and 72% at 45% gravel content (typical for TD studies)". (Fetched full text) - [Hiddink et al. 2017](https://ueaeprints.uea.ac.uk/64321/1/Published_manuscript.pdf)
- **Hiddink et al. 2017, uncertainty ranges as quoted by Rijnsdorp et al. 2020.** "otter trawls (median: 0.06; 5-95% range: 0.02-0.16), beam trawl (median: 0.14; 5-95% range: 0.07-0.25), and dredge (median: 0.20; 5-95% range: 0.13-0.30)". These ranges come from Hiddink's SI Table S4, which I could not open, so this is second-hand. (Fetched full text of Rijnsdorp 2020) - [Rijnsdorp et al. 2020](https://edepot.wur.nl/535091)
- **Pitcher et al. 2022, PNAS 119:e2109449119, SI Table S1.** This table back-transforms Hiddink's log-response values with a symmetric SE.
  - OT: lnRR i = -0.058, d = 0.057, SE 0.105, 95% CL -0.175 to 0.243. Based on 1 gravel, 6 sand and 5 mud studies.
  - BT: d = 0.139 (CL -0.388 to 0.466).
  - TD: d = 0.202 (CL 0.084 to 0.305).
  - HD: d = 0.412 (CL 0.278 to 0.522).
  - The negative lower bounds are an artefact of the symmetric approximation. They also show that the OT estimate does not differ significantly from zero at 95%. (Fetched full text, SI) - [Pitcher et al. 2022, Europe PMC](https://europepmc.org/article/PMC/PMC8764683)
- **Pitcher et al. 2022, depletion by sediment.** These values are predicted from gear penetration depth, not measured separately on each sediment. "Average depletion rates ranged from 0.047 to 0.261 depending on gear and habitat ... Otter trawls caused the lowest depletion followed by beam trawls and towed dredges. Depletion rates were lower in sand than in gravel and mud."
  - SI Table S3, predicted d (95% CL):
    - OT gravel 0.108 (-0.073 to 0.241); OT mud 0.115 (-0.060 to 0.245); OT sand 0.047 (-0.176 to 0.208).
    - BT mud 0.181 (0.042 to 0.286).
    - TD mud 0.261 (0.145 to 0.364).
  - SI Table S2: estimated otter-trawl penetration depth 2.0 cm in mud (CL 0.3-5.3), 1.1 cm in sand and 1.9 cm in gravel.
  - (Fetched full text and SI) - [Pitcher et al. 2022](https://europepmc.org/article/PMC/PMC8764683)
- **Pitcher et al. 2022, what d measures.** The pooled data are "change in biota abundance (pooled relative biomass and numbers of epifauna and infauna)", with d = 1 - e^i. The parameters are "representative of the composition of benthic invertebrate communities, including primarily biomass of epi-fauna, with some infauna and count data". (Fetched full text) - [Pitcher et al. 2022](https://europepmc.org/article/PMC/PMC8764683)
- **Rijnsdorp et al. 2020, ICES J. Mar. Sci. 77:1772-1786, Table 1.** Depletion rates by fishery (métier), with the subsurface ratio, i.e. the proportion of gear width that penetrates 2 cm or more:
  - OT_CRU, "Nephrops, Pandalus, mixed fish": SSR 0.304, d 0.100.
  - OT_DMF, "Cod or plaice": 0.078, 0.026.
  - OT_MIX_1: 0.229, 0.075.
  - OT_MIX_2: 0.220, 0.073.
  - OT_SPF: 0.028, 0.009.
  - TBB_DMF: 1.000, 0.140.
  - DRB_MOL: 1.000, 0.200.
  - Method: "The subsurface ratio of the standard otter trawl was set equal to the mean subsurface ratio of all otter trawl métiers weighted over their swept area (ratio = 0.18) ... The depletion rates of each otter metiers m were then estimated by 0.06 x SSRm/0.18." The OT_CRU value is therefore a scaled value, not an empirical measurement. (Fetched full text) - [Rijnsdorp et al. 2020](https://edepot.wur.nl/535091)
- **Sciberras et al. 2018, Fish and Fisheries 19:698-715.** Results:
  - 122 experiments. "A gear pass reduced benthic invertebrate abundance by 26% and species richness by 19%."
  - "gears that penetrate deeper into the sediment having a significantly larger impact".
  - "Sediment composition (% mud and presence of biogenic habitat) and the history of fishing disturbance prior to an experimental fishing event were also important predictors of depletion, with communities in areas that were not previously fished, predominantly muddy or biogenic habitats being more strongly affected".
  - The 26% is an abundance (numbers) effect across all gears, including experiments in previously unfished areas. It is not a per-pass biomass d for otter trawls.
  - Gear-specific values and CIs were not retrieved. One search tool attributed "4.7-26.1%" to this paper, but that range is Pitcher et al. 2022's 0.047-0.261.
  - (Fetched abstract, via Crossref) - [Sciberras et al. 2018](https://doi.org/10.1111/faf.12283)

### Inferences
- For a single pooled benthic biomass, use a pooled biomass d. Hiddink found no difference between biomass and numbers, and no sediment effect on the biomass response. The sediment split in Pitcher 2022 comes from a penetration-depth model, so mud = 0.115 is a prediction with an interval that spans zero.
- All three otter-trawl values (0.06 pooled, 0.10 OT_CRU, 0.115 OT on mud) lie inside each other's uncertainty ranges. The literature cannot separate them statistically. Their order (pooled < Nephrops/shrimp gear < mud prediction) is still consistent: muddy sediments let the ground gear sink in deeper.

## Q2. Recovery rate r and the equilibrium model

### Takeaway
Hiddink et al. 2017 estimate the community-biomass recovery rate at r = 0.82 /yr (5-95% 0.42-1.53) for an untrawled community. Recovery from 0.5K to 0.95K takes 3.6 years (1.9-6.4). The r estimate rises with trawling frequency, to 1.73 /yr (0.89-3.23) at F = 10 /yr, because trawled communities shift toward fast life histories. Pitcher et al. 2022 give lower, cell-scale recovery rates of R = 0.29 (gravel) to 0.68 (mud) /yr, with lower CLs of 0.25-0.48. Both papers use the same equilibrium of the logistic (Schaefer) model, B/K = 1 - F d / r, which is zero when F >= r/d. The model's r = 1.46 /yr is about 1.8 times Hiddink's untrawled estimate and about 2.1 times Pitcher's mud value. It falls in the upper part of Hiddink's interval, the range that describes an already heavily trawled, fast-turnover community. So with the model's r, the same SAR produces roughly half the equilibrium depletion that the literature parameters give.

### Cited Findings
- **Hiddink et al. 2017, the model.** Recovery is described "by the logistic growth equation", and at equilibrium under chronic trawling B/K = 1 - F d/r [Eq. 1]. Here "F is trawling frequency, d is the depletion of biota caused by each trawl pass (expressed as a proportion), and r is rate of increase interpreted here as the recovery rate." In the comparative studies, log10(B/K) falls roughly linearly with F with slope b = d/r, so r is estimated as d/b once d is known from experiments. (Fetched full text) - [Hiddink et al. 2017](https://ueaeprints.uea.ac.uk/64321/1/Published_manuscript.pdf)
- **Hiddink et al. 2017, r values.** "Mean community r ... increased with trawling frequency from 0.82 y-1 when there was no trawling (5-95% uncertainty intervals = 0.42-1.53) to 1.73 (0.89-3.23) y-1 when the trawling frequency was 10 y-1 (using the mean estimated d across gears OT, BT, and TD; d = 0.13) ... relatively slight across ranges of trawling frequencies that dominate those on real fishing grounds (e.g., 0-1 y-1)." "If the fraction depleted D = 0.5K, then recovery time is 3.6 y (5-95% uncertainty intervals = 1.9-6.4 y)." For numbers, r ranged "from 0.18 y-1 for TD on 45% gravel to 4.47 y-1 for BT on 0% gravel", with recovery times of 0.7-16.6 years. "biomass recovery rates were slower and ... recovery times were longer than those for numbers". The authors recommend "use of recovery rates for community biomass when modeling trawl" impacts. (Fetched full text) - [Hiddink et al. 2017](https://ueaeprints.uea.ac.uk/64321/1/Published_manuscript.pdf)
- **Check (my calculation).** For the logistic model, the time from 0.5K to 0.95K is T = ln(19)/r = 2.944/r. At r = 0.82 this gives 3.59 years, which reproduces Hiddink's 3.6 years. With the model's r = 1.46 it gives 2.0 years, and with Pitcher's mud R = 0.68 it gives 4.3 years.
- **Pitcher et al. 2022, recovery by sediment.** "Average recovery rates ranged from 0.29 to 0.68 (lower confidence limits, CLs = 0.25 to 0.48) along a gravel to mud gradient." These were estimated "for an untrawled community ... including some slower-growing, larger-bodied, and longer-lived biota". "These rates correspond to a range of maximum longevities ... averaging 8 y in mud to 18 y in gravel (and up to 11 to 22 y, respectively, for lower CL recovery rates)." On sensitivity: "mud habitats have intermediate sensitivity because of high d-values and high R-values. All habitats are more sensitive to towed dredges and less sensitive to otter trawls." (Fetched full text) - [Pitcher et al. 2022](https://europepmc.org/article/PMC/PMC8764683)
- **Pitcher et al. 2022, equation and scale.**
  - Equation: "B/K = 1 - F D/R where F < R/D, otherwise B/K = 0". "R/D is the corresponding critical F".
  - Scale: grid cells "of size ~1 to 3 km2 - a scale at which the distribution of most individual trawls has been shown to be random". At that scale "D = d; however, R < r and is related to r and d through the equation R = rd/[-ln(1 - d)]" (Ellis et al.).
  - Mixed gears: "the cumulative RBS was estimated by summing the depletion (Fd/R) due to the d and F values for each gear."
  - (Fetched full text) - [Pitcher et al. 2022](https://europepmc.org/article/PMC/PMC8764683)
- **Pitcher et al. 2017, Methods Ecol. Evol. 8:472-480.** Introduced the relative benthic status (RBS) index as the equilibrium of the logistic equation. (Snippet) - [UEA record](https://ueaeprints.uea.ac.uk/61513/)
- **Hiddink et al. 2019, J. Appl. Ecol. 56:1075-1084, longevity.** Rijnsdorp et al. 2020 quote it as "r x longevity = 5.31 (upper 95% CL = 11.43, lower 95% CL = 2.43)". The PD (population-dynamic) method then sums over longevity classes: I_PD = 1 - sum_i K_i (1 - sum_m T_m d_m / r_i). (Fetched full text of Rijnsdorp 2020; Hiddink 2019 itself is a Snippet) - [Rijnsdorp et al. 2020](https://edepot.wur.nl/535091); [Hiddink et al. 2019 record](https://research.bangor.ac.uk/en/publications/assessing-bottom-trawling-impacts-based-on-the-longevity-of-benth/)
- **Rijnsdorp et al. 2020, North Sea mud habitat (Table 3, 2010-2012).**
  - Footprint 0.87 of the seafloor.
  - Surface trawling intensity within the footprint 3.05 /yr; subsurface 0.56 /yr.
  - PD biomass reduction 0.21. PD recovery time to 0.9K 7.6 months.
  - Reduction in median longevity (L2) 0.26.
  - The text adds: "muddy habitats were impacted the most".
  - (Fetched full text) - [Rijnsdorp et al. 2020](https://edepot.wur.nl/535091)

### Inferences
- **The model's r implies a short-lived community.** With r x longevity of about 5.31, r = 1.46 /yr implies an effective longevity of about 3.6 years. This is longevity in Hiddink 2019's sense (Pitcher calls it maximum longevity), not Rijnsdorp's community median. Hiddink's 0.82 implies about 6.5 years and Pitcher's mud 0.68 about 7.8 years, which matches Pitcher's "8 y in mud". So the model's r represents a community already shifted to short-lived taxa. That fits a heavily trawled Skagerrak seabed (see Q3), but it double-counts the trawling effect if trawling is then also applied on top. The literature applies Eq. 1 with r for the untrawled community. The rise in r with trawling is "relatively slight" over F = 0-1, and reaches 1.73 only at F = 10.
- **Consequence for equilibrium depletion.** Depletion relative to K is d*SAR/r. Going from r = 1.46 to 0.82 multiplies it by 1.78; going to 0.68 multiplies it by 2.15. The critical SAR at which B* = 0 falls the same way (see the table under Q4).
- **The literature r already includes predation.** Hiddink's 0.82 and Pitcher's R were fitted on real fishing grounds where predators were present. They are realized recovery rates, net of natural predation. The model takes gadoid predation explicitly, so using 0.82 as the model's logistic r would count predation twice: once inside r and once through gadoid intake. The right r for a model with explicit predation is therefore unresolved. It is bounded by 0.82 below and the current 1.46 above, and is probably somewhat above 0.82.
- **Scale.** The model's 1 km cells are at Pitcher's recommended scale, so D = d applies. The gear-to-cell correction R/r = d/[-ln(1-d)] is 0.97 for d = 0.06, 0.95 for d = 0.10 and 0.94 for d = 0.115 (my calculation). That correction is small next to the gap between the model's r and the literature r.

## Q3. Skagerrak, Kattegat, Nephrops and Pandalus grounds, and deep mud

### Takeaway
I found no experimental study that measures d specifically for Pandalus shrimp trawls. The regional evidence is the following:
- Pitcher et al. 2022 put Skagerrak-Kattegat among the three worst-status regions of the 24 assessed worldwide: mean RBS 0.633, regional SAR 3.33 /yr, 22.6% of the area at RBS = 0.
- In the Kattegat, Skold et al. 2018 sampled Nephrops otter-trawl grounds on mud at SAR 0-15 /yr. They found falling diversity, but almost no detectable net effect on total biomass (GAM R2 = 0.015). Burrowing brittle stars increased at medium intensity, which they attribute to release from predation by flatfish and Nephrops.
- In Gullmarsfjorden, an experiment with shrimp trawling (Hansson et al. 2000) found that biomass and abundance tended to decrease, and Amphiura decreased significantly after 7-12 months.
- "Light ground gear" does not apply to OT_CRU. The BENTHIS gear survey (Eigaard et al. 2016) classed the bobbin/roller/chain ground gear for Nephrops or shrimp on soft bottom as entirely subsurface. In the North Sea framework OT_CRU therefore gets the highest otter-trawl d (0.100).
- Deep-mud sea pens are the most sensitive part of the community. Funiculina cannot retract into the sediment and is in more than 50% of bycatches in the Skagerrak / Norwegian Trench.

### Cited Findings
- **Pitcher et al. 2022, SI Table S5, region 3 "Skagerrak and Kattegat (ICES 3a)", trawl data 2008-2010.**
  - Area 54,894 km2. Regional SAR 3.328. Uniform footprint 54.4%.
  - Mean RBS 0.633. Area with RBS > 0.8: 55.1%; RBS = 0: 22.60%; RBS = 1: 26.7%.
  - Main text: "Average RBS was <0.7 in three European regions: the Adriatic Sea (0.25), west of Iberia (0.60), and Skagerrak-Kattegat (0.63)". "Otter trawling is the most widespread and greatest contributor to cumulative reductions of RBS".
  - (Fetched full text and SI) - [Pitcher et al. 2022](https://europepmc.org/article/PMC/PMC8764683)
- **Skold et al. 2018, MEPS 586:41-55, Kattegat.**
  - Setting: mud, 23-65 m depth. Fishery "dominated by otter trawling targeting Norway lobster Nephrops norvegicus". The gradient includes a closed area (MPA).
  - "Trawling intensity estimated as swept area ratio ranged from 0 - 15 yr-1 ... (median 2.1 yr-1)."
  - The effect on "total biomass and abundance of benthos ... was negative for values below approx. 5 yr-1 ... and the relationship became slightly positive for larger values". The biomass model had R2 = 0.015.
  - "decrease in number of species and diversity from low to medium trawling intensities". "Amphiura chiajei increased in abundance from low to medium trawling intensities ... a consequence of reduction in predation by benthivorous flatfish and Nephrops norvegicus".
  - (Fetched full text, accepted manuscript) - [Skold et al. 2018, DTU Orbit](https://backend.orbit.dtu.dk/ws/portalfiles/portal/190977577/Microsoft_Word_Skold_et_al_Kattegatt_Revised_171124clean.docx.pdf)
- **Hansson et al. 2000, MEPS 198:191-201, Gullmarsfjorden.** A manipulative experiment over 1.5 years with 3 trawled and 3 control sites, in an area closed to trawling 6 years earlier. Gear and intensity matched pre-ban shrimp trawling. Biomass and abundance tended to decrease, few taxa differed significantly, and Amphiura brittle stars decreased significantly after 7-12 months. (Snippet; the abstract page returned 429) - [MEPS 198:191](https://int-res.com/abstracts/meps/v198/p191-201)
- **Eigaard et al. 2016, ICES J. Mar. Sci. 73(S1):i27-i43, ground gear.** "the OT bobbin/roller/chain groundgear for Nephrops or shrimp on soft bottom (OT_CRU) ... was assigned to have impacts entirely at the subsurface level". Otter trawling for Nephrops and mixed demersal fish sweeps about 1.2 km2 per fishing hour, "for which impact at the subsurface level is estimated to be the highest of all métiers (0.3 km2)". (Fetched full text) - [Eigaard et al. 2016, Ifremer](https://archimer.ifremer.fr/doc/00310/42137/41505.pdf)
- **Rijnsdorp et al. 2020.** OT_CRU covers "Nephrops, Pandalus, mixed fish". It ranked highest for impact: "otter trawls targeting crustaceans (OT_CRU) operate in muddy" habitats. (Fetched full text) - [Rijnsdorp et al. 2020](https://edepot.wur.nl/535091)
- **Buhl-Mortensen, Thangstad, Sovik & Wehde 2023, Mar. Biol. Res. 19:191-206, Skagerrak and Norwegian Trench.** The study used 543 bycatches from 2017-2021 and 35 ROV dives. Funiculina quadrangularis and Kophobelemnon stelliferum were most abundant below 200 m, and ROV densities were much higher than bycatch suggested. Funiculina was in more than 50% of trawl catches. Sea pen skeletons and a coral-eating anemone point to bottom-trawl pressure. (Snippet; the IMR repository did not resolve and Taylor & Francis returned 403) - [IMR record](https://imr.brage.unit.no/imr-xmlui/handle/11250/3093703)
- **MarLIN, sea pens and burrowing megafauna in circalittoral fine mud.**
  - Sensitivity rating: abrasion "High" sensitivity for the Funiculina-dominated biotope (low resistance, low resilience). Subsurface penetration "High" for all three sea pens.
  - Kenchington et al. 2011: otter-trawl catch efficiency for sea pens 3.7-8.2%.
  - Eno et al. 2001: 50% of displaced Funiculina re-established after 72 h; all Pennatula did.
  - Malecha & Stone 2009, Halipteris as a proxy: after 372 days, 92% of dislodged and 100% of fractured specimens were dead or had lost substantial tissue.
  - Tuck et al. 1998: no change in sea pen density over 18 months of experimental trawling in a Scottish sea loch.
  - Downie et al. 2021: 71% of suitable Funiculina habitat in the Greater North Sea is trawled. Sigwart et al. 2021: near-total loss in the North Sea.
  - (Fetched; secondary review) - [MarLIN biotope](https://www.marlin.ac.uk/habitats/detail/239/seapens_including_funiculina_quadrangularis_and_burrowing_megafauna_in_undisturbed_circalittoral_fine_mud)
- **Blomqvist, Svedang & Wikstrom 2018 (Havsmiljoinstitutet 2018:3, appendix 3 to the HaV report; authors confirmed on the title page).** The report says habitats defined by large upright species, such as "svampdjur och sjöpennor är speciellt känsliga för trålning och har ofta en lång återhämtningstid" ("sponges and sea pens are especially sensitive to trawling and often have a long recovery time"). Its reference list confirms the authors of the Gullmarsfjorden study as Hansson, Lindegarth, Valentinsson & Ulmestrand 2000. (Fetched full text) - [HaV appendix 3](https://www.havochvatten.se/download/18.47bf2cd7163855d85cab5edd/1527688697577/ru-fiske-marint-skyddade-omraden-bilaga-3-effekter-av-bottentralning-pa-ekosystemtjanster.pdf)

### Inferences
- The 35 mm mesh and the 19 mm sorting grid change what is retained in the catch. They do not change ground-gear contact, so they do not lower d. For seabed contact, the BENTHIS OT_CRU class (bobbin/roller/chain ground gear, fully subsurface) is the right proxy for the Skagerrak Pandalus and Nephrops trawls. That points to d at or above the pooled 0.06, not below it.
- Skold 2018 is the main warning against expecting a strong measurable biomass signal on Nephrops mud. Total biomass varied almost independently of SAR (R2 = 0.015), and part of the response ran through predator release. In the model, gadoid predation on the benthos is endogenous, so part of that food-web feedback is already represented. Note that the site is the shallow Kattegat, not the deep Skagerrak.
- The Kenchington per-pass catch efficiency of 3.7-8.2% for sea pens is of the same order as the community d. Sea pens differ mainly in their much slower recovery (long-lived, patchy recruitment). A single biomass pool cannot capture that. A sea pen sub-pool with r of about 0.1-0.3 /yr would be needed, which is my extrapolation and not a sourced value.

## Q4. Translation to the model (all INFERENCE)

### Recommended d for the model's bottom-trawl mix in muddy Skagerrak
- **Central value d = 0.10 per pass.** This equals Rijnsdorp's OT_CRU (Nephrops / Pandalus / mixed fish, 0.100) and lies between Hiddink's pooled OT (0.057-0.06) and Pitcher's predicted OT on mud (0.115). Most trawl hours in the grid are OT_CRU on mud.
- **Range 0.06-0.115.** The low end is the pooled global OT; the high end is the mud prediction. The formal 5-95% uncertainty is 0.02-0.16, from Hiddink via Rijnsdorp.
- **Use the biomass-based (pooled) d.** Hiddink found d does not differ between biomass and numbers, and the model's benthic_community is a single biomass. Pair it with a biomass recovery rate, never a numbers-based r: biomass recovers more slowly.
- **Keep d and SAR on the same gear-width basis.** Hiddink's d and Rijnsdorp's 0.100 are per pass of the whole gear (surface swept area, including doors and sweeps). If the sister note's SAR is a whole-gear SAR, use d = 0.10. If it is a subsurface-only SAR, scale d up by about 1/0.304 (OT_CRU) or 1/0.18 (average OT), giving roughly 0.33. Using a subsurface SAR with d = 0.10 would understate the loss about threefold.
- **Fisheries mix.** If the effort map can be split by fishery, sum d_m * SAR_m over fisheries, as Pitcher 2022 and Rijnsdorp 2020 do. Fish otter trawls (OT_DMF) get d of about 0.026.

### Implementation check
The model applies a loss per tick of d * SAR / 1460 to B. For a cell trawled at random, that is exactly the continuous loss rate d*F per year that Eq. 1 assumes. With logistic growth at r per year, the equilibrium is B*/K = 1 - d*SAR/r, and it is 0 when SAR >= r/d. Gadoid predation draws on the same production, so the model's actual equilibrium is lower: approximately 1 - (d*SAR + predation loss rate)/r. Per-tick loss with d = 0.10 is 6.8e-5 at SAR 1 and 3.4e-4 at SAR 5.

### Equilibrium B*/K = 1 - d*SAR/r (trawling only; my calculation)

| d | r (/yr) | source of r | SAR 1 | SAR 2 | SAR 5 | critical SAR r/d |
|---|---|---|---|---|---|---|
| 0.06 | 1.46 | model | 0.959 | 0.918 | 0.795 | 24.3 |
| 0.06 | 0.82 | Hiddink 2017 median | 0.927 | 0.854 | 0.634 | 13.7 |
| 0.06 | 0.68 | Pitcher 2022 mud R | 0.912 | 0.824 | 0.559 | 11.3 |
| 0.06 | 0.42 | Hiddink 2017 5% bound | 0.857 | 0.714 | 0.286 | 7.0 |
| 0.10 | 1.46 | model | 0.932 | 0.863 | 0.658 | 14.6 |
| 0.10 | 0.82 | Hiddink 2017 median | 0.878 | 0.756 | 0.390 | 8.2 |
| 0.10 | 0.68 | Pitcher 2022 mud R | 0.853 | 0.706 | 0.265 | 6.8 |
| 0.10 | 0.42 | Hiddink 2017 5% bound | 0.762 | 0.524 | 0 | 4.2 |
| 0.115 | 1.46 | model | 0.921 | 0.842 | 0.606 | 12.7 |
| 0.115 | 0.82 | Hiddink 2017 median | 0.860 | 0.720 | 0.299 | 7.1 |
| 0.115 | 0.68 | Pitcher 2022 mud R | 0.831 | 0.662 | 0.154 | 5.9 |

At the grid maximum effort, if that is around SAR 9-10 (to be confirmed by the sister note), d = 0.10 gives B*/K of about 0.34 with r = 1.46. With r = 0.82 or 0.68 the cell would be fully depleted, because the critical SAR is 8.2 or 6.8.

## Implications for the model
1. **Add the trawl loss with d = 0.10 as default and 0.06-0.115 as the sensitivity range.** This is inference built from three sourced values. Keep d and SAR on the same whole-gear basis (see Q4).
2. **The model's r = 1.46 /yr is optimistic against the literature, but the right value is unresolved.** The literature equilibrium framework uses r for the untrawled community: 0.82 /yr (Hiddink), or 0.68 /yr for mud with a lower CL of 0.48 (Pitcher). The Skagerrak-Kattegat regional RBS of 0.63 was computed with those R values. With 1.46, the trawl effect at a given SAR comes out about 1.8-2.1 times smaller than the literature implies, and the critical SAR about twice as high. However, the literature r values already include predation and the model adds gadoid predation explicitly (Q2 Inferences). So the model's r should lie between 0.82 and 1.46 /yr, not necessarily at 0.7-0.8. For benthic_community, r is the hand-set per-tick growth_rate (r/1460; this FG has no r_max), so 1.0 /yr would be about 0.00068 and 0.82 /yr about 0.00056. If r stays at 1.46, document that it represents an already trawl-shifted, short-lived community, with an effective longevity of about 3.6 years.
3. **Gadoid predation is the deciding factor.** Trawling and predation compete for the same r*B*(1-B/K) production. Lowering r to the literature value also lowers what gadoids can sustainably take. Choosing r is therefore a joint calibration of the trawl loss and the gadoid diet. Check the gadoid budget (budget_gate) after any change.
4. **No sediment split is needed inside the grid** if it is uniformly muddy. If the grid contains sand patches, the Pitcher sand value d = 0.047 is about 0.4 times the mud value; the sediment map should be verified separately.
5. **Sea pens and other long-lived epifauna are not represented** by a single fast-recovering biomass pool. The model will understate structural and biodiversity impacts on the Norwegian Trench slope even with a correct d.

## Source status

| Source | Status | Used for |
|---|---|---|
| Hiddink et al. 2017, PNAS 114:8301 | Fetched full text (author copy); SI Table S4 not retrieved | d by gear, biomass = numbers, r 0.82 (0.42-1.53), r rising to 1.73 at F 10, 3.6 y recovery, Eq. 1 |
| Pitcher et al. 2022, PNAS 119:e2109449119 | Fetched full text + SI (Tables S1-S5) | d by gear x sediment, R 0.29-0.68, R vs r scaling, Skagerrak-Kattegat RBS 0.633 / SAR 3.33 |
| Rijnsdorp et al. 2020, ICES JMS 77:1772 | Fetched full text | OT_CRU d 0.100, SSR scaling, Hiddink 5-95% ranges, PD method, mud indicators |
| Eigaard et al. 2016, ICES JMS 73:i27 | Fetched full text | OT_CRU ground gear fully subsurface |
| Skold et al. 2018, MEPS 586:41 | Fetched full text (accepted manuscript) | Kattegat Nephrops-ground SAR gradient, weak biomass response, predator release |
| MarLIN sea pen / burrowing megafauna review | Fetched (secondary review) | Sea pen sensitivity, Kenchington 3.7-8.2%, Eno, Malecha & Stone, Tuck |
| Blomqvist, Svedang & Wikstrom 2018, HaV appendix 3 | Fetched full text | Swedish context, Koster Pandalus fishery, sea pen sensitivity |
| Sciberras et al. 2018, Fish Fish. 19:698 | Fetched abstract (Crossref); full text 403 | 26% abundance per pass, mud / unfished areas more affected |
| Hiddink et al. 2019, J. Appl. Ecol. 56:1075 | Snippet; r x longevity = 5.31 only via Rijnsdorp 2020 | Longevity-recovery link |
| Pitcher et al. 2017, Methods Ecol. Evol. 8:472 | Snippet | Origin of RBS equilibrium |
| Hansson et al. 2000, MEPS 198:191 | Snippet (429 on the abstract page) | Gullmarsfjorden shrimp-trawl experiment |
| Buhl-Mortensen et al. 2023, Mar. Biol. Res. 19:191 | Snippet (repository unreachable, T&F 403) | Skagerrak / Norwegian Trench sea pens |
| ICES WGFBIT reports | Not retrieved | Gap: did not confirm which d values ICES currently uses for 3a |
