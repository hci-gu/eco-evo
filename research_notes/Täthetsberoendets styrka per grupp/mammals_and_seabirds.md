# Density dependence of M1 for marine mammals (harbour porpoise, seals) and seabirds: where it acts, how strong it is, and how much of it belongs in `natural_mortality`

Written 2026-10-15 for the Mareld / Poseidon Nord model. No code or config was edited.
Earlier work this builds on, not redone here: r_max, M1 and demography in
`research_notes/Litteraturbaserad growth rate per FG/mammals.md` and `.../seabirds.md`.

## 0. Conventions

**Read status of each source.**
- **[F]** = I read the full text (PDF or HTML).
- **[A]** = I read the abstract only (OpenAlex, Europe PMC or the publisher page).
- **[S]** = search-engine snippet or summary only. Numbers tagged [S] are not verified.
- **[2]** = secondary: the claim is cited inside another source that I read, and the original was not opened.
- **[N]** = not obtained. Failed fetches are listed in section 7.

**Labels.** Each section separates **Literature** (what the sources say), **Calculation** (arithmetic I did, with every input shown) and **Inference** (my judgement).

**Model facts used** (from the assignment and the repo):
- Porpoise: r_max = 0.10/yr, M1 = 0.10/yr, B_ref = 12 t on the 60x60 km grid, with no predator and no bycatch impact in `mareld2.yaml`.
- Seals: r_max 0.11, M1 0.10. Seabirds: r_max 0.10, M1 0.15. Both groups are muted.
- Food limitation is explicit in the model. Starvation is about 39 % of porpoise losses in the current run, and reproduction runs through the reserve, q = s - u.
- The site is at about 58.1 N, 10.5-10.8 E (`coordinates.txt`, `mareld2.yaml` latitude 58.15), i.e. the Skagerrak.

**Definitions** (from the assignment):
- beta_M = dM1/d ln B at B_ref, per year. beta_M > 0 means compensatory density dependence (M1 rises with B).
- gamma = beta_M / M1(B_ref), dimensionless. For the form M1 = M1_ref (B/B_ref)^gamma, gamma is exactly that elasticity.
- With a density-independent floor, M1 = M_floor + M_dd (B/B_ref)^g', the elasticity at B_ref is gamma = g' * M_dd / M1_ref. So g' = gamma * M1_ref / M_dd is larger than gamma.
- B is the **grid-total** biomass. The mechanisms below act at population or colony scale, not per 1 km cell (see section 6.4).

---

## 1. Framework: converting theta-logistic shape and B/K into beta and gamma

### Calculation
Theta-logistic per-capita net growth: g(B) = r [1 - x^theta], with x = B/K.
- Total density-dependence strength: beta_tot(x) = -dg/d ln B = -x dg/dx = r theta x^theta, per year. At K it is r*theta.
- Maximum net productivity level: MNPL/K = (1 + theta)^(-1/theta). Inverting gives theta = 1, 2.39, 3.48, 5.04, 11.2 and 18.2 for MNPL/K = 0.50, 0.60, 0.65, 0.70, 0.80 and 0.85 (solved numerically).
- Equilibrium under a constant removal fraction h: r[1 - x^theta] = h, so x^theta = 1 - h/r and beta_tot = theta (r - h). This closed form is used for bycatch in section 3.

How the total splits into births and deaths:
- g = b(B) - M(B) and beta_tot = beta_b + beta_M, where beta_b = -db/d ln B.
- Only beta_M belongs in M1: beta_M = phi * beta_tot, where phi is the share of the total that acts through mortality.
- The model already runs part of the mortality share explicitly (starvation), and part of the fecundity share through the reserve, (s - u). So M1 should carry only phi_nf * beta_tot, where phi_nf is the **non-food** share of the mortality response (disease, crowding, space).
- Hence gamma = phi_nf * r * theta * x^theta / M1_ref.

Upper bound, with every bit of density dependence in M1 (phi = 1) and B_ref = K:
- gamma_max = r * theta / M1_ref.
- Porpoise (r = 0.10, M1 = 0.10): gamma_max = theta, i.e. 1.0 at theta = 1 and 2.4 at theta = 2.39.

### Inference
- A gamma near theta (about 1-2.4) is the "all density dependence in M1" ceiling. In this model it would double-count the starvation channel.
- A defensible gamma is therefore phi_nf times that ceiling.
- Sections 2-5 constrain theta, x = B_ref/K and phi_nf for each group.

---

## 2. Where density dependence acts in long-lived mammals (general theory)

### Literature
- **Fowler (1981)**: "most density-dependent change occurs at high population levels (close to the carrying capacity) for species with life history strategies typical of large mammals"; the reverse holds for insects and some fishes [A]. [Fowler, C.W. 1981. Density dependence as related to life history strategy. Ecology 62: 602-610. doi:10.2307/1937727](https://doi.org/10.2307/1937727)
- **Eberhardt (1977)**: self-regulation in long-lived species proceeds in stages: "Immature mortality is postulated as the first stage, followed by age of first reproduction and then reproductive rates of adult females. The fourth and last stage, as populations reach maximum sustainable levels, is assumed to involve changes in adult survival." He also says the MSY point "may be well above the logistic-curve 50% point" [A]. [Eberhardt, L.L. 1977. Optimal policies for conservation of large mammals, with special reference to marine ecosystems. Environ. Conserv. 4: 205-212. doi:10.1017/S0376892900025819](https://doi.org/10.1017/S0376892900025819)
- **Eberhardt (2002)**: "early survival decreases first, followed by lower reproductive rates; ultimately, adult female survival may decrease"; the sensitivity of lambda follows the same sequence [A]. [Eberhardt, L.L. 2002. A paradigm for population analysis of long-lived vertebrates. Ecology 83: 2841-2854. doi:10.1890/0012-9658(2002)083[2841:APFPAO]2.0.CO;2](https://doi.org/10.1890/0012-9658(2002)083[2841:APFPAO]2.0.CO;2)
- **Gaillard et al. (2000)**, large herbivores: "Regardless of the source of variation, adult female survival shows little year-to-year variation [CV <10%]". Prime-age fecundity and yearling survival vary moderately (CV <20%); juvenile survival and the fecundity of young females vary strongly (CV >30%). The authors note that density dependence and environmental variation "have similar effects on various fitness components" [A]. [Gaillard, J.-M., Festa-Bianchet, M., Yoccoz, N.G., Loison, A. & Toigo, C. 2000. Temporal variation in fitness components and population dynamics of large herbivores. Annu. Rev. Ecol. Syst. 31: 367-393. doi:10.1146/annurev.ecolsys.31.1.367](https://doi.org/10.1146/annurev.ecolsys.31.1.367)
- **Gaillard et al. (1998)**, TREE 13: 58-63, "variable recruitment with constant adult survival": only the title and secondary descriptions were found [2/S]. [doi:10.1016/S0169-5347(97)01237-8](https://doi.org/10.1016/S0169-5347(97)01237-8)
- **Taylor & DeMaster (1993)** [F] [Taylor, B.L. & DeMaster, D.P. 1993. Implications of non-linear density dependence. Mar. Mamm. Sci. 9: 360-371 (NOAA SWFSC PDF)](https://storage.googleapis.com/nmfs_odp_swfsc/Publications/Files/CR/1993/9388.pdf)
  - Counts from Fowler (1984): "evidence of regulation for fecundity in nine species, age of first reproduction (AFR) in ten species, juvenile survival in five species and adult survival in one species".
  - "The lack of empirical evidence makes the form of density dependence for adult survival purely speculative" (p. 361).
  - Conclusion: "It seems likely that 0.5 < MNPL/K < 0.85" (p. 369). The authors say MNPL/K > 0.8 is "unsupported", and density dependence "is likely to involve more than a single parameter".
  - Their Table 3, odontocete model (lambda_max 1.04): if adult survival alone regulated the population, it would have to fall from 0.999 at N = 0 to 0.961 at K. In the pinniped model (lambda_max 1.10) it falls from 1.000 to 0.914.
  - Elasticity example: a 10 % cut in adult survival, juvenile survival or birth rate changes lambda by -7.5 %, -2.4 % or -1.2 %. Adult survival is the most powerful lever, so small changes in it would suffice.
  - On Eberhardt's rigid ordering: Siniff (1984) found that three sympatric Antarctic seals responded differently.
  - Fur-seal example: MNPL/K 0.58-0.73, with highest frequency at 0.60-0.65 (Ragen 1990, as cited).
- **Wade (1998)**: PBR = N_min * 0.5 R_max * F_R, with R_max = 0.04 for cetaceans and 0.12 for pinnipeds. It is evaluated against MNPL [A]. [Wade, P.R. 1998. Calculating limits to the allowable human-caused mortality of cetaceans and pinnipeds. Mar. Mamm. Sci. 14: 1-37. doi:10.1111/j.1748-7692.1998.tb00688.x](https://doi.org/10.1111/j.1748-7692.1998.tb00688.x)
- **Counter-evidence on curve shape.**
  - Sibly et al. (2005), across 1780 time series of mammals, birds, fish and insects: growth rate "decline[s] rapidly with increasing population size and then flatten[s] out", i.e. strongly concave (theta < 1) [A]. [Sibly, R.M., Barker, D., Denham, M.C., Hone, J. & Pagel, M. 2005. Science 309: 607-610. doi:10.1126/science.1110760](https://doi.org/10.1126/science.1110760)
  - Clark et al. (2010) reply that theta and r_m "play-off against each other (providing a constant product)", that theta-logistic fits to census data are not robust, and they "refute the claim that concave growth responses are shown to dominate in nature" [A]. [Clark, F., Brook, B.W., Delean, S., Akcakaya, H.R. & Bradshaw, C.J.A. 2010. The theta-logistic is unreliable for modelling most census data. Methods Ecol. Evol. 1: 253-262. doi:10.1111/j.2041-210X.2010.00029.x](https://doi.org/10.1111/j.2041-210X.2010.00029.x)

### Calculation
- MNPL/K of 0.5-0.85 (Taylor & DeMaster) corresponds to theta = 1-18. Their likely core of 0.60-0.65 corresponds to theta = 2.4-3.5.
- Adult-survival-only regulation in the odontocete model (Taylor & DeMaster Table 3) means adult mortality rises from -ln 0.999 = 0.001 to -ln 0.961 = 0.040/yr, so dM = 0.039/yr, about lambda_max - 1 = 0.04. In the pinniped model it rises from 0 to -ln 0.914 = 0.090/yr.
- General rule: if mortality alone did all the regulating, it would have to rise by about r_max between N -> 0 and K.
- The Clark et al. point helps here. The identifiable quantity is the product r*theta, which equals beta_tot at K. That is exactly the quantity this note needs, and it can be estimated even when r and theta cannot be separated (used for harbour seals in section 4).

### Inference
- **The ordering matters most.** Every source agrees that adult survival responds last and least, and that juvenile survival, age at first reproduction and fecundity carry most of the density dependence.
  - In an age-less biomass pool, fecundity and AFR belong to the growth term, which the model already ties to the reserve.
  - Juvenile survival is formally mortality, but the calves are a small share of the biomass.
- **Share acting through mortality.** Fowler's (1984) counts are 1 of 25 species-parameter cases for adult survival, and 5 + 1 = 6 of 25 for any survival. I take the mortality share phi as roughly 0.2-0.3 overall, with 0.05 or less for adults alone.
- **Shape.** Theta > 1 (convex, density dependence concentrated near K) is the theoretical and management default for mammals, though Sibly et al. (2005) dispute it. Theta = 2.4 (MNPL/K = 0.6) is used as the central value below, and theta = 1 as a weaker alternative.

---

## 3. Harbour porpoise

### Literature: evidence of food-mediated density dependence (body condition, pregnancy, starvation)
- **IJsseldijk et al. (2021)**, Dutch necropsies plus 16 other areas: "females in poor health had lower probabilities of being pregnant and generally did not sustain pregnancy throughout gestation". Maternal nutritional status affected foetal size, and "Pregnancy rates across the Northern Hemisphere were best explained by" the mean energy density of prey in the diet (MEDD) [A]. [IJsseldijk, L.L., Hessing, S., Mairo, A. et al. 2021. Nutritional status and prey energy density govern reproductive success in a small cetacean. Sci. Rep. 11. doi:10.1038/s41598-021-98629-x](https://doi.org/10.1038/s41598-021-98629-x)
- **MacLeod et al. (2007)**, Scottish North Sea: starvation was the cause of death for 33 % of porpoises in spring 2002-2003, against 5 % in the 1993-2001 baseline. It was linked to a lower share of sandeel in the diet [A]. [MacLeod, C.D., Santos, M.B., Reid, R.J., Scott, B.E. & Pierce, G.J. 2007. Biol. Lett. 3: 185-188. doi:10.1098/rsbl.2006.0588](https://doi.org/10.1098/rsbl.2006.0588)
- **Learmonth et al. (2014)**, Scotland: pregnancy rate 0.34-0.40, "lower than recorded elsewhere, but is likely underestimated due to the prevalence of mature females of poor health status in the sample" [A]. [Learmonth, J.A., Murphy, S., Luque, P.L. et al. 2014. Mar. Mamm. Sci. 30: 1427-1455. doi:10.1111/mms.12130](https://doi.org/10.1111/mms.12130)
- **Murphy et al. (2015)**, UK: 19.7 % of mature females showed direct reproductive failure. Pregnancy was about 50 % even in healthy females that died of trauma. PCB burden and health status predicted reproductive status [A]. [Murphy, S., Barber, J.L., Learmonth, J.A. et al. 2015. PLoS ONE 10: e0131085. doi:10.1371/journal.pone.0131085](https://doi.org/10.1371/journal.pone.0131085)
- **Read & Hohn (1995)**, Gulf of Maine: "Most female porpoises matured at age three and became pregnant each year thereafter" [A]. The paper itself does not compare pregnancy with density. [Read, A.J. & Hohn, A.A. 1995. Life in the fast lane. Mar. Mamm. Sci. 11: 423-440. doi:10.1111/j.1748-7692.1995.tb00667.x](https://doi.org/10.1111/j.1748-7692.1995.tb00667.x)
- **Kesselring et al. (2017)**, German North Sea and Baltic: female maturity at 4.95 yr and mean age at death 5.70 / 3.67 yr. Read in the earlier note [F there]; no density comparison.

### Literature: causes of natural death (to separate food-mediated from other mortality)
- **Swedish strandings 2006-2020** (n = 128, almost all from the North Sea and Belt Sea populations): bycatch plus probable bycatch 31.4 %, undetermined 18.5 %, infectious disease 17.6 %, emaciation 10.2 %, trauma 10.2 %, abandoned (neonates) 8.3 %. "13 (68.4%) of the 19 porpoises that succumbed to infectious disease were in poor to emaciated nutritional condition"; the authors say it is "difficult to assess whether compromised nutritional status led to impaired immunity and disease, or whether it was the result of disease" [F]. [Neimanis, A., Stavenow, J., Agren, E.O., Wikstrom-Lassa, E. & Roos, A.M. 2022. Causes of death and pathological findings in stranded harbour porpoises from Swedish waters. Animals 12: 369 (PMC8833703)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8833703/)
- **Dutch post-mortem programme 2024** (n = 65): infectious disease 36.9 %, bycatch 15.4 %, blunt trauma 10.8 %, neonatal problems 12.3 %, grey seal attack 4.6 %, starvation or emaciation 1.5 % [A, Dutch abstract]. [van Schalkwijk, L., Schotanus, Grone, A. & IJsseldijk, L.L. 2025. Postmortaal onderzoek van bruinvissen uit Nederlandse wateren, 2024. WOT Natuur & Milieu. doi:10.18174/692222](https://doi.org/10.18174/692222)
- **Grey seal predation** (Leopold et al. 2015): in the Netherlands, 25 % of 271 sufficiently fresh porpoises (2003-2013) had grey seal bite marks. Victims were "mostly healthy juveniles that had a thick blubber layer"; predation is "one of the main causes of death in harbour porpoises in The Netherlands" [A]. [Leopold, M.F., Begeman, L., van Bleijswijk, J.D.L. et al. 2015. Exposing the grey seal as a major predator of harbour porpoises. Proc. R. Soc. B 282: 20142429. doi:10.1098/rspb.2014.2429](https://doi.org/10.1098/rspb.2014.2429)
  - Citation note: the assignment's "Leopold et al. 2015 on starvation" seems to mix this paper up with Leopold et al. 2015, "Porpoises: from predators to prey", J. Sea Res. 97: 14-23 ([doi:10.1016/j.seares.2014.12.005](https://doi.org/10.1016/j.seares.2014.12.005)), whose abstract I could not obtain [N]. The starvation evidence above comes from MacLeod, IJsseldijk and Neimanis.
- **No study found tests density dependence in porpoise vital rates** (survival or pregnancy against population density over time). The evidence is food and condition against vital rates, which is a mechanism, not a measured density response.

### Literature: population status relative to K
**Management units.** The Belt Sea population covers the western Baltic, the Belts, the Sound and the southern Kattegat, bounded by "an east-west line ... at 56.95N in the Kattegat". The North Sea population "rang[es] from the northern Kattegat through the Skagerrak to the whole North Sea" [F]. [Owen, K., Gilles, A., Authier, M. et al. 2024. A negative trend in abundance and an exceeded mortality limit call for conservation action for the Vulnerable Belt Sea harbour porpoise population. Front. Mar. Sci. 11: 1289808. doi:10.3389/fmars.2024.1289808](https://doi.org/10.3389/fmars.2024.1289808)

**The site is at 58.1 N, so it belongs to the North Sea assessment unit.**

North Sea AU:
- Abundance 289,200 (1994), 355,400 (2005), 345,400 (2016); "estimated yearly change ... 0,00 (p-value = 0,93)"; "no decline in abundance was evidenced" [F]. [OSPAR QSR 2023, Abundance and distribution of cetaceans](https://oap.ospar.org/en/ospar-assessments/quality-status-reports/qsr-2023/indicator-assessments/abundance-distribution-cetaceans/)
- SCANS-IV 2022: 338,918 (95 % CI 243,063-476,203), CV 0.17, mean density 0.55 animals/km2. Trend +0.51 %/yr (95 % CI -1.14 to 2.20, p = 0.32). The four surveys could only detect declines of 0.88 %/yr or more at 80 % power [F]. [Gilles, A. et al. 2023. Estimates of cetacean abundance in European Atlantic waters in summer 2022 from the SCANS-IV aerial and shipboard surveys (Tables 29-30, Fig. 8)](https://tethys.pnnl.gov/sites/default/files/publications/Gilles_etal_2023.pdf)
- Bycatch (OSPAR QSR 2023): N_best 345,000. The threshold of 1,622/yr comes from the Removals Limit Algorithm (RLA) under the objective "80% of K after 100 years with probability 0.8". The 2020 bycatch estimate was 5,974 (95 % CI 3,176-10,739), or 1,627 (922-3,325) with one country's data removed. "both estimates result in the threshold being exceeded" [F]. [OSPAR QSR 2023, Marine mammal bycatch](https://oap.ospar.org/en/ospar-assessments/quality-status-reports/qsr-2023/indicator-assessments/marine-mammal-bycatch/)
- Local density: in SCANS-III 2016, aerial block P1 "Skagerrak/Kattegat" (23,451 km2) held 1.077 animals/km2 (CV 0.30; 25,367 animals) and block P held 0.823/km2 [F]. [Hammond, P.S. et al. 2021. Estimates of cetacean abundance in European Atlantic waters in summer 2016 from the SCANS-III aerial and shipboard surveys (revised), Table 6](https://scans3.wp.st-andrews.ac.uk/files/2021/06/SCANS-III_design-based_estimates_final_report_revised_June_2021.pdf)
  - I could not tell which SCANS-IV block covers the Skagerrak: the map was not readable, so the NS-L and NS-M densities are not used.

Belt Sea AU (for contrast; it reaches the southern Kattegat):
- Abundance 27,901 (2005), 40,475 (2012), 42,324 (2016), 17,301 (2020), 14,403 (2022). Trend -2.68 %/yr, 90.5 % probability of decline.
- Owen et al. assume K = 50,000 (the 1994 estimate, "rough"). Their mPBR is 24/yr, with R_max 0.04, N_min 12,091 and F_R 0.1.
- Bycatch is about 900/yr from Danish and Swedish set nets (Kindt-Larsen et al. 2023, as cited), possibly up to 3,000 [F]. [Owen et al. 2024](https://doi.org/10.3389/fmars.2024.1289808)

### Calculation
**Bycatch as a fraction of abundance:**
- North Sea: 5,974/345,000 = 1.73 %/yr, or 1,627/345,000 = 0.47 %/yr. The threshold 1,622/345,000 is 0.47 %.
- Belt Sea: 900/14,403 = 6.2 %/yr (up to 3,000/14,403 = 21 %).
- Belt Sea x = 14,403/50,000 = 0.29.

**North Sea equilibrium x with h = 0.0047-0.0173, using x^theta = 1 - h/r:**

| r | theta | x at h = 0.0047 | x at h = 0.0173 |
|---|---|---|---|
| 0.10 | 1 | 0.95 | 0.83 |
| 0.10 | 2.39 | 0.98 | 0.92 |
| 0.04 | 1 | 0.88 | 0.57 |
| 0.04 | 2.39 | 0.95 | 0.79 |

**beta_tot at that equilibrium = theta (r - h):**
- r = 0.10: 0.083-0.095 (theta = 1) and 0.20-0.23 (theta = 2.39).
- r = 0.04: 0.023-0.035 (theta = 1) and 0.054-0.084 (theta = 2.39).

**gamma = phi_nf * r * theta * x^theta / M1, with r = 0.10, theta = 2.39 and M1 = 0.10:**

| x | beta_tot (/yr) | phi_nf = 0.05 | 0.10 | 0.20 | 0.30 | 1.0 |
|---|---|---|---|---|---|---|
| 1.0 | 0.239 | 0.12 | 0.24 | 0.48 | 0.72 | 2.39 |
| 0.9 | 0.186 | 0.09 | 0.19 | 0.37 | 0.56 | 1.86 |
| 0.8 | 0.140 | 0.07 | 0.14 | 0.28 | 0.42 | 1.40 |

**Model density against survey density:**
- B_ref = 12 t / 3,600 km2 = 3.33 kg/km2, which is 0.056-0.067 animals/km2 at 50-60 kg.
- At the SCANS-III P1 density of 1.077/km2, the grid would hold about 3,900 animals, i.e. 194-233 t. At the North Sea mean of 0.55/km2 it would hold 1,980 animals, i.e. 99-119 t.
- So B_ref is about 0.06-0.11 of the survey-based biomass for this area.

### Inference
- **The North Sea population, which includes the Skagerrak, is probably close to K, held only slightly below it by bycatch.**
  - Abundance has been flat for about 28 years under 0.5-1.7 %/yr bycatch.
  - With porpoise-specific r of about 0.10 and theta of 1-2.4, equilibrium x is about 0.83-0.98. With the conservative r = 0.04, x is about 0.57-0.95.
  - The flat trend is also consistent with a slowly changing K: the distribution moved from the north-west to the south between 1994 and 2005 (Gilles et al. 2023 [F]). So K itself is not well defined.
- **The Belt Sea population, which reaches the southern Kattegat, is far below K and declining under bycatch** far above its limit. It is not the population at the site, but it is the one whose numbers are most often quoted for the Swedish west coast.
- **Food is the dominant density-dependent pathway in porpoises.** Pregnancy follows prey energy density and body condition, starvation appears when key prey fail, and about two-thirds of infectious-disease deaths occur in poor or emaciated animals. The model represents this pathway explicitly through starvation and reserve-dependent growth.
- **Non-food density-dependent mortality is weakly supported.** Candidates are disease transmission and social or space effects. Porpoises live in small groups, without haul-outs or colonies, so contact-driven epizootics of the seal kind are not documented.
- **phi_nf: 0.05 central, range 0-0.15.**
- **Grey seal predation** is a real, unmodelled source of natural mortality. It is not density dependent in porpoise numbers in any documented way, and seals are muted. It belongs in a density-independent M1, not in gamma.
- **Model caveat 1: no bycatch.** `mareld2.yaml` has no porpoise bycatch, so the model population settles at its own food-limited K, where x = 1 and beta_tot = r * theta is at its maximum. If today's real population sits at x of about 0.85-0.95, the model will drift 5-15 % above today's level.
- **Model caveat 2: low B_ref.** B_ref is about 1/10-1/17 of the survey-based local biomass (Calculation above). This deserves a separate check of how 12 t was chosen. If B_ref is far below the real local K, the "near K" reasoning applies to the model's own K, not to the real stock.

---

## 4. Seals

### Literature: harbour seal, Kattegat-Skagerrak
**Carroll et al. (2025)** [F, including Supplementary Tables S2 and S3] [Carroll, D., Ahola, M.P., Carlsson, A.M., Galatius, A., Nilssen, K.T., Harkonen, T. & Harding, K.C. 2025. Declining harbour seal abundance in a previously recovering meta-population. PLoS ONE 20(6): e0326933 (PMC12208499)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208499/)
- Logistic fit to the 2003-2023 moult counts (S2 Table):
  - Kattegat-Skagerrak: K = 13,965 +/- 772 counted seals, N0 (2003) = 5,930 +/- 1,290, **mu = 0.28 +/- 0.10/yr** (p = 0.016).
  - Kattegat: K 9,509, mu 0.25 +/- 0.08. Skagerrak: K 4,517, mu 0.33 +/- 0.15.
  - The logistic model beat the exponential on AIC.
- Counts: maximum 14,189 in 2017; 12,507 in 2023 (S3 Table). The Skagerrak peaked at 4,767 in 2016 and was 3,710 in 2023.
- PDV outbreaks in 1988 and 2002 "killed up to 60% of the population on each occasion".
- After 1988 the population grew 10.2-13.6 %/yr. Growth "began to slow with mean counts of approximately 10,000" before 2002, and was 5 % (Kattegat) and 6.5 % (Skagerrak) after 2002.
- Adult survival is "often > 95% in harbour seals with variation in population growth being primarily driven by birth rates and juvenile survival".
- The authors write that the recent decline "may be the result of well documented environmental degradation affecting prey availability", that "a mixture of density dependent and independent control of population growth is likely", and that haul-out behaviour may have changed with density.

**Harding et al. (2018)**: in 2002, asymptotic length in the Skagerrak was 7 cm shorter than in 1988, while the Kattegat was unchanged. Seal density per km2 of feeding ground was three times higher in the Skagerrak. The authors call this "an early signal of density dependence", expected to work through delayed maturity, higher juvenile mortality and lower fecundity: a "smooth route towards carrying capacity" [A]. [Harding, K.C., Salmon, M., Teilmann, J., Dietz, R. & Harkonen, T. 2018. Population wide decline in somatic growth in harbor seals - early signs of density dependence. Front. Ecol. Evol. 6: 59. doi:10.3389/fevo.2018.00059](https://doi.org/10.3389/fevo.2018.00059)

**Harkonen et al. (2002)**: intrinsic rates "cannot exceed 13% per year (lambda_1max = 1.13)". Higher observed rates reflect non-stable age structure [A]. [Can. J. Zool. 80: 1498-1510. doi:10.1139/z02-141](https://doi.org/10.1139/z02-141)

### Literature: is epizootic mortality density dependent?
- **Harkonen et al. (2006)** [F] [Harkonen, T., Dietz, R., Reijnders, P. et al. 2006. The 1988 and 2002 phocine distemper virus epidemics in European harbour seals. Dis. Aquat. Org. 68: 115-130. doi:10.3354/dao068115](https://doi.org/10.3354/dao068115)
  - More than 23,000 seals died in 1988 and more than 30,000 in 2002.
  - In 1988, mortality exceeded 50 % in the Kattegat, Skagerrak, Baltic, Wadden Sea and Wash, all reached in summer and early autumn. Late-autumn areas lost much less (Moray Firth 13 %, Orkney and Shetland possibly 1 %), "since contact among seals decreases in late autumn".
  - In 2002, "the highest mortality was observed in the northern Skagerrak, where 66% died".
  - Mortality in the Danish Kattegat and the Baltic was lower in 2002 (Baltic 16 % vs 50 %), possibly because of immunity. The paper makes no explicit test against density.
- **Harding, Harkonen & Caswell (2002)**: the 1988 outbreak "killed 58% of the population". Recurrent epizootics "would reduce the long-term stochastic growth rate ... by half" [A]. [Ecol. Lett. 5: 727-732. doi:10.1046/j.1461-0248.2002.00390.x](https://doi.org/10.1046/j.1461-0248.2002.00390.x)
- **Swinton et al. (1998)**: high coupling between colonies plus a small annual birth cohort "act to make persistence of infection impossible in this harbour seal population at realistic population levels". So PDV is introduced, not endemic [A]. [J. Anim. Ecol. 67: 54-68. doi:10.1046/j.1365-2656.1998.00176.x](https://doi.org/10.1046/j.1365-2656.1998.00176.x)
- **Lonergan et al. (2010)**, UK: the differences between the 1988 and 2002 epizootics were "primarily due to a 27% ... fall in R0". Geographic differences were "mainly ... differences in case mortality" [A]. [Dis. Aquat. Org. 88: 183-188. doi:10.3354/dao02153](https://doi.org/10.3354/dao02153)

### Literature: grey seal
**SCOS 2022 (UK)** [F] [Special Committee on Seals 2022. Scientific advice on matters related to the management of seal populations: 2022. SMRU](https://www.smru.st-andrews.ac.uk/files/2023/09/SCOS-2022.pdf)
- The population model includes density dependence "in pup survival (but not in adult female fecundity)". "density dependence acts through pup survival only, so adult survival does not vary with time or between regions"; the posterior adult survival is a constant 0.96 (SE 0.01). Fecundity is 0.90.
- Maximum unconstrained pup survival is 0.46 (SE 0.07). Current pup survival is 0.43 in the North Sea, where density dependence "is having little effect", and 0.11 in the Hebrides and Orkney. This is close to the 0.1-0.14 expected at K (Thomas et al. 2019).
- "Model selection criteria suggest that density dependence is acting mainly on pup survival". The density-dependent pup-survival model fits better than one with density-dependent fecundity.
- UK pup production (2019) was 67,850; the 2022 population was 162,000.
- Environmental and fecundity links:
  - Kauhala et al. (2019): Baltic pregnancy rates of 0.6-0.95, tied to herring and sprat weight.
  - Badger et al. (2020): at higher density, poor-quality females' reproduction declined while high-quality females' did not.
- **Thomas et al. (2019)**: an age-structured Bayesian state-space model "which includes density dependence in pup survival, with separate carrying capacities" in four regions. Three regions "appear to have reached carrying capacity" [A]. [Thomas, L., Russell, D.J.F., Duck, C.D. et al. 2019. Aquat. Conserv. 29(S1): 6-23. doi:10.1002/aqc.3134](https://doi.org/10.1002/aqc.3134)

### Calculation
**Harbour seal r*theta from Carroll:**
- The logistic slope at K is -mu. By Clark et al., r*theta is what a census fit identifies. So beta_tot at K is about 0.28 (0.18-0.38)/yr.
- With r = ln 1.12 = 0.113 (`mammals.md` section 2): theta = 0.28/0.113 = 2.5 (range 1.6-3.4), i.e. MNPL/K = 0.60 (0.55-0.65). This agrees with Taylor & DeMaster's 0.60-0.65 core.
- Today x = 12,507/13,965 = 0.90, so x^theta = 0.76 (0.69-0.84) and **beta_tot(today) = 0.21 (0.15-0.26)/yr**.
- Check: the logistic predicts 0.28 x (1 - 0.90) = +0.029/yr growth in 2023, whereas counts declined. This is consistent with the authors' view that density-independent deterioration (prey) is also acting.
- Caveats on mu = r*theta:
  - The step matches the slope at K of the fitted logistic with that of a theta-logistic. It is a local linearisation, not an identity.
  - The fit starts in 2003, right after a 50-60 % kill. Carroll et al. and Harkonen et al. (2002) both say a post-epizootic age structure skewed towards young females inflates observed growth. That bias, together with possible haul-out changes in the late counts, pushes mu, and hence the 0.21/yr, upward. Treat it as upper-leaning.

**gamma = phi_nf * 0.21 / 0.10:**
- phi_nf = 0.1 gives gamma = 0.21; 0.2 gives 0.43; 0.3 gives 0.64.
- At K: phi_nf = 0.1 gives 0.28.

**PDV mortality averaged over time:**
- One event killing a fraction k every T years averages to -ln(1 - k)/T.
- k = 0.5-0.58 with T = 14 (1988 to 2002) gives 0.050-0.062/yr. k = 0.55 with T = 17.5 (two events in 35 years, 1988-2023) gives 0.046/yr.
- If that mean loss scaled with density at elasticity e (e = 1 for simple mass-action transmission), its contribution to gamma would be e * 0.05/0.10 = 0.5e. That holds only if the 0.10 M1 excludes epizootics. The Leslie-based M1 in `mammals.md` is built on inter-epizootic survival, so it probably does.

**Grey seal pup density dependence weighted by biomass:**
- Pups per 1+ animal: 67,850/162,000 = 0.42/yr.
- Extra first-year loss at K: 0.42 x (0.46 - 0.12) x (weaning mass / mean 1+ mass).
- With 40/150, 45/125 and 50/100 kg (assumed masses, not sourced here) that is 0.038, 0.051 and 0.071 of standing biomass per year, i.e. 0.4-0.7 x M1.
- If this rose linearly with N, its slope at K would equal that same change (dM/d ln N = dM * N/K = dM at K).

### Inference
- **Harbour seal.** The Kattegat-Skagerrak population is at about 0.9 K. Its total density dependence (about 0.2/yr) is consistent with mammal theory (theta about 2.5).
  - Most of it shows up as somatic growth, pup counts and birth rates, with adult survival near 0.95. That is the food channel, which the model would capture explicitly.
  - The non-food candidates are PDV and haul-out crowding.
  - PDV acts as a rare, introduced pulse whose severity depended on season and immunity, not demonstrably on density. The 66 % loss in the high-density northern Skagerrak in 2002 (Harkonen 2006; Harding 2018) is suggestive but anecdotal.
  - A smooth gamma can only stand in for the time-averaged loss (about 0.05/yr). If epizootics matter for a scenario, a stochastic pulse event is the faithful representation, and gamma should then stay small.
- **Grey seal.** Density dependence is in pup survival (0.46 to about 0.12 at K); adult survival is constant at 0.96 in the UK assessment model.
  - If the pup losses are crowding and colony effects, they belong in M1. In biomass terms that is about 0.04-0.07/yr between 0 and K, so gamma could reach about 0.4-0.7.
  - If they are post-weaning food limitation, they are already in the starvation channel.
  - SCOS does not separate the two.
- **Seals as a group** (muted, so low priority): **gamma 0.2 (range 0-0.5)**. This takes phi_nf of about 0.1 of beta_tot (0.21) at x = 0.9, with the upper end reserved for the case where density-dependent epizootic and colony losses are wanted inside M1.

---

## 5. Seabirds (shorter)

### Literature
- **Ashmole's halo and colony size.**
  - Birt et al. (1987) found fish densities "significantly lower in bays used by cormorants for feeding", supporting "Ashmole's (1963) hypothesis that seabird populations are limited by food supplies during the breeding season" [A]. [Birt, V.L., Birt, T.P., Goulet, D., Cairns, D.K. & Montevecchi, W.A. 1987. Mar. Ecol. Prog. Ser. 40: 205-208. doi:10.3354/meps040205](https://doi.org/10.3354/meps040205)
  - Jovani et al. (2016), 43 species and 28,262 colonies: "foraging range imposes a ceiling on seabird colony sizes" (slope about 2, i.e. area) [A]. [Jovani, R. et al. 2016. Oikos 125: 968-974. doi:10.1111/oik.02781](https://doi.org/10.1111/oik.02781)
  - Patterson et al. (2022) report that murre foraging trip distance scales as colony size^0.33 [S]. [Curr. Biol. 32: 3800-3807. doi:10.1016/j.cub.2022.06.084](https://doi.org/10.1016/j.cub.2022.06.084)
  - Furness & Birkhead (1984), Nature 311: 655: only the title was retrieved [N]. [doi:10.1038/311655a0](https://doi.org/10.1038/311655a0)
- **Lewis et al. (2001)**: per-capita growth of gannet colonies in Britain and Ireland "declined with increasing population size"; mean foraging trip duration "is positively correlated with colony size, both among colonies ... and within colonies as they change in size" [A, via Europe PMC]. [Lewis, S., Sherratt, T.N., Hamer, K.C. & Wanless, S. 2001. Evidence of intra-specific competition for food in a pelagic seabird. Nature 412: 816-819. doi:10.1038/35090566](https://doi.org/10.1038/35090566). Note: this is Nature, not Proc. R. Soc. B.
- **Horswill & Robinson (2015), JNCC Report 552, section 7 and Tables 34-35** [F] [JNCC Report 552](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf)
  - Number of studies by demographic rate:
    - Survival: compensation 8, absent 0.
    - Productivity: depensation 14, compensation 8, absent 12.
    - Recruitment and age at first breeding: compensation 16, absent 3.
    - Dispersal: compensation 12. Population growth: compensation 9.
  - "density-dependent regulation on survival rates can also be age-specific, such that juvenile birds are more impacted ... than birds older than age one" (Marvelde et al. 2009).
  - "Other studies indicate that density-independent factors may have a stronger impact on survival rates at higher colony densities (Frederiksen & Bregnballe 2000b ...)".
  - "The identification of compensation was most prevalent in studies that examined rates of recruitment."
  - Breeding-population losses are often buffered by recruits from a non-breeding pool until that pool is exhausted.
- **Horswill, O'Brien & Robinson (2017)**: "Compensatory regulation was reported across all of the demographic processes". The regulation of productivity was "highly context-dependent". Depensation was "consistently attributed to increased rates of predation at lower population densities" [A]. [J. Appl. Ecol. 54: 1406-1414. doi:10.1111/1365-2664.12841](https://doi.org/10.1111/1365-2664.12841)
- **Frederiksen & Bregnballe (2000)**, Danish great cormorant (Vorso, n = 11,000): adult survival ranged 0.74-0.95 (mean 0.88). European population size combined with winter temperature explained 52-64 % of the variation, with "evidence for density dependence in adult survival, at least in cold winters" [A]. This is a Kattegat-area example. [J. Anim. Ecol. 69: 737-752. doi:10.1046/j.1365-2656.2000.00435.x](https://doi.org/10.1046/j.1365-2656.2000.00435.x)
- **Fay et al. (2015)**, wandering albatross: "strong evidence for density-dependent mortality of juveniles"; population size explained 41 % of its variation [A]. [J. Anim. Ecol. doi:10.1111/1365-2656.12390](https://doi.org/10.1111/1365-2656.12390)
- **Cury et al. (2011)**: below about one-third of maximum prey biomass, seabirds "experience consistently reduced and more variable productivity" (14 species, 7 ecosystems). The response is in breeding success [A]. [Science 334: 1703-1706. doi:10.1126/science.1212928](https://doi.org/10.1126/science.1212928)
- **Miller et al. (2019)**: state-space fits to three seabird time series showed "clear environmental regulation in two of our three species" [A]. [J. Appl. Ecol. 56: 2118-2130. doi:10.1111/1365-2664.13448](https://doi.org/10.1111/1365-2664.13448)

### Calculation
No source gave a slope of adult survival against population size that I could read, so gamma cannot be computed from data.

Bracket instead, with seabird r_max of about 0.10 and M1 0.15 (`seabirds.md`), theta = 1-2.4 at x = 1:
- beta_tot = 0.10-0.24/yr.
- gamma = phi_nf * beta_tot / 0.15, so phi_nf = 0.1 gives gamma = 0.07-0.16 and phi_nf = 0.2 gives 0.13-0.32.

### Inference
- In seabirds, density dependence acts mainly in the breeding season, through food depletion around colonies.
  - Its targets are productivity, chick condition, recruitment and age at first breeding, and juvenile survival.
  - Adult survival responds weakly and mainly in interaction with winter conditions.
  - Survival compensation is reported (8 studies), but several of those cases concern juveniles. The review collated mainly positive findings, so "0 absent" is not evidence of universality.
- Whether the model captures the halo depends on whether seabirds are tied to colonies during breeding. If birds roam the whole grid freely, local depletion around colonies cannot arise, and part of the real density dependence would be missing from the food channel.
- That missing part belongs to breeding output, i.e. the growth term, not to adult M1.
- Depensation (Allee effects through predation at small colonies) is as common as compensation for productivity, so the density-independent floor should stay.
- **Seabirds: gamma 0.1 (range 0-0.3)**, by judgement within the bracket above; seabirds are muted.

---

## 6. Synthesis for the model

### 6.1 Recommended gamma (elasticity of M1 at B_ref, grid-total B)

The central values assume that the explicit food channel (starvation plus reserve-dependent growth) already delivers most of the literature total, about 0.15-0.25/yr at B_ref. They are conditional on the check in 6.4, item 1; do not lift them without it.

| Group | M1_ref (/yr) | Total beta_tot at B_ref (lit.) | Non-food share phi_nf | **gamma central** | **range** | beta_M = gamma * M1 (/yr) |
|---|---|---|---|---|---|---|
| Porpoise | 0.10 | 0.14-0.24 (r 0.10, theta 2.39, x 0.8-1); 0.02-0.10 if r = 0.04 or theta = 1 | 0.05 (0-0.15) | **0.1** | 0-0.3 | 0.010 (0-0.03) |
| Seals (harbour seal basis) | 0.10 | 0.21 (0.15-0.26) at x = 0.90, from Carroll mu = 0.28 +/- 0.10 | 0.1 (0-0.25) | **0.2** | 0-0.5 | 0.020 (0-0.05) |
| Seabirds | 0.15 | 0.10-0.24 (bracket, theta 1-2.4) | about 0.1 (0-0.2) | **0.1** | 0-0.3 | 0.015 (0-0.045) |

Per 6-h tick, divide beta_M by 1460 (e.g. porpoise 0.010/yr = 6.8e-6 per tick). gamma is dimensionless and independent of tick length.

With a floor, M1 = M_floor + M_dd (B/B_ref)^g', and g' = gamma * M1_ref / M_dd. Example: a porpoise split of M_floor 0.08 + M_dd 0.02 with gamma 0.1 gives g' = 0.5.

### 6.2 What the model already captures and what would belong in M1
- **Already explicit (food):**
  - Starvation deaths, about 39 % of porpoise losses.
  - Suppressed reproduction at a low reserve (q = s - u), which corresponds to pregnancy failure in poor condition (IJsseldijk 2021; Murphy 2015) and to the somatic-growth and fecundity signals in seals (Harding 2018; Kauhala 2019 via SCOS).
  - For seabirds, the halo is captured only if breeding birds are colony-bound.
- **Would belong in M1 (non-food):**
  - Density-dependent disease transmission. For seals this is epizootics, better modelled as pulses. For porpoises it is at most a minor share, since most disease deaths co-occur with poor condition.
  - Crowding at breeding or haul-out sites (grey seal pup survival at dense colonies, if not food-mediated).
  - Social and space effects.
  - Of these, only the seal pathways have quantitative support.
- **Belongs in the density-independent M1 floor:** grey seal predation on porpoises (Leopold 2015), neonatal and parturition losses, contaminant-linked reproductive failure (Murphy 2015), and the senescent tail.

### 6.3 The literature case for gamma = 0
- Adult survival is the last vital rate to respond and the least variable: Eberhardt 1977/2002, Gaillard 2000, Fowler 1981.
- In Fowler's (1984) review only 1 species showed density-dependent adult survival, and Taylor & DeMaster (1993) call its form "purely speculative".
- UK grey seal assessments fix adult survival as density independent (0.96) and put all density dependence in pup survival (SCOS 2022).
- Harbour seal adult survival stays above 0.95 while growth varies (Carroll 2025).
- For porpoises there is no direct demonstration of density-dependent mortality at all. The demonstrated pathways are nutritional and therefore already explicit.
- If the explicit food channel alone produces a total density-dependence strength of about 0.15-0.25/yr at B_ref, then gamma = 0 is the consistent choice.

### 6.4 Caveats and a check worth running (suggestion only; nothing was run)
1. **Measure what the food channel already delivers.**
   - Run the frozen policies with the initial porpoise biomass scaled to 0.5x, 1x and 2x.
   - Measure per-capita (growth - starvation - M1) against ln B.
   - The slope is the model's beta_tot,food. Compare it with the literature total of 0.14-0.24/yr (theta 2.39) or 0.08-0.10/yr (theta 1).
   - If the model's slope is already at that level, gamma should be 0-0.1. If it is near zero, the gap is a food-web calibration question first, and putting it all into M1 would be the "action hack" route that AGENTS.md section 7 warns against.
2. **Grid-total, not per cell.** Every mechanism cited (pregnancy versus regional prey, epizootics across colonies, colony-scale halos) acts at population or colony scale. A per-cell (B/B_ref)^gamma term would also penalise aggregation, which is a different mechanism with no support here.
3. **No porpoise bycatch in the model** (section 3). The model's x will tend to 1, where any gamma bites hardest.
4. **B_ref against survey density.** 12 t is about 0.06-0.11 of the biomass implied by SCANS densities for the area (section 3, Calculation). Worth checking before trusting "B_ref is near K".
5. **Theta is uncertain.** It may be below 1 in the Sibly et al. (2005) sense. With theta = 1 every gamma above shrinks by a factor of about 2.4.

---

## 7. Source status and fetch log

| Source | Status | Route |
|---|---|---|
| Taylor & DeMaster 1993 | [F] | NOAA SWFSC PDF, text layer via pdftotext |
| Fowler 1981; Eberhardt 1977; Eberhardt 2002; Gaillard 2000; Sibly 2005; Wade 1998; Clark 2010 | [A] | OpenAlex abstracts (api.openalex.org) |
| Gaillard 1998 TREE | [2/S] | Title and secondary only; no abstract |
| Fowler 1984 (IWC SI 6), Fowler 1987 | [2] | As cited in Taylor & DeMaster 1993 |
| Owen et al. 2024 | [F] | Frontiers HTML (first 100k of 151k characters) |
| OSPAR QSR 2023 bycatch; OSPAR QSR 2023 cetacean abundance | [F] | oap.ospar.org HTML (first 100k characters of each) |
| Gilles et al. 2023 (SCANS-IV); Hammond et al. 2021 (SCANS-III) | [F] | PDFs, pdftotext; tables read, maps not readable |
| IJsseldijk 2021; MacLeod 2007; Learmonth 2014; Murphy 2015; Read & Hohn 1995; Leopold 2015 PRSB; Dutch 2024 post-mortem report | [A] | OpenAlex; authors checked on Crossref |
| Neimanis et al. 2022 | [F] | Europe PMC full-text XML |
| Carroll et al. 2025, plus S2 and S3 Tables | [F] | Europe PMC XML; PLoS supplementary PDFs |
| Harkonen et al. 2006 DAO | [F] | int-res PDF |
| Harding 2018; Harkonen 2002; Harding, Harkonen & Caswell 2002; Swinton 1998; Lonergan 2010; Thomas 2019 | [A] | OpenAlex |
| SCOS 2022 | [F] | SMRU PDF, sections on the grey seal model and demography |
| JNCC Report 552 | [F] | Section 7 and Tables 34-35 |
| Horswill 2017; Frederiksen & Bregnballe 2000; Birt 1987; Cury 2011; Jovani 2016; Miller 2019; Fay 2015 | [A] | OpenAlex |
| Lewis et al. 2001 | [A] | Europe PMC REST |
| Patterson et al. 2022 | [S] | Search summary; abstract not shown |

**Failed or not obtained:**
- Semantic Scholar API: HTTP 429.
- A uvm.edu "Fowler.pdf" turned out to be an Ecology 1995 scan, not Fowler 1981; not used.
- No free full text found for Fowler 1981, Eberhardt 1977/2002, Gaillard 1998/2000, Horswill 2017 (the Glasgow repository has no full text), Lewis 2001 or Furness & Birkhead 1984 (title only).
- The Miller 2019 Wiley PDF returned HTML (blocked).
- Leopold et al. 2015 J. Sea Res.: no abstract.
- Not used: Bonenfant et al. 2009 (Adv. Ecol. Res.; no abstract returned) and the SCOS 2023 report (not found; SCOS 2022 used instead).
- Kindt-Larsen et al. 2023, the source of the Belt Sea bycatch figure, was not opened; that figure is [2] via Owen et al. 2024.
- Olsen et al. (named in the assignment for harbour seals): not searched for or opened. The Kattegat-Skagerrak seal data come from Carroll et al. 2025, Harkonen et al. 2006 and Harding et al. 2018.
- Kesselring et al. (Germany) was not re-read; it is cited from `mammals.md`. No porpoise paper by "Kesselring et al." testing density dependence was found.
