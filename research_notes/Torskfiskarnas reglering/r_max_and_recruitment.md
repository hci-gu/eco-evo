# Is r_max = 0.51/yr plausible for the gadoid pool, and should growth at today's state be well below it?

Scope note:
- Sources come from a literature sub-search plus my own arithmetic on ICES SAG data.
- The FishBase species pages, Froese et al. 2016, Myers et al. 1997, the Free et al. 2019 supplement, Sguotti et al. 2019
  and Froese & Luna 2004 were fetched in full text. The other papers were read as abstracts or snippets, as marked.
- This builds on `research_notes/Litteraturbaserad growth rate per FG/fish.md` (Q4, r_max). That note gives CMSY r =
  2 Fmsy: cod 0.62, haddock 0.38, saithe 0.72, and no whiting value.
- Labels:
  - **Literature**: cited, with fetch status.
  - **Computed**: my arithmetic.
  - **Inference**: my reasoning.

## Takeaway
- **As a maximum, r_max = 0.51/yr is plausible.**
  - It is the biomass-weighted mean of the FishBase CMSY priors for the five species. The priors are 0.48-0.52, each with
    a 95 % CL of about 0.33-0.77. The library value is a faithful reading of the source.
  - Stock-specific Schaefer fits give r ≈ 2 Fmsy = 0.37-0.74. Myers et al. 1997 give cod r_m 0.53-0.82.
  - My confidence that 0.51 is within the literature range of a low-density maximum is high.
- **It is a low-density maximum from the 1960s-1990s. Today's state is neither low-density nor high-productivity.**
  1. In a Schaefer model, which is the source model of the CMSY prior, net growth at today's state is r(1 - B/K) - F. If
     the stock is near its fished equilibrium, that is about 0. The model applies r without the (1 - B/K) factor
     (`density_dependence.md`).
  2. Productivity has fallen:
     - North Sea cod recruits per spawner are about 2.5x lower than before 1990.
     - Haddock and hake R/SSB fell 3-5x after 2000-2010.
     - Warming has cut the surplus-production MSY of the North Sea stocks by 35 % (Free et al. 2019; cod θ = -0.44 per °C).
     - Inference: a present-day r for the pool is probably 0.3-0.45, not 0.51. The evidence is moderate, and no paper
       re-estimates r for today's conditions.
- **Depensation is not a supported reason for slow growth today.** Myers et al. 1995: 3 of 128 stocks. Myers et al. 1997:
  "no evidence" in cod. Hilborn et al. 2014: fewer than expected by chance. The pool is not at depleted levels except
  for cod.
- **Observed net growth at today's state is about -5 % to +11 %/yr** (pooled SSB 2023-2025). For hake and saithe it is
  negative. It is nowhere near r_max - F ≈ 0.39.

## Cited Findings

### FishBase CMSY priors (Fetched full text, fishbase.se species summary pages, via sub-search)

| Species | FishBase text (verbatim) | Resilience text |
|---|---|---|
| Gadus morhua | "Prior r = 0.51, 95% CL = 0.34 - 0.76, Based on 66 full stock assessments." | "Medium, minimum population doubling time 1.4 - 4.4 years (rm=0.2-1.1 ...)" |
| Melanogrammus aeglefinus | "Prior r = 0.50, 95% CL = 0.33 - 0.75, Based on 41 full stock assessments." | rm = 0.54-1.2 |
| Merlangius merlangus | "Prior r = 0.49, 95% CL = 0.33 - 0.74, Based on 20 full stock assessments." | |
| Pollachius virens | "Prior r = 0.52, 95% CL = 0.34 - 0.78, Based on 21 full stock assessments." | rm = 0.55-0.87 |
| Merluccius merluccius | "Prior r = 0.48, 95% CL = 0.32 - 0.72, Based on 34 full stock assessments." | |
| Pollachius pollachius | "Prior r = 0.63, 95% CL = 0.42 - 0.95, Based on 3 data-limited stock assessments." | |
| Molva molva | "Prior r = 0.55, 95% CL = 0.36 - 0.82, Based on 9 full stock assessments." | |

URL pattern: https://www.fishbase.se/summary/Gadus-morhua.html, and likewise for the others.

Computed: with W1 weights (cod 0.058, haddock 0.543, whiting 0.267, saithe 0.121, hake 0.011), the mean is
0.058×0.51 + 0.543×0.50 + 0.267×0.49 + 0.121×0.52 + 0.011×0.48 = **0.501**. With equal weights it is 0.50. The library's
0.511 is within rounding of any weighting.

Note: the priors are pooled over all stocks of each species worldwide ("66 full stock assessments" for cod). They are not
fits to the 3.a/4 stocks.

### Stock-specific surplus-production fits
- **Froese R, Garilao C, Winker H, Coro G, Demirel N, Tsikliras A, Dimarchopoulou D, Scarcella G, Sampang-Reyes A 2016.**
  "Exploitation and status of European stocks". Report for Oceana.
  [oceanrep.geomar.de/34476](http://oceanrep.geomar.de/34476/), [PDF](https://www.fishbase.de/rfroese/StockStatusReport.pdf)
  (Fetched full text.) It underlies Froese et al. 2018, Marine Policy 93:159-170, doi
  [10.1016/j.marpol.2018.04.018](https://doi.org/10.1016/j.marpol.2018.04.018).
  - Table 2, life-history r priors (r and 2 SD range):
    - cod 0.47 (0.23-0.96)
    - haddock 0.48 (0.23-1.00)
    - whiting 0.51 (0.25-1.01)
    - hake 0.46 (0.22-0.95)
    - saithe 0.40 (0.21-0.75)
  - BSM/CMSY results, Fmsy and B/Bmsy:

    | Stock | Fmsy | B/Bmsy |
    |---|---|---|
    | cod-347d | 0.22 | 0.64 |
    | cod-kat | 0.26 | 0.39 |
    | had-346a | 0.22 | 0.21 |
    | sai-3a46 | 0.26 | 0.78 |
    | whg-47d | 0.37 | 0.35 |
    | whg-kask | 0.23 | 0.14 |
    | hke-nrtn | 0.19 | 1.07 |

  - The model includes "a simple stock-recruitment model ... to account for reduced productivity at severely depleted
    stock sizes".
  - Computed (Schaefer, r = 2 Fmsy): cod-347d 0.44, haddock 0.44, saithe 0.52, whiting-47d 0.74, whiting-kask 0.46,
    hake 0.37. W1-weighted: about 0.53.
  - These fits end around 2014-2015, before the haddock pulse. The haddock B/Bmsy of 0.21 then contrasts with SSB more
    than 4x higher today.
- **Sparholt H et al. 2021.** "Estimating Fmsy from an ensemble of data sources to account for density dependence in
  Northeast Atlantic fish stocks". ICES J. Mar. Sci. 78:55-69, doi
  [10.1093/icesjms/fsaa175](https://doi.org/10.1093/icesjms/fsaa175).
  - CMSY Fmsy: cod.27.47d20 0.31, had.27.46a20 0.19, pok.27.3a46 0.36, so r = 0.38-0.72.
  - (Fetched; Table 1 partly garbled. Values as cross-read in `fish.md`.)
- **Myers RA, Mertz G, Fowlow PS 1997.** "Maximum population growth rates and recovery times for Atlantic cod, *Gadus
  morhua*". Fishery Bulletin 95:762-772.
  [PDF](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1997/954/myers.pdf) (Fetched full text.)
  - Table 1, r_m / α̃ / age at maturity / bottom temperature:
    - Kattegat 0.53 / 3.8 / 3 / 6.5 °C
    - North Sea 0.56 / 9.0 / 4 / 8.6 °C
    - Skagerrak (north IIIa) 0.82 / 11.2 / 3 / 6.5 °C
  - "At colder temperatures, r m is around 18% a year for all populations".
  - r_m is in numbers and was fitted to stock-recruit data up to the mid-1990s.
- **Myers RA, Bowen KG, Barrowman NJ 1999.** "Maximum reproductive rate of fish at low population sizes". CJFAS
  56:2404-2419, doi [10.1139/f99-201](https://doi.org/10.1139/f99-201). (Abstract only.)
  - Annual spawners per spawner at low density, from Froese & Luna 2004 ("No relationship between fecundity and annual reproductive rate in bony fish", Acta Ichthyol. Piscat. 34:11-20, doi
    [10.3750/aip2004.34.1.02](https://doi.org/10.3750/aip2004.34.1.02); fetched full text): cod 3.94, haddock 2.05,
    whiting 3.13, saithe 3.19.
- **Hutchings JA 2005.** "Life history consequences of overexploitation to population recovery in Northwest Atlantic
  cod". CJFAS 62:824-832, doi [10.1139/f05-081](https://doi.org/10.1139/f05-081). (Abstract only.)
  - Earlier maturation "can reduce annual population growth in Atlantic cod by 25%-30%".

### Recruitment decline and warming
- **Free CM, Thorson JT, Pinsky ML, Oken KL, Wiedenmann J, Jensen OP 2019.** "Impacts of historical warming on marine
  fisheries production". Science 363:979-983, doi [10.1126/science.aau1758](https://doi.org/10.1126/science.aau1758).
  (Fetched full supplement.)
  - Method: a Pella-Tomlinson surplus-production term multiplied by exp(θ·SST anomaly).
  - Table S4: North Sea LME, 9 stocks, mean θ -0.36, MSY change 1930-2010 "-34.6" %. Celtic-Biscay -15.2 %.
  - Table S2, stocks with significant θ: CODNS -0.44, HADNS-IIIa -0.36, WHITNS-VIId -0.35, POLLNS-VI-IIIa -0.36
    (saithe).
- **Beaugrand G, Brander KM, Lindley JA, Souissi S, Reid PC 2003.** "Plankton effect on cod recruitment in the North
  Sea". Nature 426:661-664, doi [10.1038/nature02164](https://doi.org/10.1038/nature02164). (Abstract.)
  - "rising temperature since the mid-1980s has modified the plankton ecosystem in a way that reduces the survival of
    young cod."
- **O'Brien CM, Fox CJ, Planque B, Casey J 2000.** "Climate variability and North Sea cod". Nature 404:142, doi
  [10.1038/35004654](https://doi.org/10.1038/35004654). (Snippet only.)
  - Higher temperatures "might prevent the stock from being able to produce recruitments as high as those that occurred
    during the 1960s and 1970s, even if the spawning-stock biomass were to rebuild".
- **Britten GL, Dowd M, Worm B 2016.** "Changing recruitment capacity in global fish stocks". PNAS 113:134-139, doi
  [10.1073/pnas.1504709112](https://doi.org/10.1073/pnas.1504709112). (Abstract.)
  - "average recruitment capacity has declined at a rate approximately equal to 3% of the historical maximum per decade".
  - "highly negative trends in the North Atlantic".
- **Sguotti, Otto, Frelat, Langbehn, Ryberg, Lindegren, et al. 2019** (family names checked against Crossref).
  "Catastrophic dynamics limit Atlantic cod recovery". Proc. R. Soc. B 286:20182877, doi
  [10.1098/rspb.2018.2877](https://doi.org/10.1098/rspb.2018.2877). (Fetched full text, PMC6458326.)
  - North Sea cod: exploitation "has been drastically reduced; however, SB levels remained significantly lower compared
    to the beginning of the study period, when fishing mortality was similar. This hysteresis ... is related to an
    increase in SST".
  - "only two stocks fully recovered" of 19.
- **Computed by the sub-search from ICES SAG** (median recruits per unit SSB):
  - North Sea cod, 2022 single-stock assessment (key 17652): 6.87 (1963-89), 6.15 (1990s), 4.92 (2000s), 2.58 (2010-19),
    2.68 (2015-22). That is about 2.5x lower than before 1990.
  - Haddock: 30-50 before 2000, 6-11 in 2000-2024.
  - Hake: from 11-17 to 3.4-3.8 after 2010.
  - Saithe: from about 1.0 to 0.63.
  - Whiting: no decline.
  - SSB is lower now than in the 1980s for cod, which under compensation should *raise* R/SSB. The productivity loss is
    therefore at least this large.

### Depensation
- **Myers RA, Barrowman NJ, Hutchings JA, Rosenberg AA 1995.** "Population dynamics of exploited fish stocks at low
  population levels". Science 269:1106-1108, doi [10.1126/science.269.5227.1106](https://doi.org/10.1126/science.269.5227.1106).
  (Abstract.) Of 128 stocks, "only 3 stocks with significant depensation".
- **Keith DM, Hutchings JA 2012.** "Population dynamics of marine fishes at low abundance". CJFAS 69:1150-1163, doi
  [10.1139/f2012-055](https://doi.org/10.1139/f2012-055). (Abstract.) "some exhibit evidence of an Allee effect, such as
  Atlantic cod".
- **Hilborn R, Hively DJ, Jensen OP, Branch TA 2014.** "The dynamics of fish populations at low abundance and prospects
  for rebuilding and recovery". ICES J. Mar. Sci. 71:2141-2151, doi [10.1093/icesjms/fsu035](https://doi.org/10.1093/icesjms/fsu035).
  (Abstract.) Depensatory models improved fits in fewer stocks "than expected by chance", but the authors "could not rule
  out depensation at very low stock sizes". "considerable evidence that the majority of fish stocks are impacted by
  changes in productivity regimes".
- **Perälä T, Kuparinen A 2017.** Proc. R. Soc. B 284:20171284, doi [10.1098/rspb.2017.1284](https://doi.org/10.1098/rspb.2017.1284).
  (Abstract.) This is on herring, not gadoids. It argues that Allee effects look rare because data at low abundance are scarce.
- **Kuparinen A, Hutchings JA 2014,** R. Soc. Open Sci. 1:140075. Adult M of NW Atlantic cod rose at low abundance,
  attributed to seals (via `regulation_and_models.md`).

### Recovery rates observed
- **Hutchings JA 2000.** "Collapse and recovery of marine fishes". Nature 406:882-885, doi
  [10.1038/35022565](https://doi.org/10.1038/35022565). (Abstract.) Many gadids "have experienced little, if any,
  recovery as much as 15 years after 45-99% reductions in reproductive biomass".
- **Neubauer P, Jensen OP, Hutchings JA, Baum JK 2013.** "Resilience and recovery of overexploited marine populations".
  Science 340:347-349, doi [10.1126/science.1230441](https://doi.org/10.1126/science.1230441). (Abstract.) Moderate
  overfishing allows swift recovery, while prolonged intense overexploitation "delays rebuilding".
- **Computed (sub-search and `stock_trends_and_mortality.md`).**
  - Hake: total-biomass production rate (B_t+1 - B_t + C_t)/B_t was 0.12-0.20 in 2015-2024 at F about 0.2. TSB fell from
    514 to 194 kt.
  - North Sea cod (old series): rose about 10 %/yr from 2006 to 2015-17 at F 0.4-0.7, then fell again.
  - Haddock: ln growth of SSB about 0.5/yr in 2020-23. This was the 2019-2020 year classes, a pulse, not r.
  - Pooled SSB (5 stocks): +20 %/yr in 2019-2023, about +3 %/yr on average in 2023-2025.
  - My own SAG-2026 version of the pooled trend is -3.4 %/yr over 2023-2026 (`stock_trends_and_mortality.md`). The
    sub-search used a 2025 haddock value that differs from the 2026 table (885 against 751 kt), which explains the
    difference in sign.

## Inferences

1. **Which r belongs in the budget?**
   - The derived g (`lib/world/growth_budget.py`) pays r_max + M1 + M2 at the ideal-conditions reserve s*, so at ideal
     food the pool grows at r_max - F *at any density*.
   - That is correct only for a stock far below K. The CMSY prior is defined with K.
   - Keeping r_max at 0.51 is defensible only together with a density-dependent closure term
     (`density_dependence.md`, option C1).
   - Lowering r_max alone to make today stationary would need r_max ≈ F ≈ 0.12-0.2. That is below every literature r
     for these species (lowest about 0.37) and would remove the stock's ability to recover from low density.
2. **A "present-productivity" r of 0.35-0.45 is defensible if the user wants to reflect the recruitment decline.**
   - Reasoning:
     - Cod R/SSB has fallen 2.5x. By Euler-Lotka, r ≈ ln(α̃)/T, so ln(9.0/2.5)/T with T ≈ ln(9)/0.56 = 3.9 yr gives
       r ≈ 0.33 for cod.
     - Haddock R/SSB fell 3-5x, but its recent strong year classes show high productivity is still possible.
     - Free et al. imply 25-35 % lower production per unit biomass for the North Sea gadoids.
   - Taking 0.51 × (0.65-0.85) gives **0.33-0.43**. This is my estimate, not a published value. Confidence low-moderate.
3. **Depensation should not be invoked for the pool.** It is not supported at today's pool level, and only cod is
   depleted. For cod the depensation-like mechanism discussed in the literature is seal predation (Kuparinen & Hutchings
   2014), and the model has seals explicitly.
4. **Bottom line for Q3.**
   - The growth at today's state should be well below r_max, at about 0 ± 0.1/yr.
   - The reason is mainly that today's stock is near its fished equilibrium (the K term), and secondarily lower
     productivity than in the period r was fitted on (recruitment and warming). Depensation does not play a part.
   - A lower r_max (0.35-0.45) helps somewhat: it removes 0.06-0.16 of the 0.39 surplus. It is not sufficient alone.
