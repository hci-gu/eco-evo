# How strong should density-dependent M1 be? Cross-taxon theory, surplus-production shape, and where compensation acts

Scope: this note covers cross-taxon theory and syntheses. Sibling notes in the same folder cover each group's species literature. The question is whether the Mareld residual mortality M1 (`natural_mortality`, now a constant per-tick rate) should depend on density, and how strongly, measured as

    beta_M = dM1/d ln B   at B_ref (per year),   gamma = beta_M / M1(B_ref)   (dimensionless)

Work done: about 45 tool calls, with PDFs converted locally with pdftotext and one figure page rendered and read by eye. Each citation says how the source was read:
- **fulltext**: the PDF or HTML text was read by me.
- **figure read**: values read off a rendered figure; approximate.
- **summariser**: a page-summarising fetch; weaker, because a second model filtered the text.
- **abstract** or **snippet**: abstract only, or a search-result snippet only.
- **secondary**: cited by another source; the original was not reached.

Sections are split into **Literature** (what sources say), **Calculation** (my arithmetic, shown in full) and **Inference** (my reading).

**Corrections to the brief:**
- Knape & de Valpine is *Ecology Letters* 14:17-23 (**2011**), not 2012. Two bibliographic pages agree; the abstract was not reached.
- Lorenzen & Camp (2019) is in *Fisheries Research* 217:5-10, not ICES JMS. It was found only through a secondary citation.
- In Thorson, Cope, Branch & Jensen (2012) the Pella-Tomlinson shape is phi (Fletcher form). The brief's exponent is m = phi - 1.

**Not verified:**
- Clark et al. 2010, Polansky et al. 2009, Ross 2006/2009 and Knape & de Valpine 2011 were read as abstracts or snippets only (fetches got 403, an HTML page instead of the PDF, or bot protection).
- Anderson 2001 (Ecology 82:245) and Gascoigne & Lipcius 2004: bibliographic records only, no abstract.
- Lorenzen & Camp 2019: secondary citation only.
- Lorenzen 1996 (M proportional to W^-0.29): secondary, via Horbowy et al. 2025.
- Swain & Benoit 2015: int-res returned 429; known through Neuenhoff et al. 2019 (fulltext) and a search snippet.
- Pella-Tomlinson algebra: checked against Thorson et al. 2012 eqs. 1-2 (Fletcher 1978 / Quinn & Deriso 1999 form) and by finite differences. No textbook page (Quinn & Deriso, Hilborn & Walters) was read directly.

---

## Q1. Shape and strength of density dependence across taxa (theta-logistic, GPDD syntheses)

### Takeaway
- **Shape.** Across about 1,800 GPDD time series, the fitted theta-logistic shape is usually concave (theta < 1): 78 % of cases on average, with fish possibly highest (Sibly et al. 2005). Fits are statistically fragile:
  - theta and r_m trade off along a likelihood ridge, so their product is the identified quantity (Clark et al. 2010; Polansky et al. 2009);
  - negative theta is not meaningful (Ross 2009);
  - observation error pushes theta toward 0 and the pgr-on-ln N slope toward -1 (Sibly et al. 2005 Fig. 4; Barker & Sibly 2008 snippet; Knape & de Valpine 2011 secondary).
- **Evidence for density dependence.** Support is pervasive (median multi-model weight 0.75; Brook & Bradshaw 2006) but weakest in fish (0.60).
- **Strength.** The Gompertz annual slope, d ln(N_t+1/N_t)/d ln N_t, is literally the brief's beta_total. Medians: fish -0.7, birds -0.6, mammals -0.4 (Herrando-Perez et al. 2012), and stronger in small, short-lived species.
- **These magnitudes are not usable as model parameters.** Pure measurement error alone gives a slope of -1, and for slow mammals the medians are arithmetically impossible (see the inference). The robust cross-taxon lesson is qualitative: density feedback is common and usually concave, and the slope at equilibrium (r x theta) is what time series constrain.

### Literature
- Sibly, Barker, Denham, Hone & Pagel (2005) fitted pgr = r_m[1-(N/K)^theta] to GPDD series. The abstract says: "We estimated this relationship for 1780 time series of mammals, birds, fish, and insects ... rates of population growth are high at low population densities but ... decline rapidly with increasing population size and then flatten out, for all four taxa."
  - Sample: "birds (150 species), mammals (79), bony fish (64), and insects (381)".
  - "The average proportion of cases where theta < 1 is 0.78 ... there is a suggestion that the proportion is higher in fish than in mammals, birds, and insects (... chi2 = 6.9, nonsignificant)".
  - "theta was significantly (P < 0.05) different from 1 in 613 of the 1780 time series analyzed, being less than 1 in 581 cases and greater than 1 in 32."
  - The authors expected pgr-density relationships to be "convex for large mammals and similar species but concave for species with life histories like those of insects and some fish". Their data contradict this: in mammals theta falls with body size (r = -0.32, P < 0.05).
  
  (fulltext, course mirror PDF; the uidaho copy returned 403/301) - [Sibly et al. 2005, Science 309:607-610, doi:10.1126/science.1110760](https://doi.org/10.1126/science.1110760); [PDF used](https://kevintshoemaker.github.io/EECB-703/Sibly%20et%20al.%20-%202005%20-%20On%20the%20Regulation%20of%20Populations%20of%20Mammals,%20Birds.pdf)
- Sibly et al. (2005), Fig. 4: "Measurement error alone predicts a slope of -1 in regressions of pgr against loge density." Mean slopes read from the figure: mammals about -0.59, birds about -0.81, fish about -0.87, insects about -1.03, simulated pure-error series about -1.03. The text says "our insect populations also fall close to -1.0, possibly suggesting that measurement error has affected the estimates of theta". Fig. 3 histograms (theta by species, hybrid axis): in fish, most species lie between -10 and 1, and only a handful are above 1. (figure read, page 609 rendered) - same source.
- Brook & Bradshaw (2006): "median MMI support for density dependence = 74.7-92.2%". Table 2, the summed AICc weight of density-dependent models (RDD wt), by taxon:

  | Taxon | n | Median years | RDD weight | Gompertz weight |
  |---|---|---|---|---|
  | Insects | 604 | 16 | 0.795 | 0.443 |
  | Birds | 224 | 20 | 0.797 | 0.455 |
  | Mammals | 152 | 22 | 0.644 | 0.375 |
  | Fish | 115 | 15 | 0.601 | 0.351 |

  Support "increased consistently with the number of generations monitored". (fulltext) - [Brook & Bradshaw 2006, Ecology 87:1445-1451](https://conservationbytes.com/wp-content/uploads/2014/12/brook-bradshaw-2006-ecology.pdf)
- Herrando-Perez, Delean, Brook & Bradshaw (2012) use the same 1,198 species. The response variable is the Gompertz slope of r = ln(N_t+1/N_t) on ln N_t, with an annual time step.
  - "Median strength of density feedback was -0.7 (-1.4 to -0.2)". By group: "plants were highest at -0.9 (-1.3 to -0.6), followed by herpetiles at -0.9 (-1.4 to -0.5), aquatic invertebrates at -0.8 (-1.5 to -0.2), insects at -0.8 (-1.4 to -0.3), fish at -0.7 (-1.3 to -0.2), birds at -0.6 (-1.3 to -0.2), and mammals at -0.4 (-1.4 to -0.1)."
  - "Overcompensatory density feedback (strength < -1) occurred in ~20% of the censuses", including fish (11 species, 21 %) and mammals (16, 18 %).
  - "density feedback was consistently stronger for smaller-bodied and shorter-lived species". Life-history traits explain about 10-30 % of the variance.
  - Measurement-error simulations used only 5-15 % error.
  
  (fulltext, PMC HTML) - [Herrando-Perez et al. 2012, Ecol. Evol. 2:1922-1934, doi:10.1002/ece3.298](https://pmc.ncbi.nlm.nih.gov/articles/PMC3433995/)
- Clark, Brook, Delean, Akcakaya & Bradshaw (2010): theta and r_m "are able to play-off against each other (providing a constant product), thus allowing both to adopt extreme and ecologically implausible values". Estimates of extinction risk that depend on this model are "prone to imprecision". (snippet; Wiley 403; Adelaide repository page returned only "DSpace") - [Clark et al. 2010, Methods Ecol. Evol. 1:253-262, doi:10.1111/j.2041-210X.2010.00029.x](https://doi.org/10.1111/j.2041-210X.2010.00029.x)
- Polansky, de Valpine, Lloyd-Smith & Getz (2009): using the theta-Ricker model, multimodal and ridged likelihood surfaces arise "even without model misspecification or observation error". In the sparrowhawk case the best-fit models are implausible outside the data, and similar results hold for 25 more datasets. (summariser of the repository abstract; the UCLA PDF link returned HTML) - [Polansky et al. 2009, Ecology 90:2313-2320, hdl:2263/13888](http://hdl.handle.net/2263/13888)
- Ross (2009): negative theta lets estimated growth be negative for extant species. Theta "should almost always be greater than -1". Ross (2006) is a Science comment that Sibly's analysis should be "used cautiously". (snippet) - [Ross 2009, Ecol. Model. 220:3472-3474](https://ideas.repec.org/a/eee/ecomod/v220y2009i23p3472-3474.html); [Ross 2006, Science 311:1100](https://digital.library.adelaide.edu.au/items/d7190d5f-dc8c-4d42-a430-beb491d4c657)
- Barker & Sibly (2008): "Large measurement errors bias estimates of theta towards zero." (snippet) - [Ecol. Model. 2008](https://www.sciencedirect.com/science/article/abs/pii/S0304380008004109)
- Knape & de Valpine (2011), as cited by Johnson (2024): observation error "could give the illusion of strong density dependence". (secondary; the abstract was not reached, and de Valpine's page returned 404) - [Johnson 2024, arXiv:2412.17313](https://arxiv.org/abs/2412.17313); bibliographic: [de Valpine GPDD tag page](https://nature.berkeley.edu/~pdevalpine/tag/gpdd)

### Calculation
1. **The product r_m x theta is the brief's beta at K.** Differentiate pgr = r_m[1-(N/K)^theta]: d pgr/d ln N = -r_m theta (N/K)^theta, which equals -r_m theta at N = K. The parameter combination Clark et al. call identified is therefore exactly the slope at K.
2. **Discrete to continuous Gompertz.** Continuous Gompertz, d ln N/dt = b(ln K - ln N), gives ln N_t+1 - ln K = e^-b (ln N_t - ln K). The annual-step slope is therefore beta_G = e^-b - 1, so b = -ln(1 + beta_G):

   | Median beta_G | Group | b (/yr) |
   |---|---|---|
   | -0.4 | mammals | 0.51 |
   | -0.6 | birds | 0.92 |
   | -0.7 | fish | 1.20 |
   | -0.8 | insects | 1.61 |

   Values with beta_G < -1 have no continuous-time analogue.
3. **Sanity bound.** The slope b can be sustained over an observed range N_min..K only if b ln(K/N_min) <= r_max, because pgr cannot exceed r_max at N_min. For harbour porpoise (r_max 0.10 /yr) over a 3-fold range (ln 3 = 1.10), b <= 0.10/1.10 = 0.091 /yr. That is 5.6 times below the mammal median of 0.51 /yr. For gadoids (r_max 0.51) over a 10-fold range (ln 10 = 2.30), b <= 0.22 /yr. That is 5.4 times below the fish median of 1.20 /yr.

### Inference
- The GPDD medians are dominated by observation error and by fast-lived taxa; GPDD "mammals" include many rodents. They should not be read as beta_total for Mareld's slow groups. If anything, Sibly's Fig. 4 shows the fish slope (-0.87) sits close to the pure-error value (-1).
- What does transfer:
  - (i) The shape is more often concave than convex, so most of the compensation acts at densities well below K.
  - (ii) Because the slope at equilibrium is the identified quantity, any DD-M1 calibration should be stated as a slope at B_ref (as the brief does), not as a theta.
  - (iii) Strength scales with the pace of life.

---

## Q2. Fish surplus-production shape (Bmsy/B0, Pella-Tomlinson exponent) and steepness for Gadidae and Clupeidae

### Takeaway
- **Production shape.** Thorson et al. (2012), 147 stocks: SB_MSY/SB_0 = 0.40 pooled, 0.44 for Gadiformes and 0.26 for Clupeiformes (the lowest).
  - Mapped to the brief's exponent m, Gadiformes give m = 0.46 (from the mean ratio) to 0.73 (from the mean phi). Clupeiformes give m = -0.46 to -0.40, i.e. "flatter than Fox".
  - The two routes differ because the ratio is a concave function of phi averaged over large between-stock variability (Jensen's inequality).
- **Steepness.**
  - Myers, Bowen & Barrowman (1999), median: Gadidae h = 0.79 (cod 0.84, haddock 0.74, saithe 0.81), Clupeidae 0.71 (Atlantic herring 0.74, sprat 0.65).
  - Thorson (2020) FishLife reverses the order at order level: Gadiformes h = 0.63, Clupeiformes 0.74.
- Both sets put a large compensatory reserve in recruitment: lifetime maximum reproductive rate alpha_hat = 4h/(1-h) of 7-15.

### Literature
- Thorson, Cope, Branch & Jensen (2012):
  - The abstract says: "The estimated SB_MSY/SB_0 = 0.40 for all 147 stocks is intermediate between the values assumed by the Fox and the Schaefer models, although Clupeiformes and Perciformes have lower and Gadiformes and Scorpaeniformes have higher" values.
  - Table 2, mean (between-stock SD): All 0.404 (0.135); Gadiformes 0.439 (0.122); Clupeiformes 0.261 (0.097). The text adds Pleuronectiformes 0.40.
  - Table 3, shape phi (SD): pooled 1.478 (0.844); Gadiformes 1.729 (0.913); Clupeiformes 0.599 (0.316).
  - Model (Fletcher form, eqs. 1-2): S_t = gamma y SB_t - gamma y SB_0 (SB_t/SB_0)^phi, with gamma = phi^(phi/(phi-1))/(phi-1). This "generalize[s] to the Fox model when phi -> 1 and to the Schaefer model when phi = 2".
  - Model selection favours a larger SB_MSY/SB_0 for larger-bodied fish within an order.
  
  (fulltext, Jensen lab PDF; cdnsciencepub 403) - [Thorson et al. 2012, CJFAS 69:1556-1568, doi:10.1139/f2012-077](https://jensen.limnology.wisc.edu/wp-content/uploads/sites/147/2021/11/Thorson_etal_2012b_CJFAS.pdf)
- An IATTC review cites the same work as "BMSY ranged from 26-46%B0 for a range of 147 stocks". (fulltext of the review) - [IATTC SAC-08-05e(ii)](https://www.iattc.org/GetAttachment/d949eefc-7b46-4b28-a562-d986286ac48f/SAC-08-05e(ii)_Review-of-limit-reference-points.pdf)
- Sparholt et al. (2021), as seen in a search snippet, quote Thorson's Pella-Tomlinson shapes as u = 2.027 (Gadiformes) and 1.427 (Clupeiformes). These do not match Thorson's Table 3 (1.729, 0.599); I did not resolve the discrepancy and follow the primary table. (snippet) - [Sparholt et al. 2021, ICES JMS 78:55, doi:10.1093/icesjms/fsaa175](https://doi.org/10.1093/icesjms/fsaa175)
- Myers, Bowen & Barrowman (1999), Table 1. Columns: n stocks, log alpha~ (maximum annual reproductive rate), SE, sigma^2, alpha_hat (lifetime), z20, z_med, z80. Steepness is z = alpha_hat/(4 + alpha_hat).

  | Family / species | n | log alpha~ | alpha_hat | z_med (z20-z80) |
  |---|---|---|---|---|
  | **Clupeidae** | 34 | 1.06 | 17.1 | 0.71 (0.49-0.86) |
  | Atlantic herring | 18 | 0.73 | 22.1 | 0.74 (0.52-0.88) |
  | Sprat | 3 | 0.87 | 10.7 | 0.65 (0.48-0.79) |
  | **Gadidae** | 49 | 1.01 | 19.6 | 0.79 (0.67-0.87) |
  | Atlantic cod | 21 | 1.37 | 26 | 0.84 (0.76-0.90) |
  | Haddock | 9 | 0.72 | 13 | 0.74 (0.64-0.82) |
  | Saithe | 5 | 1.16 | 18 | 0.81 (0.78-0.84) |
  | Silver hake | 3 | -0.18 | 2.7 | 0.39 |

  The alpha_hat column is a mean; z_med is computed from the median lifetime rate, which is lower by a factor exp(sigma^2/2) for a lognormal. Checks:
  - herring: z_med 0.74 gives median alpha_hat = 4(0.74)/0.26 = 11.4; 22.1/11.4 = 1.94 = exp(sigma^2/2), so sigma^2 = 1.33, against 1.31 in the table;
  - cod: z_med 0.84 gives median alpha_hat = 21.0; 26/21.0 = 1.24, so sigma^2 = 0.43, against 0.37.

  This is why 4h/(1-h) in the Calculation below (15.1 for Gadidae) is below the tabled 19.6. A second check on the column reading: e^1.37 = 3.9 for cod, matching the text: "The median number of replacement spawners per spawner per year for cod at low abundance is between 3 and 4". The text also says "herring has a smaller maximum reproductive rate than many species", owing to "a few stocks in the northern North Atlantic that have been reduced to very low levels". (fulltext, SEDAR copy) - [Myers et al. 1999, CJFAS 56:2404-2419, doi:10.1139/f99-201](https://sedarweb.org/documents/s8rd21-maximum-reproductive-rate-of-fish-at-low-population-sizes/)
- Thorson (2020), Table 4, mode of the predictive distribution, h (SD) and r (SD): Actinopterygii 0.74 (0.23), r 0.49 (0.83); Clupeiformes 0.74 (0.23), r 0.61 (0.96); Gadiformes 0.63 (0.23), r 0.21 (0.46). Abstract: "little correlation between steepness and either natural mortality or asymptotic maximum size"; Sebastidae "lower steepness (mean = 0.72)". The table's "Thorson et al. (2019)" column (the brief's "steepness priors") is filled only for genus Sebastes (0.58-0.78, West Coast rockfish meta-analyses). It contributes nothing for Gadidae or Clupeidae. No family-level Gadidae or Clupeidae values are printed; they live in R package FishLife, which I did not run (R is not installed). (fulltext, NOAA repository PDF) - [Thorson 2020, Fish Fish. 21:237-251, doi:10.1111/faf.12427](https://doi.org/10.1111/faf.12427)

### Calculation
1. **Mapping phi to m.** The brief's form g(B) = (r/m)[1-(B/K)^m] gives production P = Bg = (r/m)(B - B^(m+1)/K^m). That is Fletcher's form with phi = m + 1. Setting dP/dB = 0 gives (B/K)^m = 1/(m+1), so Bmsy/K = (1/(m+1))^(1/m) = phi^(1/(1-phi)). Checks: phi = 2 gives 0.5 (Schaefer); phi -> 1 gives e^-1 = 0.368 (Fox).
2. **Solving for m numerically** (Brent root-finding, script `beta.py` in the scratchpad):

   | Order | Ratio (Table 2) | m from ratio | phi (Table 3) | m = phi - 1 | Ratio implied by mean phi |
   |---|---|---|---|---|---|
   | All | 0.404 | +0.214 | 1.478 | +0.478 | 0.442 |
   | Gadiformes | 0.439 | +0.458 | 1.729 | +0.729 | 0.472 |
   | Clupeiformes | 0.261 | -0.463 | 0.599 | -0.401 | 0.279 |

   The implied ratios exceed the reported mean ratios because phi^(1/(1-phi)) is concave in phi and the between-stock SD of phi is large (0.91 for Gadiformes). I carry both m values as a bracket.
3. **Steepness to lifetime reproductive rate.** Using alpha_hat = 4h/(1-h): h = 0.63 gives 6.8; 0.71 gives 9.8; 0.74 gives 11.4; 0.79 gives 15.1.
4. **Beverton-Holt elasticity.** In the steepness form R/R0 = 4h x/((1-h) + (5h-1)x), with x = S/S0, the elasticity of recruits per spawner is d ln(R/S)/d ln S = -(5h-1)x/((1-h) + (5h-1)x). For h = 0.79 this is -0.85 at x = 0.4 and -0.93 at x = 1.

### Inference
- Per-capita recruitment in gadoids and clupeids is close to fully compensatory near B_ref: an elasticity of about -0.85 to -0.93 when h is about 0.8.
- This is the dominant compensation channel in real stocks (Q4). An age-less biomass pool has no explicit place for it.
- The steepness reversal between Myers 1999 and Thorson 2020 means family-level differences in recruitment compensation are not robust. The production-shape difference (clupeids flatter-topped than gadoids) is better supported, because Thorson 2012 estimated it directly from 147 stock histories.

---

## Q3. How much TOTAL density-dependent per-capita loss slope must a fished stock have?

### Takeaway
- **Algebra.** For g(B) = (r/m)[1-(B/K)^m] the slope is -dg/d ln B = r (B/K)^m. This equals r - mF only when the stock sits at its fished equilibrium, g(B_ref) = F.
- **Primary values** (anchor C, explained below):
  - gadoids: **beta_total about 0.32-0.39 /yr at equilibrium**, falling to **0.18-0.27 /yr** if today's pool is at B/K 0.3-0.5;
  - pelagic fish: **0.22-0.23 /yr at equilibrium** (0.24-0.32 at B/K 0.3-0.5), or 0.55 if Schaefer.
- **Upper bound for M1.** As fractions of M1 these are gamma_total of about 0.4-0.95. That is the most M1 could carry if NONE of the compensation came from the model's explicit food limitation and interference, and if adult M were the right place for it.
- **The anchor matters as much as the shape.** The choice of which quantity to hold fixed when m changes moves beta by a factor of 2-3 at small m.

### Derivation (Calculation)
1. Define x = B/K. Then dg/dB = (r/m)(-m x^(m-1)/K) = -(r/K) x^(m-1), so **dg/d ln B = B dg/dB = -r x^m**. This holds at any B.
2. Fished equilibrium: g = F gives (r/m)(1 - x^m) = F, so x^m = 1 - mF/r and **-dg/d ln B = r(1 - mF/r) = r - mF**. This is the brief's formula, confirmed. The equilibrium is x_eq = (1 - mF/r)^(1/m).
3. MSY: Bmsy/K = (1/(m+1))^(1/m) (Q2), and **Fmsy = g(Bmsy) = (r/m)(1 - 1/(m+1)) = r/(m+1)**.
4. Low-density growth: g(0) = r/m for m > 0, and it is unbounded for m <= 0. **In this parameterisation r is the slope at K, not the low-density rate**, except under Schaefer (m = 1), where both equal r. This is the same quantity Clark et al. (2010) say time series identify (Q1).
5. Numerical check: r = 0.5, m = 0.7, F = 0.12 gives x_eq = 0.703, g(B_eq) = 0.12000, and a finite-difference slope of 0.41600 = r - mF. Verified.
6. The Pella-Tomlinson algebra matches Thorson et al. (2012) eqs. 1-2, the Fletcher form after Quinn & Deriso 1999 (Q2, fulltext). No textbook page was read.

### Three anchorings of the library r_max
The library r_max (gadoids 0.511, pelagic 0.648 /yr) are "biomass-weighted FishBase CMSY priors" (fg_library.yaml:181-183, 225-228). A sibling note states r = 2 Fmsy *because* CMSY is Schaefer ([Litteraturbaserad growth rate per FG/fish.md:138](../Litteraturbaserad%20growth%20rate%20per%20FG/fish.md)). When m departs from 1, one of three quantities must be held fixed:
- **A**: r_max is the true low-density growth, so g(0) = r/m = r_max and r = m r_max. This is undefined for m <= 0.
- **B**: r_max is the slope at K, so r = r_max.
- **C (primary)**: the Schaefer Fmsy = r_max/2 is what CMSY constrains, so r = (m+1) r_max/2 and beta = (m+1) Fmsy - mF.

Results at fished equilibrium (beta in /yr, x_eq = B_eq/K; the last column is anchor C divided by the library M1):

| Group (r_max, F, M1) | Shape | m | A: beta (x_eq) | B: beta (x_eq) | **C: beta (x_eq)** | C / M1 |
|---|---|---|---|---|---|---|
| Gadoids (0.511, 0.12, 0.415) | Fox | 0 | n/a | 0.511 (0.79) | **0.256 (0.63)** | 0.62 |
| | Thorson phi - 1 | 0.729 | 0.285 (0.69) | 0.424 (0.77) | **0.354 (0.74)** | 0.85 |
| | m from ratio 0.439 | 0.458 | 0.179 (0.56) | 0.456 (0.78) | **0.318 (0.71)** | 0.77 |
| | Schaefer | 1 | 0.391 (0.77) | 0.391 (0.77) | **0.391 (0.77)** | 0.94 |
| Pelagic (0.648, 0.10, 0.580) | Thorson phi - 1 | -0.401 | n/a | 0.688 (0.86) | **0.234 (0.63)** | 0.40 |
| | m from ratio 0.261 | -0.463 | n/a | 0.694 (0.86) | **0.220 (0.60)** | 0.38 |
| | Fox | 0 | n/a | 0.648 (0.86) | **0.324 (0.73)** | 0.56 |
| | Schaefer | 1 | 0.548 (0.85) | 0.548 (0.85) | **0.548 (0.85)** | 0.94 |

Per 6-h tick (1460 ticks/yr), anchor C gives gadoids 1.75e-4 to 2.68e-4 and pelagic 1.51e-4 to 3.75e-4 per tick per unit ln B.

**Off equilibrium.** Under anchor C the slope r x^m at other B/K is:

| Case | x = 0.2 | x = 0.3 | x = 0.5 | x = 0.7 | x = 1.0 |
|---|---|---|---|---|---|
| Gadoids, m 0.729 | 0.137 | 0.184 | 0.267 | 0.341 | 0.442 |
| Gadoids, m 0.458 | 0.178 | 0.215 | 0.271 | 0.316 | 0.373 |
| Pelagic, m -0.401 | 0.370 | 0.315 | 0.256 | 0.224 | 0.194 |
| Pelagic, Fox | 0.324 | 0.324 | 0.324 | 0.324 | 0.324 |

For m < 0 the slope *rises* as B falls.

### Is B_ref at its fished equilibrium?
The 2026 ICES advice for the stocks that make up the pool says no, at least for gadoids ([Fiskeridödlighet i Skagerrak idag/ices_gadoid_f.md](../Fiskeridödlighet%20i%20Skagerrak%20idag/ices_gadoid_f.md), [ices_pelagic_f.md](../Fiskeridödlighet%20i%20Skagerrak%20idag/ices_pelagic_f.md), sibling notes reading ICES fulltext):
- Gadoids:
  - Northern shelf cod: the Southern and Northwestern substocks are "below MSY Btrigger and Blim" (Southern SSB 8,741 t against Blim 14,433 t).
  - Haddock: SSB about four times MSY Btrigger.
  - Saithe: below Blim after the 2026 revision.
- Pelagic:
  - North Sea autumn-spawning herring: about 1.5 times MSY Btrigger.
  - Western Baltic spring-spawning herring: below Blim.
  - Mackerel: near Blim.

A biomass-weighted gadoid pool is therefore plausibly at B/K of about 0.3-0.6 and not at equilibrium: it is still adjusting, with g(B_ref) != F. For m > 0 that gives a slope below r - mF (0.18-0.27 /yr in the table above).

### Inference
- **Model-internal tension.** In Mareld's growth budget r_max *is* the low-density net growth (`growth_budget.py`, AGENTS.md section 5). Anchor C instead implies the low-density growth is r/m = (m+1) r_max/(2m), which is 1.19 r_max for m = 0.729. Anchor A keeps r_max as the low-density growth but needs m > 0 and gives a smaller beta. If the project keeps r_max as the low-density ceiling, the consistent gadoid figure is anchor A, beta about 0.18-0.29 /yr. For pelagic fish with Thorson's m < 0 a finite low-density ceiling does not exist, and the Schaefer value (0.55 /yr) is the only one consistent with a finite ceiling.
- **beta_total is the total compensation.** Mareld's food-limited production and Beddington-DeAngelis interference already supply part of it. The defensible use of the numbers is as an upper envelope: beta_M (M1) <= beta_total - beta_explicit. Here beta_explicit should be *measured* in the model: perturb B by +/-10-20 % around B_ref with frozen policies and record the slope of net per-capita growth (production - M1 - explicit predation - F) against ln B. If beta_explicit already reaches about 0.2-0.3 /yr for gadoids, the literature gives no room for a DD M1.

---

## Q4. Where does compensation act in fish: juvenile mortality, growth, or adult mortality?

### Takeaway
- **Where the literature places it.** Compensation in fish acts mainly through density-dependent mortality in the post-settlement juvenile phase. Density-dependent growth comes second; it is common in recruited fish but usually weaker than recruitment DD (Rose et al. 2001; Lorenzen & Camp 2019, secondary; Zimmermann et al. 2018 and Rindorf et al. 2022, via the sibling note).
- **Adult mortality.** Direct density dependence in *adult* natural mortality is rarely documented. The one long-term adult-survival example found is an unexploited reef fish (Okamoto et al. 2016). Horbowy, Sparholt & Cordier (2025), who build DD-M into FMSY calculations, say they "lack empirical observations" and derive it from DD growth through the size dependence of M (Lorenzen 1996).
- **That channel is weak.** It gives gamma of about 0.03-0.14 (computed below).
- **Consequence for Mareld.** The case for a DD M1 in an age-less pool is not that adult M is density dependent. It is that M1 is the only place the missing juvenile compensation can go.

### Literature
- Rose, Cowan, Winemiller, Myers & Hilborn (2001):
  - "Population regulation via compensatory responses must ultimately occur through density-dependent changes in rates of mortality or in reproductive success ... Other responses, such as density-dependent changes in growth rate or movement are indirect, and must lead to changes in mortality rate or reproductive success if they are to affect population size."
  - "Myers and Cadigan (1993) explicitly accounted for measurement errors and isolated a density-dependent component in mortality rates of juvenile-stage demersal marine species."
  - "Predation as the cause of density-dependent mortality results from the numerical or functional responses of the predators to prey density".
  - "Density-dependent growth can result in density-dependent mortality; slower growth leads to prolonged stage duration".
  - "Adults also can exhibit density-dependent growth" (kokanee).
  - No adult density-dependent *mortality* example is given in the processes section.
  
  (fulltext) - [Rose et al. 2001, Fish Fish. 2:293-327](https://aquaticecology.tamu.edu/wp-content/uploads/sites/3/2012/07/Rose_etal_FishFisheries.pdf)
- Lorenzen & Camp (2019), as summarised by a UF/IFAS extension sheet co-authored by Lorenzen: "Eventually the fish will grow large enough that density-dependent mortality ceases and the fish are considered as recruited or recruit size (this probably occurs between 10% and 20% of the average maximum fish size; Lorenzen and Camp 2019)." (secondary; the original was not reached) - [Camp, Ahrens, Collins & Lorenzen, UF/IFAS FA222](https://edis.ifas.ufl.edu/publication/FA222); original: Lorenzen & Camp 2019, Fish. Res. 217:5-10.
- Horbowy, Sparholt & Cordier (2025) simulated 36 stocks.
  - DD-M is M(y) = M corrM(y) with corrM = corrW^p, p = -0.37 "to reflect the highest DD effects on M" (usual range -0.3 to -0.37, after Lorenzen 1996/2000). corrW is set to 0.75 or 0.57 at the maximum stock.
  - The authors "lack empirical observations to validate" DD-M when DD growth is not observed.
  - Effect: FMSY is +17 % (medium DD) or +33 % (strong DD) higher, with yield changes of only +0.2 % and +2 %.
  
  (summariser of fulltext) - [Horbowy et al. 2025, ICES JMS 82:fsaf002, doi:10.1093/icesjms/fsaf002](https://doi.org/10.1093/icesjms/fsaf002)
- Okamoto, Schmitt & Holbrook (2016): in a largely unexploited reef fish, "population density and stochastic food supply" affected adult survival, and "density dependent adult survival can dampen impacts of stochastic recruitment". No coefficients were in the abstract. (abstract) - [Okamoto et al. 2016, Ecol. Lett. 19:153-162, doi:10.1111/ele.12547](https://arxiv.org/abs/1510.07502)
- Sibling-note findings (not re-read by me; they read fulltext):
  - Lorenzen & Enberg (2002): DD growth is significant in 9 of 16 stocks, with g = 3.3 B^-1. It was not significant for North Sea cod, haddock or Norway pout.
  - Zimmermann et al. (2018): recruitment DD in 70 % of 70 NE Atlantic stocks against growth DD in 26 % or 69 %, and "97%" show a larger impact of DD recruitment.
  - Rindorf et al. (2022): late weight-at-age is about 30 % lower from minimum to maximum biomass in significant demersal stocks.
  - Hazlerigg et al. (2012): DD mortality "virtually ceased prior to maturation".
  
  [Täthetsberoende och interferens hos torskfiskar/gadoid_density_dependent_growth.md](../Täthetsberoende%20och%20interferens%20hos%20torskfiskar/gadoid_density_dependent_growth.md); [regulation_and_models.md](../Täthetsberoende%20och%20interferens%20hos%20torskfiskar/regulation_and_models.md)

### Calculation: the growth-to-M channel (Lorenzen M proportional to W^b)
If density lowers weight-at-size or asymptotic weight by a fraction d per doubling of B, M scales by (1-d)^b, so gamma = ln((1-d)^b)/ln 2. The d values come from the sibling note's reading of Lorenzen & Enberg (a 3.3 cm drop in L_inf per doubling): 8 % for cod-sized fish (L_inf 120 cm), 16 % (60 cm), 23 % (40 cm), 28 % (32 cm).

| d | b = -0.288 | b = -0.37 |
|---|---|---|
| 0.08 | gamma 0.035 | 0.045 |
| 0.16 | 0.072 | 0.093 |
| 0.23 | 0.109 | 0.140 |
| 0.28 | 0.136 | 0.175 |

These d values come from the significant stocks; North Sea gadoids were not significant, so for gadoids **gamma(adult, via growth) is about 0.03-0.07**. For gadoid M1 = 0.415 that is beta_M of about 0.015-0.03 /yr. The exponent -0.288 is from Lorenzen 1996 (secondary); -0.37 is the value Horbowy et al. use.

### Inference
- In the data, real fish stocks get most of their beta_total from juvenile mortality, i.e. steepness (Q2: per-capita recruitment elasticity about -0.85 to -0.93). Mareld has no juvenile pool. The only places that compensation can live are food-limited production (explicit, acting on all biomass) and M1.
- If a DD M1 is introduced, it should be framed and documented as a stand-in for missing juvenile compensation, not as adult biology. The density-independent floor should be at least the non-FG predation part plus a residual M0 (Q5).
- The true adult channel is an order of magnitude smaller than beta_total. It is likely already covered implicitly if the model's food-limited production lowers reserve and condition at high B.

---

## Q5. The sign of density dependence from predation by unmodelled predators

### Takeaway
- **Theory.** Per-capita mortality from a type II predator *falls* with prey density (depensatory). It is roughly *independent* of prey density when the prey is a minor item for a generalist ("bycatch"). It *rises* (compensatory) only with type III or switching, aggregative or numerical responses.
- **Marine evidence is mixed in a structured way:**
  - compensatory predation is shown mostly for newly settled juveniles on reefs (Hixon & Jones 2005: predation is the sole or major cause in 12 of 14 species);
  - inverse density dependence appears for aggregating prey (Sandin & Pacala 2005) and for grey seals on adult cod (Neuenhoff et al. 2019: exploitation per seal up 3.4-fold as 5+ cod fell 9-fold, an elasticity of about -0.56);
  - stock-level depensation is rare above about 20 % of the maximum observed stock (Myers et al. 1995; Hilborn et al. 2014).
- **For Mareld:** the unmodelled-predator part of M1 (gadoids 0.214, pelagic 0.189 /yr: generalist fish, rays, squid and sharks for which these groups are one prey among many) has no support for a *rising* term. Keep it density independent. Any DD belongs to the M0 part, as a juvenile stand-in.

### Calculation: general elasticity of per-capita predation mortality
Take an unmodelled predator of density P, with P proportional to N^k (k is the numerical or aggregative response), attack exponent q (q = 1 for type II, q = 2 for type III), and other prey summarised in O. Per-capita mortality of the focal prey is

    mu = a P N^(q-1) / (1 + a h N^q + O)

Differentiating the logarithm:

    gamma_pred = d ln mu / d ln N = k + (q - 1) - q s,    s = a h N^q / (1 + a h N^q + O)

Here s is the fraction of the predator's time spent handling the focal prey. Cases:
- **Type II, no numerical response:** gamma = -s, which lies between -1 and 0. It is depensatory, and about 0 when the prey is a minor diet item (s about 0).
- **Type III:** gamma = 1 - 2s, which is compensatory at low saturation.
- **Aggregative or numerical response:** adds +k.

The contribution to M1's elasticity is the share-weighted sum: gamma_M1 = sum over i of (M1_i/M1) gamma_i.

**Applied to the Neuenhoff numbers:** ln(3.4)/ln(49/440) = 1.224/(-2.195) = **-0.56**, i.e. s of about 0.56 for grey seals on 5+ cod, averaged over the decline (my computation).

### Literature
- Hixon & Jones (2005):
  - "For predation to cause density-dependent mortality of prey, there must be some combination of an aggregative response or Type 3 functional response over short time periods, or of a developmental response or appropriate numerical response over longer temporal scales."
  - "14 species of marine fish have been examined ... and for 12 species predation was demonstrated to be the sole or major cause of density dependence".
  - In their own 17-month experiment, "competition was the main source of density-dependent mortality, with predation being merely a proximate agent of death".
  - They note that the assumption of no inverse density dependence "is not always the case (e.g., marine fishes: Booth 1995, Sandin and Pacala 2005b)".
  
  (fulltext) - [Hixon & Jones 2005, Ecology 86:2847-2859, doi:10.1890/04-1455 (DOI from search result)](https://researchonline.jcu.edu.au/6235/1/6235_Hixon_%26_Jones_2005.pdf)
- Sandin & Pacala (2005), blue chromis on continuous reefs: loss "was not characteristic of direct density dependence, but instead was commonly inversely related to density". Predators struck "in a pattern that was strongly inversely density dependent". (summariser of the abstract) - [Sandin & Pacala 2005, Ecology 86:1520-1530](https://collaborate.princeton.edu/en/publications/fish-aggregation-results-in-inversely-density-dependent-predation/)
- McLellan, Serrouya, Wittmer & Boutin (2010):
  - "One proposed mechanism is through predation, in which a type II functional response leads to increasing predation rates as prey numbers decline."
  - However, with a multi-prey disc equation, "Allee effects caused by predation on relatively rare secondary prey may not occur because handling time of the abundant prey dominates the functional response such that secondary prey are largely 'bycatch'".
  - They cite Sinclair et al. 1998, Cantrell et al. 2001 and Gascoigne & Lipcius 2004 for type II predator-mediated Allee effects.
  
  (fulltext) - [McLellan et al. 2010, Ecology 91:286-292](https://wildlifescience.ca/wp-content/uploads/2022/11/McLellan-et-al.-2010-Ecology.pdf)
- Neuenhoff, Swain, Cox, McAllister, Trites, Walters & Hammill (2019), southern Gulf of St Lawrence cod:
  - The model "incorporated a type II functional response ... for grey seals preying on the 5+ age group".
  - "the predicted exploitation rate exerted per seal increased 3.4-fold as 5+ cod abundance declined from 440 million to 49 million".
  - "Given a type II functional response, the predation mortality generated per predator increases as prey abundance declines, resulting in a demographic Allee effect (Gascoigne and Lipcius 2004)".
  - At fine scales grey seals show prey switching (type III; Smout et al. 2014), "This may not be evident at the larger spatial ... scales ... because of the aggregative behavior of cod".
  
  (fulltext) - [Neuenhoff et al. 2019, CJFAS 76:168-184, doi:10.1139/cjfas-2017-0190](https://mmru.ubc.ca/wp-content/pdfs/Neuenhoff%20et%20al.%202019.pdf)
- Swain & Benoit (2015): very high M in depleted demersal fishes, linked to a predation-driven "predator pit" or Allee effect with grey seals. (snippet; int-res 429) - [Swain & Benoit 2015, MEPS 519:165-182, doi:10.3354/meps11012 (DOI from search result)](https://doi.org/10.3354/meps11012)
- Hilborn, Hively, Jensen & Branch (2014):
  - "Depensatory processes can include predation if the number killed by predators is largely independent of the abundance of the population".
  - Compensation mechanisms include "each individual more likely to find refuge habitat from predators at low densities".
  - In over 100 stocks depleted below 20 % of maximum, "the majority of the evidence was for no depensatory behaviour but could not rule out depensation at very low stock sizes".
  
  (fulltext) - [Hilborn et al. 2014, ICES JMS, doi:10.1093/icesjms/fsu035](https://www.arlis.org/docs/vol2/hydropower/Hilborn%202014.pdf)
- Myers, Barrowman, Hutchings & Rosenberg (1995): of 128 stocks, only 3 showed significant depensation. (snippet) - [Myers et al. 1995, Science 269:1106-1108 (record)](https://latam-cl.demo.elsevierpure.com/en/publications/population-dynamics-of-exploited-fish-stocks-at-low-population-le)
- Gascoigne & Lipcius (2004), "Allee effects driven by predation", J. Appl. Ecol. 41:801-810: keywords aggregative response, functional response, positive density dependence. (bibliographic record only) - [doi:10.1111/j.0021-8901.2004.00944.x](https://doi.org/10.1111/j.0021-8901.2004.00944.x)
- Anderson (2001), "Predator responses, prey refuges, and density-dependent mortality of a marine fish", Ecology 82:245-257, is cited by Rose et al. (2001) among juvenile predation-DD examples. (bibliographic only)
- Anderson & Gentleman (2019) on Steele & Henderson (1992): with a type II response, "clearance rates increase as prey density decreases", producing instability at low prey density. (fulltext) - [J. Plankton Res. 41:609-620, doi:10.1093/plankt/fbz042](https://pmc.ncbi.nlm.nih.gov/articles/PMC6862939/)

### Inference
- **Mareld's unmodelled predators of gadoids and pelagics are generalists:**
  - gadoids: squid, halibut, gurnards, long-rough dab, megrim, rays, monkfish (fg_library.yaml:230-233);
  - pelagic fish: sharks, rays, other fish, larvae (:185-186).
  
  For most of them the focal group is one item among many, so s is about 0 and gamma_pred is about 0. Treat this part of M1 as density independent.
- A depensatory term (gamma < 0) would matter only near collapse (Neuenhoff). It is supported at stock level only below roughly 20 % of the historical maximum, below the range the model should be calibrated for.
- A compensatory term from predation needs switching or aggregation onto *juveniles*, which the model does not resolve; in the pool it is folded into the juvenile stand-in.
- The explicit Mareld predators (gadoids on herring, porpoises) already carry type II/III and BD interference, so their sign is emergent and should not be duplicated in M1.

---

## Q6. How ecosystem and end-to-end models represent residual density dependence

### Takeaway
- **Ecosim:** no explicit DD-M for adults. Regulation comes from:
  - foraging-arena vulnerability (equivalent to BD interference without handling, per the sibling derivation);
  - feeding time, which scales both predation risk and M0 (so M0 rises as per-capita food falls, an exposure mechanism);
  - emergent Beverton-Holt recruitment in multi-stanza groups.
- **Atlantis:** M = (mL + mQ B + ...) B, with mQ recommended "low (or zero) for lower trophic levels and of moderate size for the largest of the top predators", plus Beverton-Holt recruitment.
- **mizer:** background (external) mortality depends only on size (z0 = 0.6 w_inf^(n-1) by default); compensation comes from Beverton-Holt R_max and food-dependent growth.
- **OSMOSE:** emergent, through size-based predation and starvation.
- **NPZ models:** close with quadratic zooplankton mortality. The rationale is that unresolved predators track their prey (Steele & Henderson 1992); it gives gamma = 1 for that term.

### Literature
- Walters, Pauly, Christensen & Kitchell (2000), EcoSim II: animals "may translate changes in food availability into changes in foraging time that in turn affects predation risk. These options, along with model relationships for limits on prey availability caused by predation avoidance tactics, tend to cause strong compensatory responses in modeled populations." (abstract, "submitted" version on FishBase) - [Walters et al. 2000, Ecosystems 3:70-83, doi:10.1007/s100210000011 (DOI from search result)](https://fishbase.se/Ecopath/EcoSimD.htm)
- North Sea EwE (sibling note, fulltext there): "The fraction of unexplained (other) mortality sensitive to changes in feeding time ... was assumed to be 1 for all except planktonic and sessile groups". The fitted adult gadoid vulnerabilities are 3-6 (1991). (secondary, via sibling) - [Mackinson & Daskalov 2007, Cefas TR 142](https://www.cefas.co.uk/publications/techrep/tech142.pdf); sibling [regulation_and_models.md Q3](../Täthetsberoende%20och%20interferens%20hos%20torskfiskar/regulation_and_models.md)
- Atlantis User Guide (sibling note, fulltext there):
  - "linear mortality terms should be as low as possible ... while quadratic mortality should be low (or zero) for lower trophic levels and of moderate size for the largest of the top predators";
  - "If high linear and quadratic mortality values are used, the model will be insensitive to species interactions."
  
  (secondary, via sibling) - [CSIRO Atlantis User Guide Part I, 2025](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf)
- Fulton, Smith & Johnson (2003): "A quadratic mortality term applied to intermediate predators ... is sufficient to achieve plausible model behaviour." (snippet, via sibling) - [Ecol. Model. 169:157-178](https://doi.org/10.1016/S0304-3800(03)00268-0)
- mizer `setExtMort`: "The external mortality is all the mortality that is not due to fishing or predation by predators included in the model"; mu_ext = z0 + z_ext w^d, with z0 = z0pre w_inf^z0exp, "Default value is 0.6" and z0exp = n - 1. There is no abundance term. (summariser of the docs page) - [mizer reference](https://sizespectrum.org/mizer/reference/setExtMort.html)
- Steele & Henderson (1992) as reproduced by Anderson & Gentleman (2019):
  - "Mortality is either linear (w = 1 ...) or quadratic (w = 2 ...)";
  - SH92 is "the classic citation when it comes to justifying the functional form of closure";
  - "The use of linear mortality generally promotes instability".
  - Edwards & Yool (2000) and Edwards & Brindley (1999) are the critiques.
  
  (fulltext) - [Anderson & Gentleman 2019, J. Plankton Res. 41:609-620](https://pmc.ncbi.nlm.nih.gov/articles/PMC6862939/)

### Calculation
For the Atlantis or linear-plus-quadratic form M(B) = mL + mQ B: gamma = mQ B_ref/(mL + mQ B_ref), and beta_M = mQ B_ref. The pure quadratic case has gamma = 1. Matching a target gamma at B_ref with total M1_ref gives mQ = gamma M1_ref/B_ref and mL = (1 - gamma) M1_ref, so the floor is mL. This is the brief's linear form with M0 = mL and k = mQ.

### Inference
- Ecosim's feeding-time coupling of M0 is the closest precedent for a DD M1 in fish groups. It works through exposure: hungrier animals forage longer and die more.
- Mareld's Eat vs Rest/hide behaviour, with hidden biomass protected from predation, is the mechanistic analogue for *explicit* predation. A DD M1 written on biomass would duplicate part of it. An exposure-weighted M1 (AGENTS.md section 139, implemented but switched off) is the Ecosim-like option.
- Atlantis's guidance supports small mQ for fish and larger mQ for top predators, and warns that large closure terms swamp interactions.
- The NPZ quadratic closure fits zooplankton: its unmodelled predators (carnivorous zooplankton, jellyfish, larvae) do track zooplankton biomass. The zooplankton sibling note covers this in depth.

---

## Implications for Mareld (inference; numbers are envelopes, not calibrations)

| Group | M1 now (/yr) | beta_total envelope at B_ref (/yr) | gamma if ALL of beta_total sat on M1 | Literature-supported direct adult DD-M | Suggested treatment |
|---|---|---|---|---|---|
| Gadoids | 0.415 (M0 0.20 + non-FG predation 0.214) | 0.18-0.39 (anchor C; low end if B/K 0.3-0.5; anchor A 0.18-0.29) | 0.43-0.94 | gamma about 0.03-0.07 (via the W^-0.3 channel) | Measure beta_explicit first. DD only on the M0 part, as a juvenile stand-in: beta_M <= beta_total - beta_explicit. Keep the 0.214 predation part constant. |
| Pelagic fish | 0.580 (M0 0.391 + non-FG predation 0.189) | 0.22-0.32 (anchor C, Thorson shape); 0.55 if Schaefer | 0.38-0.56 (0.94 Schaefer) | similar or smaller | Same as gadoids. Aggregation (schooling) argues against any compensatory unmodelled-predation term (Sandin & Pacala). |
| Porpoises | 0.10 | at K with F = 0, beta = theta r_max = 0.1 theta; theta 1-3 gives 0.10-0.30, lower if B_ref < K | 1-3 | not assessed here (mammal sibling note) | Absolute values are tiny (<= 0.3 /yr). Sibly found mammal theta < 1 against the earlier convex expectation for large mammals; the GPDD mammal slopes fail the sanity bound (Q1). |
| Zooplankton | 19.6 | n/a (theta-logistic not applicable) | n/a | n/a | NPZ precedent: linear + quadratic closure, gamma = mQ B/(mL + mQ B). See the zooplankton sibling note. |

Key cautions:
1. **The upper envelope is not a target.** gamma for M1 = (beta_total - beta_explicit)/M1, floored at 0. The explicit food and interference mechanisms (BD w = 1.0 for gadoids, 0.7 for pelagic) may already supply much of beta_total.
2. **Use a floor.** Use M1(B) = M1_floor + M_DD(B), where M1_floor >= the non-FG predation part, rather than a pure power law. A power law with gamma about 0.5 lets M1 fall to about 70 % of its reference value at half B_ref; computed: 0.5^0.5 = 0.707.
3. **The shape uncertainty in beta_total is about 2x, and the anchor uncertainty is another 2x.** The literature cannot pin gamma for M1 to better than "0 to about 0.9 of M1 per e-fold of B", and the low end is supported by the absence of direct adult evidence.

---

## Source status

| Source | How read | Used for |
|---|---|---|
| [Sibly et al. 2005](https://kevintshoemaker.github.io/EECB-703/Sibly%20et%20al.%20-%202005%20-%20On%20the%20Regulation%20of%20Populations%20of%20Mammals,%20Birds.pdf) | Fulltext; Fig. 3-4 figure read (uidaho copy 403/301; Reading centaur behind Anubis bot block) | theta distribution, slope vs measurement error |
| [Brook & Bradshaw 2006](https://conservationbytes.com/wp-content/uploads/2014/12/brook-bradshaw-2006-ecology.pdf) | Fulltext | DD support by taxon |
| [Herrando-Perez et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3433995/) | Fulltext (HTML) | Gompertz strength by taxon |
| [Clark et al. 2010](https://doi.org/10.1111/j.2041-210X.2010.00029.x) | Snippet (Wiley 403; Adelaide page empty) | theta-r_m trade-off |
| [Polansky et al. 2009](http://hdl.handle.net/2263/13888) | Summariser of abstract (UCLA PDF link served HTML) | likelihood ridges |
| [Ross 2009](https://ideas.repec.org/a/eee/ecomod/v220y2009i23p3472-3474.html), [Ross 2006](https://digital.library.adelaide.edu.au/items/d7190d5f-dc8c-4d42-a430-beb491d4c657), [Barker & Sibly 2008](https://www.sciencedirect.com/science/article/abs/pii/S0304380008004109) | Snippets | theta critiques |
| Knape & de Valpine 2011, via [Johnson 2024](https://arxiv.org/abs/2412.17313) | Secondary (de Valpine page 404) | observation-error illusion |
| [Thorson et al. 2012](https://jensen.limnology.wisc.edu/wp-content/uploads/sites/147/2021/11/Thorson_etal_2012b_CJFAS.pdf) | Fulltext (cdnsciencepub 403) | Bmsy/B0, phi by order; model equations |
| [IATTC review](https://www.iattc.org/GetAttachment/d949eefc-7b46-4b28-a562-d986286ac48f/SAC-08-05e(ii)_Review-of-limit-reference-points.pdf) | Fulltext | 26-46 % range |
| [Sparholt et al. 2021](https://doi.org/10.1093/icesjms/fsaa175) | Snippet | unresolved u discrepancy |
| [Myers, Bowen & Barrowman 1999](https://sedarweb.org/documents/s8rd21-maximum-reproductive-rate-of-fish-at-low-population-sizes/) | Fulltext | alpha, steepness Gadidae/Clupeidae |
| [Thorson 2020](https://doi.org/10.1111/faf.12427) | Fulltext (NOAA repository PDF) | FishLife h, r by order |
| [Rose et al. 2001](https://aquaticecology.tamu.edu/wp-content/uploads/sites/3/2012/07/Rose_etal_FishFisheries.pdf) | Fulltext | where compensation acts |
| [UF/IFAS FA222](https://edis.ifas.ufl.edu/publication/FA222) (Lorenzen & Camp 2019) | Fulltext of the sheet; secondary for L&C | end of DD mortality at 10-20 % of Lmax |
| [Horbowy et al. 2025](https://doi.org/10.1093/icesjms/fsaf002) | Summariser of fulltext | DD-M form, no empirical adult evidence, FMSY effects |
| [Okamoto et al. 2016](https://arxiv.org/abs/1510.07502) | Abstract | adult DD survival (reef fish) |
| [Hixon & Jones 2005](https://researchonline.jcu.edu.au/6235/1/6235_Hixon_%26_Jones_2005.pdf) | Fulltext | conditions for predation DD; 12/14 species |
| [Sandin & Pacala 2005](https://collaborate.princeton.edu/en/publications/fish-aggregation-results-in-inversely-density-dependent-predation/) | Summariser of abstract | inverse DD from aggregation |
| [McLellan et al. 2010](https://wildlifescience.ca/wp-content/uploads/2022/11/McLellan-et-al.-2010-Ecology.pdf) | Fulltext | multi-prey "bycatch" argument |
| [Neuenhoff et al. 2019](https://mmru.ubc.ca/wp-content/pdfs/Neuenhoff%20et%20al.%202019.pdf) | Fulltext | type II seal predation, 3.4-fold |
| [Swain & Benoit 2015](https://doi.org/10.3354/meps11012) | Snippet (int-res 429) | predator pit |
| [Hilborn et al. 2014](https://www.arlis.org/docs/vol2/hydropower/Hilborn%202014.pdf) | Fulltext | depensation rare; predation as depensatory mechanism |
| [Myers et al. 1995](https://latam-cl.demo.elsevierpure.com/en/publications/population-dynamics-of-exploited-fish-stocks-at-low-population-le) | Snippet | 3/128 depensatory |
| [Gascoigne & Lipcius 2004](https://doi.org/10.1111/j.0021-8901.2004.00944.x), Anderson 2001 | Bibliographic only | - |
| [Walters et al. 2000 EcoSim II](https://fishbase.se/Ecopath/EcoSimD.htm) | Abstract (submitted version) | feeding-time compensation |
| [mizer setExtMort](https://sizespectrum.org/mizer/reference/setExtMort.html) | Summariser of docs | background mortality form |
| [Anderson & Gentleman 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6862939/) | Fulltext (HTML) | NPZ closure, type II instability |
| Atlantis guide, Mackinson & Daskalov 2007, Fulton et al. 2003, Lorenzen & Enberg 2002, Zimmermann 2018, Rindorf 2022 | Secondary, via sibling notes (fulltext there) | model guidance; DD growth |
| SPiCT vignette (GitHub raw) | Failed (14-byte response) | intended textbook check of Bmsy formula; replaced by Thorson eqs. 1-2 |
| ICES 2026 stock status | Via sibling notes `Fiskeridödlighet i Skagerrak idag/*` | B_ref vs equilibrium |
| Library values | Local file `fgconfig/fg_library.yaml` lines 181-242, 98-101 | r_max provenance, M1 composition |
