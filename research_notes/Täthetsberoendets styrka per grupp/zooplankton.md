# Density-dependent M1 for the Mareld `zooplankton` (mesozooplankton) pool: how strong should it be?

Literature review, 2026-10-15. No code or configuration was changed.

**Question.** Should the zooplankton `natural_mortality` (M1) depend on density, and how strongly?
The task defines the strength at B_ref as follows:

- beta_M = dM1/d ln B at B_ref (per year);
- gamma = beta_M / M1(B_ref), a dimensionless elasticity.

| form | elasticity at B_ref |
|---|---|
| M1_ref (B/B_ref)^gamma | gamma |
| M0 + k B | gamma = k B_ref / M1_ref |
| M0 + (M1_ref - M0)(B/B_ref)^kappa | gamma = (1 - M0/M1_ref) kappa |
| per-capita a B/(K + B) (hyperbolic closure) | gamma = K/(K + B_ref) |

gamma = 1 is the classic quadratic closure, gamma = 0 a constant rate. A negative gamma means
per-capita mortality falls with density (saturating predators).

**Tags.**

- **[F]** fulltext read;
- **[A]** abstract only;
- **[S]** search snippet or press summary only;
- **[2]** secondary (another paper's or a sibling note's account of the source);
- **[C]** computed here;
- **[I]** my inference;
- **[M]** model fact from the repository.

Numbers tagged [S] or [2] are not verified against the original.

## Model context [M]

- **M1.** `natural_mortality: 0.0134` per tick x 1460 = **19.56 /yr**
  (fgconfig/fg_library.yaml, zooplankton block).
  - Derivation (library comment and `reports/Zooplankton energibudget.md`): total copepod
    mortality ~25 /yr minus the 21 % taken by modelled fish.
  - M1 therefore stands for carnivorous zooplankton, fish larvae, jellyfish, other non-modelled
    fish and non-predation losses.
- **Other parameters.** r_max 33.4 /yr; `interference: 0.0`.
- **Reference biomass.** B_ref = 35,000 t / 3,600 km2 = **9.72 t WW/km2**.
- **Growth budget.** g = (r_max + M1 + checked M2) / ((s* - u) 1460), so M1 is paid at B_ref by
  construction (`lib/world/growth_budget.py`).
- **Food.** Food limitation is explicit: Holling II on phytoplankton, half-saturation 20 t/km2.
  Phytoplankton settles at 3-5 t/cell in policy runs (report sections 7-8).
- **Fish predation.** Pelagic fish predation is explicit (Holling II with interference).
- **Temperature.** Zooplankton is on the `mean_0_50m` layer (mareld2.yaml), annual mean
  9.3 C, monthly range 5.2-13.8 C. Q10 acts only on metabolism; M1 is not temperature-scaled.
- **Observed equilibrium.** Under the frozen policies zooplankton settles near 0.66 B_ref (report
  sections 7-8).

## Source status and failed fetches

- **Fulltext read:**
  - Ohman & Hirche 2001 (eScholarship PDF, obtained through WebFetch's binary save, then
    pdftotext);
  - Ohman et al. 2002 and Ohman et al. 2004 (eScholarship, same route);
  - Hirst & Kiorboe 2002 (DTU Orbit);
  - Eiane & Ohman 2004 (int-res);
  - Vergara-Soto et al. 2010 (SciELO);
  - Anderson et al. 2015 EMPOWER (GMD);
  - Hill Cruz et al. 2021 (Biogeosciences 18:2891-2916);
  - Edwards 1997 PhD thesis (White Rose eTheses; the source behind Edwards & Brindley 1999 and
    Edwards & Yool 2000).
- **Abstract only:**
  - Steele & Henderson 1992;
  - Edwards & Yool 2000;
  - Plourde, Pepin & Head 2009; Plourde, Maps & Joly 2009;
  - Ohman et al. 2008; Eiane et al. 2002;
  - Gentleman & Neuheimer 2008; Fulton et al. 2003; Omta et al. 2022; Hatton et al. 2015;
  - Mitra 2009 (abstract as summarised by the IDEAS/RePEc page);
  - Bamstedt 1990; Olesen et al. 1994; Olesen 1995; Hansson et al. 2005;
  - Lindahl & Hernroth 1983; Lynam et al. 2005; Attrill et al. 2007;
  - Majaneva et al. 2013 (truncated).
- **Secondary only:**
  - Fasham et al. 1990 and Fasham 1993/1995 (via the Edwards thesis and EMPOWER);
  - Steele & Henderson 1981 (via the Edwards thesis);
  - Hygum et al. 2000 (via Ohman et al. 2004);
  - Peterson, Tiselius & Kiorboe 1991 (via sibling note `plankton.md`).
- **Not obtained:**
  - Edwards & Brindley 1999 (Bull. Math. Biol.; no abstract available; the Edwards 1997 thesis
    was used instead);
  - Anderson, Gentleman & Sinha 2010 (no abstract retrieved, nothing read);
  - Kiorboe 2008 book (nothing read);
  - Gentleman & Neuheimer 2008 fulltext (the academic.oup.com PDF returned HTML).
- **Failed fetches:**
  - nature.com (redirect to login);
  - epic.awi.de (Anubis bot block);
  - sciencedirect.com (403);
  - Wiley `pdfdirect` for L&O papers (returned HTML);
  - escholarship.org through curl (CloudFront 403; WebFetch worked);
  - Semantic Scholar API (429 after 5 calls; abstracts elided by the publisher for Elsevier and
    Springer).

---

## Q1. Empirical density-dependent mortality in copepods

### Takeaway
All the direct evidence for density-dependent (DD) copepod mortality concerns the **eggs (to N2-N3)
of the broadcast-spawning *Calanus finmarchicus*** and is attributed to cannibalism by adult
females and C5.

- **Where it is found:** Norwegian Sea (Ocean Station M), Georges Bank, and the Gulf of St
  Lawrence / Scotian Shelf.
- **Stage-level elasticity at the mean:** gamma_egg = 0.27-0.30 (Ohman et al. 2004 regressions)
  to 0.72 (Ohman & Hirche 2001, regressed on females + C5).
- **Conditional:** on Georges Bank egg mortality is flat below ~2000 females m-2.
- **Brooders:** the egg-brooding *Pseudocalanus* shows **no** DD mortality.
- **No study** reports DD mortality of copepodites or adults, which hold almost all the biomass.
- **Hirst & Kiorboe 2002** did not test density. Their global rates are steady-state averages by
  construction. Their main covariates are temperature (Q10 ~2) and, for broadcast spawners, body
  size.

### Cited Findings
- **Ohman & Hirche 2001, Station M (Norwegian Sea), March-June 1997.**
  - Sampled every 1-2 d for 80 d. "Per capita mortality rates of eggs of Calanus finmarchicus
    ... are a function of the abundance of adult females and juveniles."
  - Median egg mortality 1.76 d-1 ("equivalent to 82.8% d-1"); only 2.7 % of eggs survive to N1;
    eggs hatch in 2.16 d at 6.4 C.
  - Fig. 3, model II regression of egg mortality on adult female + C5 abundance (0-16,000 m-2):
    y = 0.00033 x + 0.488, r2 = 0.608, P < 0.0001.
  - Egg mortality was independent of diatom, ciliate and phytoplankton biomass.
  - Non-viable eggs averaged 10.2 % of egg mortality.
  - The authors favour cannibalism and note: "Rates of zooplankton loss are often treated as
    constant and linear."
  - **[F]** [Nature 412:638-641, doi:10.1038/35088068](https://doi.org/10.1038/35088068)
    (PDF via [eScholarship](https://escholarship.org/uc/item/2h63s0h0)).
  - Press release: "a tenfold increase in adult egg-producing female copepods" leaves "half as
    many eggs" surviving. **[S]** [ScienceDaily 2001](https://www.sciencedaily.com/releases/2001/08/010809070341.htm)
- **Ohman et al. 2004, five North Atlantic sites.**
  - Rates and stage patterns of C. finmarchicus mortality "vary appreciably across the North
    Atlantic"; the predator field shapes them.
  - Georges Bank egg-N2 mortality: Y = 0.000079 X + 0.3673, r2 = 0.490. Station M egg mortality:
    Y = 0.000252 X + 1.1915, r2 = 0.415. X = adult females; the axis units were not legible in
    the text extraction.
  - Means: egg-N2 on Georges Bank 0.50 d-1 (median); eggs at Station M 1.7 d-1.
  - "Conditionally density-dependent: egg-N2 mortality rates were independent of adult population
    size at low adult abundances, but became directly proportional to adult abundance at high
    population abundances."
  - Hygum et al. 2000 mesocosms, started from sorted eggs with no cannibalism: 0.011-0.012 d-1.
  - **[F]** [ICES J. Mar. Sci. 61:687-697, doi:10.1016/j.icesjms.2004.03.016](https://doi.org/10.1016/j.icesjms.2004.03.016)
    (PDF via [eScholarship](https://escholarship.org/uc/item/7b18b86f)).
- **Ohman et al. 2002, Georges Bank, 30 cruises.**
  - The egg-N2 relationship "is relatively weak at low abundances of adult females, but becomes
    pronounced at female abundances exceeding approximately 2000 females m-2."
  - The density dependence "introduces a quadratic mortality term into population dynamic models."
  - "The egg-brooding Pseudocalanus spp., in contrast, show no evidence of density-dependent
    mortality."
  - Mortality is not usefully predicted by body size or temperature.
  - Low-density egg predation is mainly by suspended hydroids; at high density, cannibalism.
  - **[F]** [Hydrobiologia 480:55-68, doi:10.1023/A:1021228900786](https://doi.org/10.1023/A:1021228900786)
    (PDF via [eScholarship](https://escholarship.org/uc/item/4f52v6ds)).
- **Ohman et al. 2008, Georges Bank.** Egg-N3 mortality follows an index of predation potential.
  Late copepodid and adult C. finmarchicus, plus hydroids, have "the largest quantitative predation
  effect". **[A]** [L&O 53:1643-1655, doi:10.4319/lo.2008.53.4.1643](https://doi.org/10.4319/lo.2008.53.4.1643)
- **Plourde, Maps & Joly 2009, lower St Lawrence estuary and Gulf.** "Phytoplankton biomass and
  female abundance exerted opposite effects in the control of mortality", read as cannibalism
  relaxed by food. Coefficients not seen. **[A]** [J. Plankton Res. 31:371-388, doi:10.1093/plankt/fbn126](https://doi.org/10.1093/plankt/fbn126)
- **Plourde, Pepin & Head 2009, AZMP 1999-2006.**
  - The "largest signal" is temperature.
  - "Density-dependent mortality, associated with the abundance of C6 females, was the main factor
    influencing mortality in the egg-C1 transition during the period of population growth."
  - **[A]** [ICES J. Mar. Sci. 66:1942-1958, doi:10.1093/icesjms/fsp167](https://doi.org/10.1093/icesjms/fsp167)
- **Eiane et al. 2002, two Norwegian fjords.**
  - Lurefjorden (tactile, invertebrate predators): Calanus mortality ~0.35 d-1 early, ~0.00 d-1 in
    early copepodids.
  - Sorfjorden (visual, fish predators): ~0.08 d-1 throughout.
  - The predator guild sets the pattern. No density test is reported.
  - **[A]** [L&O 47:636-645, doi:10.4319/lo.2002.47.3.0636](https://doi.org/10.4319/lo.2002.47.3.0636)
- **Eiane & Ohman 2004, FLEX 1976, northern North Sea.**
  - Pseudocalanus: 0.11 d-1 at N1-N2; < 0.03 d-1 for N3-N5; ~0.00 for C2-C4; 0.05 d-1 at C5.
  - Oithona: < 0.001 d-1 after the naupliar stages.
  - No density analysis.
  - **[F]** [MEPS 268:183-193, doi:10.3354/meps268183](https://doi.org/10.3354/meps268183)
- **Hirst & Kiorboe 2002, global copepod mortality.**
  - Indirect method; assumes steady state and constant post-hatch mortality.
  - Broadcast-spawner regressions: ln beta = 0.0725 T - 1.112 for eggs and 0.0725 T - 3.415
    post-hatch. Eggs therefore die e^(3.415 - 1.112) = **10x** faster than post-hatch stages.
  - Rates "can be very variable spatio-temporally as a result of patchy food, predation and
    cannibalism".
  - Density is not a covariate anywhere in the paper.
  - Laboratory (non-predation) rate 0.019 d-1 at 5 C, slope 0.061 per C. Predation ~2/3-3/4 of
    the total.
  - **[F]** [MEPS 230:195-209, doi:10.3354/meps230195](https://doi.org/10.3354/meps230195)
    ([DTU Orbit PDF](https://backend.orbit.dtu.dk/ws/files/3696723/Kiorboe5.pdf)).
- **Peterson, Tiselius & Kiorboe 1991, Skagerrak transect, August.** Egg production was 25 % of
  total copepod production. **[2]** via sibling note `Litteraturbaserad growth rate per FG/plankton.md`;
  [J. Plankton Res. 13:131-154, doi:10.1093/plankt/13.1.131](https://doi.org/10.1093/plankt/13.1.131)

### Calculations [C]
- **Stage-level elasticity.** For a linear regression m = a + b x, gamma_stage(x) = b x / (a + b x).
  A model II or OLS line passes through the centroid, so at the mean mortality
  gamma = 1 - a/m_mean:
  - Ohman & Hirche 2001 (on females + C5): gamma_egg = 1 - 0.488/1.76 = **0.72**, at
    x* = (1.76 - 0.488)/0.00033 = 3,855 m-2.
    - Over the observed range: 0.40 at x = 1,000 and 0.92 at x = 16,000.
    - beta_egg = 0.00033 x 3855 = 1.27 d-1 per unit ln x.
  - Ohman et al. 2004, Station M (on females only): 1 - 1.1915/1.7 = **0.30** (X* = 2,018).
    - This is the same site and data as 2001 with a different regressor. It is not a
      disagreement: female + C5 abundance spans more of the range than females alone.
  - Ohman et al. 2004, Georges Bank (egg-N2): 1 - 0.3673/0.50 = **0.27** (X* = 1,680), close to
    Ohman et al. 2002's threshold of ~2000 females m-2.
    - Approximate: 0.50 d-1 is the egg-N2 median over all cruises (stage-pattern figure). The
      regression comes from a subset (seven pump cruises; Ohman et al. 2002), so its centroid
      need not equal 0.50.
- **What the egg DD does to the biomass pool.** Egg loss per unit pool biomass is
  L = eps (1 - S), with:
  - eps = pool-specific egg production;
  - S = exp(-m_e D_e) = egg survival;
  - D_e = 2.16 d = egg duration.

  Assume x is proportional to B. Then dL/d ln B = eps D_e b x S.
  - At the Station M median (m_e = 1.76): S = exp(-3.80) = 0.022 and D_e b x = 2.75, so
    dL/d ln B = **0.061 eps**.
  - The maximum over x is exp(-a D_e - 1) = **0.128 eps**, at x = 1/(D_e b) = 1,403 m-2.
  - Egg production is 25 % of production [2], and realised production is about total mortality,
    ~25 /yr. So eps ~ 0.25 x 25 = 6.25 /yr for the whole pool.
  - Pool-level beta from egg DD = 0.061 x 6.25 = 0.38 /yr (maximum 0.80 /yr).
  - Divided by M1 = 19.56 /yr: **gamma_egg,pool ~ 0.02, at most 0.04**.
    - These figures assume every broadcast spawner behaves like Calanus.
    - If only Calanus (~1/3 of the pool) does: ~0.007.
  - Why it is so small: at the reference density ~98 % of eggs already die, so extra density can
    remove only the ~2 % that survive. Over the whole observed female range, the egg loss moves
    only from 0.65 eps (x = 0) to ~1.0 eps.

### Inferences [I]
- The empirical DD acts on a stage holding negligible biomass. Its population consequence is
  compensation in recruitment numbers: survival halves for a 10x rise in females. That is DD in
  net production with a one-generation lag, not an instantaneous per-biomass loss.
- Cannibalised eggs are partly assimilated by the cannibals, so the net loss to the pool is
  smaller still.
- M1 was calibrated from Hirst & Kiorboe's **post-hatch** total mortality (~25 /yr) minus the
  fish share. Hirst & Kiorboe keep egg mortality separate, and it is 10x higher. Egg losses were
  therefore never part of M1 as calibrated. Egg-cannibalism DD is not a property of M1 at all; if
  it belongs anywhere, it belongs in the production/recruitment term.
- This line of evidence supports gamma ~ 0 (at most ~0.04) for a biomass-pool M1. It does not
  support a quadratic closure, even though Ohman et al. 2002 use that language at the
  population-number level.
- Two of the Skagerrak assemblage's dominants, Pseudocalanus and Oithona, are sac spawners with no
  egg-cannibalism pathway (Ohman et al. 2002; Eiane & Ohman 2004). This lowers the pool-level
  effect further.

---

## Q2. Linear vs quadratic closure in NPZ models

### Takeaway
- **What the quadratic term stands for.** In NPZ models it represents unresolved carnivores
  "whose population tracks that of the zooplankton". The per-capita loss d Z then assumes predator
  biomass proportional to prey. It is a modelling closure, not a measured mortality law.
- **Common forms and coefficients:**
  - linear (Fasham et al. 1990: 0.05 d-1);
  - quadratic (Steele & Henderson 1981/1992: d ~1.0 (g C m-3)^-1 d-1; Fennel et al. 2006 to
    Lima & Doney 2004: 0.025-0.25 (mmol N m-3)^-1 d-1);
  - linear + quadratic (Yool et al. / EMPOWER: 0.02 d-1 + 0.34 (mmol N m-3)^-1 d-1);
  - saturating Michaelis-Menten (Fasham 1993; Omta et al. 2022 "hyperbolic").
- **Elasticities.** In gamma terms, model practice spans 0 (FDM90) to ~0.75-0.9 (linear +
  quadratic, evaluated at Mareld's B_ref) and 1 (pure quadratic).
- **Stability.** Steele & Henderson's finding (quadratic suppresses oscillations, linear gives
  them) was challenged:
  - quadratic closure also gives limit cycles (Edwards & Yool 2000; Edwards 1997);
  - stability hinges on the food-limited clearance rate of the grazing response
    (Gentleman & Neuheimer 2008);
  - the closure is "mathematically (rather than biologically) justified" (Mitra 2009).
- **Model sensitivity.** Ecosystem models are more sensitive to the grazing form than to the
  closure (Fulton et al. 2003), though linear-only closure suits nutrient or harvest scenarios
  poorly.

### Cited Findings
- **Steele & Henderson 1992.** "The mathematical form used to close the system and the values
  given to the parameters have very marked effects on the overall response"; North Atlantic vs
  Pacific differences "may depend on this aspect as much as on the physical cycles"; "Our
  ignorance in both areas is considerable." **[A]** [J. Plankton Res. 14:157-172, doi:10.1093/plankt/14.1.157](https://doi.org/10.1093/plankt/14.1.157)
- **Edwards 1997 thesis (basis of Edwards & Brindley 1999 and Edwards & Yool 2000).**
  - Higher predation by invertebrate carnivores "is modelled by the quadratic form -dZ^2, since the
    carnivore population is assumed to change in proportion to the zooplankton population".
  - Steele & Henderson 1981 used d = 0.25-1.75 (g C m-3)^-1 d-1 and found a behaviour change near
    1.0. Edwards uses a default of 1.0 and a range of 0.25-2.0.
  - "In contrast to a previous hypothesis, the three variables can undergo oscillations when a
    quadratic function for zooplankton mortality is used." Periods are ~1 month.
  - The linear form "does not greatly affect the nature of the Hopf bifurcations and oscillations,
    although it does eliminate the homoclinicity and hysteresis".
  - Literature linear rates: 0.04 d-1 (Wroblewski 1989), 0.05 (Fasham et al. 1990), 0.07 (Evans &
    Parslow 1985).
  - Fasham 1993 switched to a Michaelis-Menten loss with a maximum of 0.3 d-1. Fasham 1995
    concluded that if carnivores (Holling III) are not modelled explicitly, "a quadratic closure
    term ... is indeed an appropriate function".
  - Steady state Z* ~0.075 g C m-3 at d = 1.
  - **[F]** for the thesis; **[2]** for the Fasham and Steele & Henderson 1981 values.
    [Edwards 1997, PhD thesis, Univ. of Leeds](https://etheses.whiterose.ac.uk/21072/)
- **Edwards & Yool 2000.** They reveal "an inconsistency in the normalization of Steele and
  Henderson's models" and show "unforced short-term oscillations (limit cycles) can occur when a
  quadratic closure term is used"; "the choice of closure term can strongly influence the dynamics".
  **[A]** [J. Plankton Res. 22:1085-1112, doi:10.1093/plankt/22.6.1085](https://doi.org/10.1093/plankt/22.6.1085)
- **Edwards & Brindley 1999.** Not obtained; see the thesis.
  [Bull. Math. Biol. 61:303-339, doi:10.1006/bulm.1998.0082](https://doi.org/10.1006/bulm.1998.0082)
- **Anderson et al. 2015 (EMPOWER-1.0).**
  - "The linear term represents density-independent natural mortality, whereas the quadratic term
    is considered to be due to predation by carnivores (whose population tracks that of the
    zooplankton)."
  - mZ = 0.02 d-1 (Yool et al. 2011); mZ2 = 0.34 (mmol N m-3)^-1 d-1 (Oschlies & Schartau 2005).
  - Removing the linear term "had relatively little impact"; removing the quadratic term changed
    phytoplankton by up to 50 %.
  - Station India fitted better with mZ = 0, and "Fasham (1993) [has] similarly not included a
    linear zooplankton loss term".
  - **[F]** [GMD 8:2231-2262, doi:10.5194/gmd-8-2231-2015](https://doi.org/10.5194/gmd-8-2231-2015)
- **Hill Cruz et al. 2021, northern Humboldt Current model.**
  - Literature quadratic rates "vary over a large range, from 0.025 (Fennel et al., 2006) up to
    0.25 (mmol N m-3)-1 d-1 (Lima and Doney, 2004)". The term "is often adjusted to tune the
    model".
  - A quadratic term "may also be interpreted as an increase in diseases ... cannibalism or
    increased predation due to high densities of prey".
  - Changing the quadratic coefficient by +50 % / -50 % changed large-zooplankton biomass by only
    -12 % / +19 %.
  - Linear mortality responds "reactively" to perturbations; DD closure buffers them (Neubert et
    al. 2004).
  - **[F]** [Biogeosciences 18:2891-2916, doi:10.5194/bg-18-2891-2021](https://doi.org/10.5194/bg-18-2891-2021)
- **Gentleman & Neuheimer 2008.** "Modellers have long been aware that the mathematical form of
  zooplankton mortality, or closure, significantly affects the dynamics". Oscillation vs stability
  is not due to satiation per se. "The necessary condition for ecological stability ... is related
  to food-limited clearance rates." **[A]** [J. Plankton Res. 30:1215-1231, doi:10.1093/plankt/fbn078](https://doi.org/10.1093/plankt/fbn078)
- **Fulton, Smith & Johnson 2003.** "Model behaviour is far more sensitive to the form of the
  grazing term than to that of the mortality terms"; "a quadratic mortality term applied to
  intermediate predators ... is sufficient"; "use of simple linear functional responses and
  mortality terms is unsuitable for models used to consider a range of nutrient loading or
  harvesting scenarios". **[A]** [Ecol. Model. 169:157-178, doi:10.1016/S0304-3800(03)00268-0](https://doi.org/10.1016/S0304-3800(03)00268-0)
- **Mitra 2009.** Zooplankton losses are largely cannibalism and carnivory, which closure terms
  represent "by mathematically (rather than biologically) justified closure functions". Explicit
  intraguild predation gives similar biomass but very different fates of primary production, and
  closures "can seriously misrepresent the fate of primary production". **[A]** (summary on
  [IDEAS/RePEc](https://ideas.repec.org/a/eee/ecomod/v220y2009i5p611-620.html));
  [Ecol. Model. 220:611-620, doi:10.1016/j.ecolmodel.2008.12.008](https://doi.org/10.1016/j.ecolmodel.2008.12.008)
- **Omta et al. 2022.** An implicit predator with a Holling II response and "empirically observed
  predator-prey scaling relations" gives "a hyperbolic mortality formulation that is a hybrid
  between the linear and quadratic forms". With it there is no destabilisation by enrichment.
  **[A]** [Ecol. Model. 475:110183, doi:10.1016/j.ecolmodel.2022.110183](https://doi.org/10.1016/j.ecolmodel.2022.110183)
- **Hatton et al. 2015.** Across 2,260 communities, including plankton, "predator and prey biomass
  follow a general scaling law with exponents consistently near 3/4". **[A]** [Science 349:aac6284, doi:10.1126/science.aac6284](https://doi.org/10.1126/science.aac6284)
- **Anderson, Gentleman & Sinha 2010.** Not read; on grazing formulations.
  [Prog. Oceanogr. 87:201-213, doi:10.1016/j.pocean.2010.06.003](https://doi.org/10.1016/j.pocean.2010.06.003)

### Calculations [C]: model coefficients evaluated at Mareld's B_ref
- **Unit chain.** B_ref = 9.72 g WW m-2.
  - x DW/WW 0.186 (sibling note `copepod_energetics.md`) x C/DW 0.40 (assumed; typical copepod
    value, not sourced here) = **0.723 g C m-2**.
  - Redfield C:N 6.625 mol/mol = 5.68 g/g gives 0.127 g N m-2 = 9.09 mmol N m-2.
  - Over the model layer H = 50 m: **0.0145 g C m-3 = 0.18 mmol N m-3**.
  - Over a 20 m surface layer: 0.036 g C m-3 = 0.45 mmol N m-3.
- **Per-capita quadratic rate d Z at B_ref**, for H = 50 m (H = 20 m in brackets):

| source | coefficient | quadratic part (/yr) | linear part (/yr) | gamma = quad/(quad + lin) |
|---|---|---|---|---|
| Fasham et al. 1990 [2] | linear 0.05 d-1 | 0 | 18.3 | **0** |
| Steele & Henderson / Edwards [2] | d = 1 (0.25-2) (g C m-3)^-1 d-1 | 5.3 (1.3-10.6) [13.2 (3.3-26.4)] | 0 | **1** |
| Fennel 2006 to Lima & Doney 2004 [2] | 0.025-0.25 (mmol N m-3)^-1 d-1 | 1.7-16.6 [4.2-41.5] | model-specific | 1 if pure quadratic |
| EMPOWER / Yool [F] | 0.02 d-1 + 0.34 (mmol N m-3)^-1 d-1 | 22.6 [56.5] | 7.3 | **0.76 [0.89]** |

- **Cross-checks:**
  - EMPOWER at H = 50 m gives a total of 29.9 /yr. Hirst & Kiorboe at 9.3 C give 23.6 /yr total.
  - Edwards' Z* = 0.075 g C m-3 at d = 1 gives 0.075 d-1 = 27 /yr.
  - Fasham's linear 18.3 /yr is close to Mareld's M1 of 19.56 /yr.
  - In NPZ models the closure stands for **all** predation, fish included. Mareld's M1 excludes
    fish, so a closure coefficient cannot be copied over unchanged.
- **Hyperbolic form.** Per-capita a B/(K + B) gives gamma = K/(K + B_ref), between 0 and 1. K for
  Fasham 1993 and Omta et al. 2022 was not retrieved.

### Inferences [I]
- The usual case for a quadratic closure combines three arguments:
  1. unresolved predators track their prey;
  2. it suppresses spurious oscillations;
  3. it is more robust under forcing scenarios.
- Argument 2 is contested (Edwards & Yool). By Gentleman & Neuheimer's criterion, stability
  depends on the grazing clearance rate. Mareld's zooplankton grazes with a half-saturation of
  20 t/km2 on a phytoplankton stock of 3-5 t/cell. Its clearance is therefore close to the
  maximum, a nearly type-I response, which is the stabilising regime. Mareld does not need a
  closure for stability on these grounds.
  - Approximate: the zooplankton energy reserve and the hunger gate also shape the response.
- Arguments 1 and 3 are about what M1 represents; they are taken up in Q3 and Q5.
- NPZ elasticities of 0.75-1 stem from closures that carry all predation and stand in for missing
  food-web structure. Mareld resolves fish predation and food limitation, so the closure share
  should be smaller (Mitra 2009, Fulton et al. 2003).

---

## Q3. Predator responses: do per-capita copepod losses to invertebrate predators rise or fall with copepod density?

### Takeaway
- **Functional response.** For a fixed predator stock, the per-capita copepod mortality is
  **flat or slightly falling** with copepod density at field concentrations.
  - In the Gullmar fjord (Skagerrak), *Sagitta setosa* ingestion rises almost linearly up to ~50
    copepods L-1 and saturates only above that. Field densities there reach ~6 L-1.
  - *Mnemiopsis leidyi* shows a type I response up to 50 L-1.
  - *Aurelia aurita* clearance falls only 30-50 % over three orders of magnitude of prey density.
    Its predation rate rises with prey abundance, and field medusae are food-limited.
  - Functional-response elasticity of per-capita mortality: ~0 to -0.1.
- **Numerical response.** This is what makes per-capita mortality rise with density.
  - Food-limited medusae grow faster at higher prey density.
  - Carnivorous zooplankton have generation times of weeks to months.
  - But North Sea jellyfish abundance varies with climate (NAO, Atlantic inflow) more than with
    copepods.
  - Across ecosystems, predator biomass scales as prey^0.75 (Hatton et al. 2015).

### Cited Findings
- **Vergara-Soto et al. 2010, Gullmar fjord (Sweden), natural copepod assemblage.**
  - *S. setosa* is type II (Ivlev), with "a nearly linear increase in consumption rate for low food
    levels (up to 50 copepods L-1), and a saturation for higher levels". Imax 3.96 copepods
    pred-1 h-1; 0.062 ug DW ug DW-1 h-1.
  - *M. leidyi*: "linear functional response (Type I)", "no saturation" up to 50 L-1; ~7 copepods
    ind-1 h-1.
  - Field maximum in the fjord (Tonnesson & Tiselius 2005) ~6,100 copepods m-3, i.e. "about 6 cop
    L-1". Ingestion there is ~0.2 copepods ind-1 h-1, a ~12 % daily ration.
  - **[F]** [Rev. Biol. Mar. Oceanogr. 45:35-42, doi:10.4067/S0718-19572010000100003](https://doi.org/10.4067/S0718-19572010000100003)
- **Bamstedt 1990, *Aurelia aurita*.** "Predation rate and daily ration were positively correlated
  with prey abundance"; "natural populations are usually food limited"; "exploitation of patches of
  prey in high abundance is an important component". **[A]** [J. Plankton Res. 12:215-229, doi:10.1093/plankt/12.1.215](https://doi.org/10.1093/plankt/12.1.215)
- **Olesen, Frandsen & Riisgard 1994, Kertinge Nor (Denmark).**
  - Ephyra clearance on rotifers "decreased 30 to 50 % with increasing prey concentration" over
    7-13,000 ind L-1.
  - Field zooplankton was ~5 ug C L-1, and *A. aurita* "was food limited".
  - **[A]** [MEPS 105:9-18, doi:10.3354/meps105009](https://doi.org/10.3354/meps105009)
- **Olesen 1995, same cove.** Population clearance was < 0.1 volumes d-1 in spring and 3.5 d-1 in
  late summer. Medusae "appear to control zooplankton" in summer. **[A]** [MEPS 124:63-72, doi:10.3354/meps124063](https://doi.org/10.3354/meps124063)
- **Hansson et al. 2005, Limfjorden.** Jellyfish clearance potential on copepods was low in spring.
  From May, *A. aurita* dominated predation. **[A]** [MEPS 304:117-131, doi:10.3354/meps304117](https://doi.org/10.3354/meps304117)
- **Lindahl & Hernroth 1983, Gullmar fjord.** Very low zooplankton biomass coincided with high
  abundance of scyphomedusae, "which preyed on zooplankton during the summer". **[A]** [MEPS 10:119-126, doi:10.3354/meps010119](https://doi.org/10.3354/meps010119)
- **Attrill, Wright & Edwards 2007, CPR since 1958.** North Sea jellyfish frequency is "positively
  related to the North Atlantic Oscillation (NAO) and Atlantic inflow". **[A]** [L&O 52:480-485, doi:10.4319/lo.2007.52.1.0480](https://doi.org/10.4319/lo.2007.52.1.0480)
- **Lynam et al. 2005.** Jellyfish abundance and herring spawning success "increase under similar
  environmental conditions". Climate mediates the jellyfish effect. **[A]** [MEPS 298:157-167, doi:10.3354/meps298157](https://doi.org/10.3354/meps298157)
- **Majaneva et al. 2013, *Mertensia ovum*.**
  - "A nonlinear functional response": 1 prey pred-1 h-1 at low and 6 at high Calanus density.
  - 1.4 % d-1 of the Calanus population over the water column, higher within aggregations; the
    abstract is truncated.
  - **[A]** [MEPS 476:87-100, doi:10.3354/meps10143](https://doi.org/10.3354/meps10143)
- **Ecopath (sibling note).** In the North Sea Ecopath diet matrix, carnivorous zooplankton take
  ~45 % and fish larvae ~7.5 % of copepod consumption. Fish take ~46 %. **[2]** via
  `research_notes/Zooplankton energibudget/copepod_energetics.md` Q6;
  [Cefas Tech. Rep. 142](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

### Calculations [C]
- **Saturation elasticity.** For a Holling II predator stock P, per-capita prey mortality is
  a P / (1 + a h B), with elasticity -a h B / (1 + a h B) = -sigma.
  - Olesen et al. 1994: clearance x0.5-0.7 over ln(13000/7) = 7.5 ln units, so
    sigma = 0.05-0.09 per ln unit.
  - *S. setosa*: near-linear below ~50 L-1 at field ≤ 6 L-1, so sigma is small, ~0.1 or less.
  - *M. leidyi*: sigma ~ 0.
- **Mareld's B_ref in numbers.** Assumptions:
  - small copepods 2/3 of B_ref = 6.5 g WW m-2 at ~5 ug DW = 27 ug WW per individual (assumed
    typical small-copepod copepodite/adult weight, not sourced here), giving 2.4 x 10^5 m-2;
  - Calanus (3.2 g WW m-2 at ~500 ug WW) adds 6.4 x 10^3 m-2.

  Over 50 m this is ~5 copepods L-1 (over 20 m, ~12 L-1). That is the field density range
  above, a factor 4-10 below *S. setosa* saturation.

### Inferences [I]
- On timescales shorter than the predators' growth response (days to weeks, i.e. tens to hundreds
  of ticks), unresolved invertebrate predation gives gamma ~ 0 to -0.1, not > 0.
- A positive gamma needs predator biomass to track copepod biomass. Within a season this partly
  happens (food-limited medusae, chaetognath and carnivorous-copepod generations), with a lag of
  weeks. Between years, jellyfish track climate.
- The cross-system exponent 0.75 (Hatton et al.) is a long-run, among-ecosystem relation. It is an
  upper reference for the slow, equilibrium numerical response, not a within-season one.
- Aggregation of predators on prey patches (Bamstedt 1990; Majaneva et al. 2013) can give local,
  per-cell DD even with no stock-level numerical response. See Q4.

---

## Q4. Temporal and spatial scale

### Takeaway
- **Timescales.** Generations are weeks for small neritic copepods at Skagerrak summer temperatures
  and months for Calanus; the spring generation at Station M took about 60 days to develop. The
  invertebrate predators respond on similar or slower timescales.
- **What a stock-level gamma means.** In a seasonally cycling pool, an instantaneous stock-level
  gamma mostly encodes how mortality co-varies with biomass through the year. Much of that
  co-variation is temperature: Hirst & Kiorboe's Q10 ~2 gives a factor 1.87 between the coldest
  and warmest months of the model layer. Plourde, Pepin & Head 2009 find temperature the largest
  mortality signal.
- **Per-cell application.** If DD is applied per cell to patchy biomass, Jensen's inequality raises
  the biomass-weighted mean M1 unless it is re-anchored.

### Cited Findings
- Ohman & Hirche 2001: the spring generation at Station M (days 81-160) develops as a "dark
  diagonal band" over ~2 months. Eggs hatch in 2.16 d at 6.4 C. **[F]** (link in Q1)
- Hirst & Kiorboe 2002: Q10 = 2.0 for mortality. Their indirect estimates assume steady state, so
  they are long-period, large-area averages: "must be true for field rates if averaged over
  sufficiently long periods and large areas". **[F]** (link in Q1)
- Plourde, Pepin & Head 2009: temperature is the largest signal; DD (C6 females) dominates egg-C1
  mortality only in the growth season. **[A]** (link in Q1)
- Kattegat copepod biomass is unimodal with a June-July peak; production comes in bursts.
  **[2]** via `plankton.md`, Kiorboe & Nielsen 1994,
  [doi:10.4319/lo.1994.39.3.0493](https://doi.org/10.4319/lo.1994.39.3.0493)

### Calculations [C]
- **Temperature.** Model layer `mean_0_50m`: annual mean 9.31 C, minimum 5.2 C, maximum 13.8 C.
  Hirst & Kiorboe broadcast post-hatch total mortality:
  - 17.5 /yr at 5.2 C, 23.6 /yr at 9.3 C, 32.6 /yr at 13.8 C (factor 1.87);
  - laboratory non-predation: 7.0 / 9.0 / 11.9 /yr.
- **Temperature mimicking density.** Suppose copepod biomass varies ~10x over the year (an
  assumption; seasonal range not verified for the Skagerrak). Attributing the 1.87x temperature
  effect to density instead would give an apparent gamma = ln 1.87 / ln 10 = **0.27**.
- **Jensen factor.** Let per-cell biomass be lognormal with coefficient of variation c, and apply
  M1(B_cell) = M1_ref (B_cell/B_ref)^gamma with B_ref equal to the grid-mean density. The
  biomass-weighted mean M1 is then raised by (1 + c^2)^(gamma (1 + gamma)/2):

| gamma | c = 0.5 | c = 1 | c = 2 |
|---|---|---|---|
| 0.25 | 1.04 | 1.11 | 1.29 |
| 0.5 | 1.09 | 1.30 | 1.83 |
| 1 | 1.25 | 2.00 | 5.00 |

### Inferences [I]
- **Interpretation.** A stock-level gamma fitted to seasonal data is not separable from
  temperature-driven mortality. Mareld already runs a temperature calendar for this group.
  - A Q10 on M1 would capture part of the "density dependence" in field mortality.
  - Adding both a Q10 on M1 and gamma ~ 0.3 would roughly double-count the seasonal signal.
  - Not implemented; this is a design note.
- **Tick vs response time.** The predator numerical response lags by weeks, but a per-tick M1(B)
  responds within 6 h. An instantaneous DD is therefore stronger and faster damping than the
  biology.
  - If DD is added, apply it to a smoothed density, e.g. an exponential moving average over
    ~2-4 weeks (120-240 ticks), and/or at stock or regional level.
- **Per-cell application.**
  - Per-cell DD is justified only for aggregative (local) predator responses.
  - Interference w is 0 for zooplankton, so a per-cell M1(B) would be the group's only
    intraspecific self-limitation apart from food.
  - It must be re-anchored so that the biomass-weighted mean stays at M1_ref: divide M1_ref by
    the realised Jensen factor, or define B_ref per cell from the reference spatial
    distribution.

---

## Q5. Conversion to gamma and beta_M for the Mareld pool

### Takeaway
The cleanest literature-anchored decomposition of M1 has two parts:

- **A density-independent floor M0** (non-predation, about 6-9 /yr at 9.3 C);
- **Unresolved invertebrate predation**, 10.6-13.6 /yr. Its elasticity equals predator biomass's
  numerical-response elasticity k minus the saturation term sigma.

This gives gamma = (1 - M0/M1) (k - sigma):

- at most **~0.54-0.69** (k = 1, the quadratic-closure assumption);
- ~0.37-0.47 with Hatton's k = 0.75;
- **~0.23-0.30** for k = 0.5;
- slightly negative for k = 0 (a fixed or climate-driven predator stock).

Egg cannibalism adds at most ~0.04.

### Calculations [C]
- **Floor M0.** At the model's annual mean temperature (9.3 C):
  - Hirst & Kiorboe laboratory longevity: 0.019 e^(0.061 (9.31 - 5)) d-1 = 9.0 /yr.
  - "Predation 2/3-3/4 of total" route: (1/4 to 1/3) x 23.6 = 5.9-7.9 /yr.
  - Mesocosm near-optimal mortality (Hygum et al. 2000 [2]): 0.011-0.012 d-1 = 4.0-4.4 /yr, a
    physiological lower bound.
  - Starvation is explicit in Mareld (energy reserve), so it does not belong in M0.
  - Take **M0 = 6-9 /yr** (central 8).
- **Predation share of M1.** s_p = 1 - M0/M1 = 1 - (6 to 9)/19.56 = **0.54-0.69** (0.59 at
  M0 = 8).
- **gamma = s_p (k - sigma)**, with sigma = 0.07 (Q3):

| numerical response k | meaning | gamma (M0 = 9 / 8 / 6) |
|---|---|---|
| 1 (sigma = 0) | predator biomass proportional to copepods = quadratic closure | 0.54 / 0.59 / 0.69 |
| 0.75 | Hatton et al. cross-system scaling | 0.37 / 0.40 / 0.47 |
| 0.5 | partial within-season tracking | 0.23 / 0.25 / 0.30 |
| 0.25 | weak, lagged tracking | 0.10 / 0.11 / 0.12 |
| 0 | climate-driven or fixed predators, saturating | -0.04 / -0.04 / -0.05 |

- **Egg cannibalism term (Q1):** + 0.007 to 0.04.
- **Central value: gamma = 0.25** (M0 = 8, k = 0.5). The forms below follow from it, with
  M1_ref = 19.56 /yr = 0.0134 /tick and B_ref = 9.72 t/km2.
  - beta_M = 0.25 x 19.56 = **4.9 /yr** per unit ln B (0.00335 per tick).
  - **Power form:** M1 = 19.56 (B/B_ref)^0.25. It gives:
    - 11.0 /yr at 0.1 B_ref;
    - 16.5 /yr at 0.5 B_ref;
    - 17.6 /yr at 0.66 B_ref (the current policy equilibrium);
    - 23.3 /yr at 2 B_ref.

    It has no floor.
  - **Floor + power form (maps onto the decomposition):**
    M1 = M0 + (M1_ref - M0)(B/B_ref)^kappa, with M0 = 8 /yr and kappa = 0.25/0.591 = **0.42**.
    As B tends to 0, M1 tends to 8 /yr.
  - **Linear form:** M1 = M0' + k' B, with M0' = (1 - 0.25) x 19.56 = **14.7 /yr**
    (0.01005/tick) and k' = 4.89/9.72 = **0.503 /yr per t/km2** (3.45e-4 per tick per t/km2).
    - M0' exceeds the literature floor of 6-9 because, with k < 1, part of the predation is
      itself density-independent.
- **Range:** gamma 0-0.5, so beta_M = 0-9.8 /yr. A hard upper bound is ~0.69 (full proportional
  tracking with the lowest floor).

### Inferences [I]
- **Equilibrium shift.** Anchoring at B_ref leaves the growth budget unchanged at B_ref. With the
  policy equilibrium near 0.66 B_ref, gamma = 0.25 lowers the realised M1 there by ~2 /yr: 17.6 vs
  19.56 (power form), 17.9 (linear form). The equilibrium would shift upward.
- **Phytoplankton.** That shift would feed back on phytoplankton grazing; quantifying it would
  need a model run.
- **Explicit fish predation.** Pelagic fish eat zooplankton through Holling II with interference,
  so the fish part contributes its own (negative, saturating) elasticity. It also has a dynamic
  numerical response through fish growth. That part of the regulation is already resolved and
  must not be counted again in M1.

---

## Q6. Recommendation and the case for gamma = 0

### Takeaway
- **Recommended:** gamma = **0.25**, plausible range **0-0.5**, with a density-independent floor
  of ~8 /yr. Suggested form: M1 = 8 + 11.56 (B/B_ref)^0.42 per year, at stock or regional scale
  on a smoothed density.
- **gamma = 0 is fully defensible** at the pool level and is the better choice if the per-tick,
  per-cell application cannot be smoothed or re-anchored.
- **Upper limit:** gamma ≥ 0.75 (NPZ closures) is not supported for Mareld, because those
  closures stand in for all predation and for missing food-web structure.

### The literature case for gamma = 0 at the pool level [I, from the findings above]
1. **Measured DD is on eggs only** (Calanus; conditional on high female density). At the
   biomass-pool level it is worth ≤ 0.04, because ~98 % of eggs already die at B_ref. Egg
   mortality is not part of the post-hatch total that M1 was calibrated from. Sac spawners
   (Pseudocalanus, Oithona) show none (Q1).
2. **No study shows DD copepodite or adult mortality.** The dominant global pattern is temperature
   and body size (Hirst & Kiorboe 2002; Plourde, Pepin & Head 2009).
3. **The predators behind M1 do not saturate at field densities.** Their functional responses are
   linear to weakly saturating (Gullmar *Sagitta*, *Mnemiopsis*, *Aurelia*), so per-capita
   mortality from a given predator stock is flat or falling (Q3).
4. **Predator biomass does not track copepods closely.** Jellyfish abundance tracks climate (NAO,
   inflow) more than copepods (Attrill et al. 2007; Lynam et al. 2005).
5. **Mareld already resolves the main regulators:**
   - explicit food limitation (half-saturation 20 t/km2);
   - explicit fish predation with a dynamic numerical response;
   - explicit starvation.

   The NPZ rationale for a closure (stand-in for unresolved regulation, damping oscillations) is
   weaker: Fulton et al. 2003, Mitra 2009, Edwards & Yool 2000, Gentleman & Neuheimer 2008.
6. **The seasonal mortality-biomass co-variation in field data can be temperature** (Q10 2 gives
   an apparent gamma of ~0.27 for a 10x seasonal biomass range). Mareld's temperature calendar
   could carry it explicitly instead (Q4).

### The case for gamma > 0 [I]
- About 55-70 % of M1 is invertebrate predation, and those predators (chaetognaths, carnivorous
  copepods, ctenophores, food-limited medusae) do grow on copepods. Over weeks to a season their
  biomass partly follows copepod biomass.
- Among ecosystems, predator biomass scales as prey^0.75.
- Local aggregation on prey patches (Bamstedt 1990) gives per-cell DD.
- A modest gamma also gives the robustness under perturbation and forcing scenarios that Fulton et
  al. and Hill Cruz et al. 2021 attribute to DD closures.

### Implementation cautions [I]
- **Anchor.** Anchor at B_ref = 9.72 t/km2 so that the derived growth budget is unchanged at B_ref.
- **Smoothing and scale.** Apply the DD to a smoothed (weeks) and/or regional density, not to the
  instantaneous per-cell biomass. If it is per cell, re-anchor for the Jensen factor:
  +11 % at c = 1 for gamma = 0.25, +30 % for gamma = 0.5.
- **Temperature.** Do not combine gamma ~ 0.3 with a Q10 on M1 without reconciling the two.
  Either alone captures roughly the same seasonal signal.
- **Checkpoints.** Any change to M1 changes the derived growth_rate path and the zooplankton
  equilibrium, so policies trained before it would need retraining.

## Open gaps
- No Skagerrak or Kattegat in situ copepod mortality series, so gamma cannot be fitted locally.
- Kiorboe 2008 (book), Anderson, Gentleman & Sinha 2010, Edwards & Brindley 1999, and the
  fulltexts of Steele & Henderson 1992 and Mitra 2009 were not read.
- The axis units in Ohman et al. 2004 Fig. 4 were not legible, but the elasticities computed at
  the centroid do not depend on them.
- The C/DW ratio (0.40) and the individual weights in Q3 are assumptions, not sourced values. They
  affect only the unit conversions of Q2 and Q3, not the recommended gamma.
- The seasonal biomass range (10x) used in Q4 is assumed.
