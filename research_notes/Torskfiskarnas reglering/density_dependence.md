# What holds a gadoid stock near its level besides food: stock-recruit compensation, carrying capacity, cannibalism, and how to put it in an age-less pool

Scope note:
- This note builds on four earlier notes and does not redo them:
  - `research_notes/Täthetsberoende och interferens hos torskfiskar/regulation_and_models.md` (cannibalism, where density
    dependence acts, how Ecosim and Atlantis implement it);
  - `.../gadoid_density_dependent_growth.md` (density-dependent growth: weak, a few % of intake per doubling);
  - `.../model_context.md` (what w can and cannot do in the simulator);
  - `research_notes/Täthetsberoendets styrka per grupp/gadoids.md` (the elasticity beta_M of a density-dependent M1:
    recommended 0.12 /yr, range 0.04-0.25).
  - `research_notes/Restdödlighet M0 hos torskfiskar/` was also read, for the M1 = 0.20 + 0.214 split and the SMS M
    conventions.
- What is new here:
  1. The arithmetic that compares the size of each brake with the model's surplus of r_max - F ≈ 0.39 /yr.
  2. The point that the library's r_max is a Schaefer (CMSY) parameter whose K was dropped.
  3. A concrete, literature-anchored form for a carrying-capacity term.
- Labels:
  - **Literature**: sourced, with fetch status, carried over from the cited notes unless marked otherwise.
  - **Computed**: my arithmetic.
  - **Inference**: my reasoning.

## Takeaway

- **The model's surplus is a missing K, not a missing mortality.**
  - The library's `r_max` 0.511 is the CMSY (Schaefer) r. In CMSY, r is the net per-biomass growth rate as B -> 0, and
    density dependence is the (1 - B/K) term of the same model.
  - The model kept r and dropped K. Its derived g then pays r_max at ideal food at any density, so the only brake left is
    food. The result is the predator-prey overshoot seen in resume 184.13.
- **Real gadoid stocks are regulated mainly before recruitment.** Candidate brakes, from best to least supported:
  1. Juvenile density-dependent survival and stock-recruit compensation. Density-dependent recruitment was found in
     68 % of 80 NE Atlantic stocks (Rindorf et al. 2022). In 8 of 11 Skagerrak coastal cod populations, density-dependent
     mortality of the 0-group has a = -0.25 to -0.50 (Fromentin et al. 2001).
  2. Cannibalism and intra-gadoid predation. This acts on ages 0-2. In SMS it is 0.016 + 0.0875 B /yr, i.e. about 0.08 /yr
     at 0.69 t/km2.
  3. Density-dependent growth. This is weak: 5-15 % of production per doubling, often undetectable in North Sea cod and
     haddock. It is already in the model through intake.
  4. Density-dependent adult M: no support, and depensatory if anything.
- **How strong at today's state?**
  - If today's pool is near its fished equilibrium, the total density-dependent brake at B_ref must by definition equal
    r_max - F, about 0.39 /yr (Schaefer).
  - A Beverton-Holt pool gives a slope beta of 0.14-0.24 /yr per ln B (`gadoids.md`, section 3).
  - The cannibalism term alone is about 0.06-0.08 /yr, and the `gadoids.md` recommendation (beta_M 0.12 on M1) about
    0.12 /yr.
  - So **neither cannibalism nor the earlier recommended M1 elasticity can by itself cancel a 0.39 /yr surplus.** They are
    the right shape, but their size was set for a different job: making M1 respond to density around a level whose net
    was assumed to be about 0.
- **A literature-supported representation exists: the Schaefer / logistic term that CMSY itself assumes.**
  - It is the standard closure in surplus-production assessments (CMSY, SPiCT). Atlantis and Ecosim implement the same
    idea as Beverton-Holt recruitment.
  - For an age-less pool, add a loss r_max·(B/K) on the regional (not per-cell) gadoid biomass.
  - Calibrate K from stationarity at today's F: K = B_ref / (1 - F/r_max) ≈ 0.90 t/km2 (range 0.8-1.4).
  - Do not count this loss in the derived g.
  - See "Proposed form" below for the details and caveats.

## Cited Findings (literature; status as recorded in the cited notes unless stated)

**Stock-recruit compensation**
- Rindorf A, van Deurs M, Howell D, Andonegi E, Berger A, Bogstad B, et al. 2022. "Strength and consistency of density
  dependence in marine fish productivity". *Fish and Fisheries* 23:812-828, doi
  [10.1111/faf.12650](https://doi.org/10.1111/faf.12650), [Archimer PDF](https://archimer.ifremer.fr/doc/00753/86468/91824.pdf).
  Title and authors were checked against Crossref.
  - Density-dependent recruitment in 68 % of 80 NE Atlantic stocks, 78 % excluding pelagic stocks with a trend.
  - Median steepness (R at 0.2 x max observed SSB / max R) 0.79.
  - Significant overcompensation in 38 %, more often in demersal stocks (0.56).
  - (Fetched full text via Archimer, `gadoids.md` 1.2.)
- Myers, Mertz & Fowlow 1997, *Fishery Bulletin* 95:762-772, "Maximum population growth rates and recovery times for
  Atlantic cod, *Gadus morhua*".
  - Lifetime slope at the origin (spawners per spawner) 3.8 (Kattegat), 11.2 (Skagerrak), 9.0 (North Sea).
  - r_m 0.53, 0.82 and 0.56 /yr.
  - "There is no evidence that mortality increases at low population size, i.e. depensation".
  - [PDF](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1997/954/myers.pdf) (Fetched full text, `gadoids.md`.)
- Myers 2001, *ICES J. Mar. Sci.* 58:937-951, "Stock and recruitment: generalizations about maximum reproductive rate,
  density dependence, and variability using meta-analytic approaches".
  - "between 1 and 7 replacement spawners are produced per spawner per year at low population size".
  - "a 20-fold variation in carrying capacity per unit area for cod".
  - Recruitment correlations at about 500 km.
  - [doi 10.1006/jmsc.2001.1109](https://doi.org/10.1006/jmsc.2001.1109) (Abstract.)
- Mackinson & Daskalov 2007, Cefas Tech. Rep. 142. North Sea stock-recruit curves "tend to 'flat' over a wide range of
  spawning stock sizes", implying "strong compensatory increases in juvenile survival rate as spawning stock declines".
  Haddock, whiting and saithe compensate more than cod.
  [PDF](https://www.cefas.co.uk/publications/techrep/tech142.pdf) (Fetched full text, `regulation_and_models.md`.)
- FishLife v2 (Thorson 2020, *Fish and Fisheries* 21:237-251, doi [10.1111/faf.12427](https://doi.org/10.1111/faf.12427)),
  Beverton-Holt steepness h: cod 0.79, haddock 0.66, whiting 0.71, saithe 0.79, hake 0.56, Gadidae 0.73. (Data extract,
  `gadoids.md`.)

**Juvenile density-dependent mortality in the Skagerrak (the model's own region)**
- Fromentin, Myers, Bjørnstad, Stenseth, Gjøsæter & Christie 2001, *Ecology* 82:567-579, "Effects of density-dependent
  and stochastic processes on the regulation of cod populations".
  - Direct density-dependent mortality a = -0.25 to -0.50 in 8 of 11 Norwegian Skagerrak coastal areas, in a Gompertz
    (log-linear) form.
  - Delayed density dependence -0.49 to -0.75 in the north-eastern fjords.
  - Mechanisms: shelter and food competition in seagrass and kelp, and 1-group cannibalism on the 0-group.
  - [doi](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2), [Archimer PDF](https://archimer.ifremer.fr/doc/00000/712/378.pdf) (Fetched full text, `gadoids.md` 1.1.)
- Bjørnstad et al. 1999, *Ecology* 80:1278-1288: survival from the 0- to the 1-group is "approximately log-linear" in
  abundance. (Abstract.)
- Myers & Cadigan 1993, *CJFAS* 50:1576-1590: "strong evidence of density-dependent mortality within cohorts during the
  juvenile stage for cod, plaice, sole, and whiting", related to "the log of juvenile abundance". (Abstract.)

**Cannibalism and intra-gadoid predation**
- SMS 2023 key run, 1974-2022, computed in `regulation_and_models.md` and `gadoids.md`.
  - Predation of cod, whiting, haddock and saithe on each other per pooled biomass: mean 0.233 /yr (0.10-0.45) at
    1.45-5.4 t/km2.
  - Linear fit 0.016 + 0.0875 B. Cod on cod is 0.073 /yr.
  - Acts on ages 0-2. It is structurally proportional to predator abundance in SMS, so it is not independent proof of k·B.
- Yaragina et al. 2009, Neuenfeldt & Köster 2000, Anderson & Gregory 2000: cannibalism is a major and variable
  juvenile loss that falls when alternative prey is plentiful. (Abstracts and snippets, `regulation_and_models.md`.)

**Density-dependent growth**
- Lorenzen & Enberg 2002, *Proc. R. Soc. B* 269:49-54: significant in 9 of 16 populations, and "alone is sufficient" to
  generate the among-population pattern. (Abstract.)
- In gadoids it is weak where detected: per-capita intake falls about 3-6 % per doubling and production about 5-15 %.
  It is often undetectable in North Sea cod and haddock. (`gadoid_density_dependent_growth.md`.)

**How ecosystem models close it**
- Atlantis: Beverton-Holt recruitment is "the most common function applied for fish species". The optional quadratic
  mortality "should be low (or zero)" except for top predators. CSIRO Atlantis User Guide Part I, 2025.
  [PDF](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf) (Fetched full text.)
- Ecosim: split juvenile and adult pools tuned to give "'emergent' Beverton-Holt type stock-recruitment relationships",
  plus foraging-arena vulnerability. (Mackinson & Daskalov 2007, fetched full text.)
- SMS: fixed M1 0.2, with M2 driven by predator abundance. Recruitment is fitted with hockey-stick or Ricker S-R curves.
- CMSY: Schaefer dB/dt = r B (1 - B/K) - C. "Prior r" ranges by resilience, and r = 2 Fmsy.
  Froese et al. 2017, *Fish and Fisheries* 18:506-526, "Estimating fisheries reference points from catch and resilience",
  doi [10.1111/faf.12190](https://doi.org/10.1111/faf.12190); [author PDF](https://www.fishbase.de/rfroese/CMSY_faf_12190_Rev.pdf)
  (Fetched full text, `Litteraturbaserad growth rate per FG/fish.md`.)

## Computed: how large is each brake against the model's surplus?

Model surplus at today's state, with ideal food:
- r_max - F = 0.511 - 0.12 = **0.39 /yr**.
- About 0.06 /yr more, because seals and porpoises take less than their budgeted M2 (resume 184.13: realised 0.11-0.12
  against 0.176 budgeted).
- The realised year-1 net in resume 184.13 was +0.35 /yr.

| Brake | Value at B_ref = 0.69 t/km2 (/yr) | At 1.4 t/km2 | Share of the 0.39 surplus at B_ref | Source |
|---|---|---|---|---|
| Restore cannibalism as SMS 0.016 + 0.0875 B (per cell) | 0.076 | 0.14 | 19 % | SMS fit |
| `gadoids.md` recommended M1 elasticity, beta 0.12 (power form, M1 renormalised to 0.415 at B_ref) | 0 extra at B_ref; +0.08 at 2x B_ref | 0.08 | 0 % at B_ref (it only resists growth away from B_ref) | `gadoids.md` 6 |
| Beverton-Holt compensation equivalent, beta_M 0.14-0.24 per ln B (h 0.49-0.79, T 2.5-5 yr) | 0 extra at B_ref if the floor is renormalised; +0.10-0.17 per doubling | | 25-43 % per doubling | `gadoids.md` 3 |
| Raise interference w to 2-3 | 0.03-0.05 (production) | | 8-13 % | `model_context.md` 4 |
| Schaefer term r_max B/K with K = 0.90 | **0.39** | 0.79 | 100 % (by construction) | this note |
| Schaefer term with K from the IBTS density (K = 3.5) | 0.10 | 0.20 | 26 % | this note; K uncertain 4x |

Notes on the arithmetic:
- **Schaefer equilibrium.** r(1 - B*/K) = F, so B*/K = 1 - F/r.
  - r 0.511, F 0.12: B*/K = 0.765, K = 0.69/0.765 = 0.90 t/km2.
  - F 0.20: B*/K = 0.61, K = 1.14.
  - F 0.26 (the local catch/biomass of `local_fishery.md`): B*/K = 0.49, K = 1.41.
  - r 0.40, F 0.12: B*/K = 0.70, K = 0.99.
  - With the survey-scaled density 2.7 t/km2 (`ices_gadoid_f.md` caveat) in place of 0.69, every K is 3.9x larger.
- **Historical cross-check.**
  - In 2010-2019 the pool was near stationary (`stock_trends_and_mortality.md`).
  - Its biomass-weighted F was about 0.23 with today's survey composition, or 0.29-0.30 with the 2005-2019 composition
    (computed in `stock_trends_and_mortality.md`, Q2).
  - For that period, Schaefer with r 0.51 gives B/K = 0.55 (F 0.23) or 0.43 (F 0.29).
  - The present F of about 0.11-0.12 is low partly because haddock F fell and partly because haddock now dominates the
    pool.
  - Lowering F to 0.12 moves the equilibrium to B/K = 0.76: a rise of about 40-75 %.
  - The initial rate would be r(1 - B/K) - 0.12 = **+0.11 to +0.17 /yr**, slowing towards 0.
  - So the surplus-production view predicts a real but decelerating rise after the 2021-2023 F cut. The observed 2020-2023
    doubling is larger and came from two year classes; the -3.4 %/yr since 2023 is the pulse ageing.
  - Inference: the data fit "Schaefer with K, near equilibrium" better than "no K". Neither the data nor Schaefer support
    +35 %/yr sustained for four years.
- **Why the M1-elasticity recommendation does not cancel the surplus.**
  - `gadoids.md` renormalises M1(B_ref) back to 0.415, so at B_ref the loss is unchanged and the surplus stays 0.39.
  - The elasticity only adds about 0.08 /yr per doubling. From 0.69 the pool would still grow until the extra loss equals
    0.39: (B/B_ref)^gamma = 1 + 0.39/0.415, so with gamma 0.3 the ratio is 9.1, i.e. about 6 t/km2.
  - Food would stop it first, as in resume 184.13.
  - This is not a flaw of that note. Its beta describes the slope of density dependence around an equilibrium. The
    *level* of the equilibrium is what K sets.

## Proposed form (Inference, built from the literature above)

**Option C1: logistic loss on regional biomass (the CMSY assumption made explicit).**
- Loss per tick on every gadoid cell: `L = (r_max / 1460) * (B_region / K)`, with B_region the grid-mean gadoid density
  (t/km2). It removes biomass and reserve in the same proportion as M1 does.
- It is NOT counted in the derived g.
  - The budget keeps paying r_max + M1 + M2 at ideal food, as it does now.
  - Net growth at ideal food then becomes r_max(1 - B/K) - F: the Schaefer model.
  - This differs from `gadoids.md` item 4, which counts the cannibalism-like term in g. That is right for a loss that is
    a real mortality (cannibalism). It is wrong for a closure term that *is* the density dependence of r.
- **K = 0.90 t/km2** (3240 t on the grid) from stationarity at F 0.12. Range 0.8-1.0 at F 0.12 depending on the surplus
  basis (block below), and 0.8-1.4 over F 0.12-0.26.
  - It must be re-derived if r_max or F change: K = B_ref/(1 - F/r_max).
  - Per tick at B_ref: 0.511/1460 x 0.69/0.90 = 2.7e-4, about the size of M1 (2.84e-4).

**Which surplus should K cancel? (Computed)**

The K values above assume the brake at B_ref must equal the ideal-food surplus r_max - F = 0.39. Two other bases are
defensible:

| Basis for L(B_ref) | L (/yr) | B_ref/K = L/r_max | K (t/km2), r 0.511, F 0.12 |
|---|---|---|---|
| Ideal-food identity r_max - F | 0.39 | 0.765 | 0.90 |
| Ideal-food identity plus the seal/porpoise M2 shortfall (about 0.06; resume 184.13) | 0.45 | 0.88 | 0.78 |
| The probe's realised year-1 net (+0.35; resume 184.13, food not ideal) | 0.35 | 0.685 | 1.01 |

- So K ≈ 0.8-1.0 at F 0.12, and 0.8-1.4 over F 0.12-0.26.
- The right target is the realised net of a retrained run, not the ideal-food identity. Set K from the identity, then
  check that pooled biomass stays within about ±10 %/yr of B_ref after retraining, and adjust K once.
- Biological reading:
  - Mostly reduced recruitment per unit of spawning biomass: juvenile habitat and shelter limitation, cannibalism, and
    juvenile food competition in coastal nurseries largely outside the grid.
  - That is why it should key on regional, not per-cell, biomass. Recruitment correlates over about 500 km (Myers 2001),
    and nursery density dependence acts at fjord scale, < 20 km (Fromentin et al. 2002).
- Cost: as a pool-wide term it gives the policy no local gradient to learn from. That is correct for a recruitment
  process, which no adult behaviour on the grid can avoid.

**Option C2: Beverton-Holt-shaped (concave) version.**
- `L(B) = (1/T) ln(1 + (alpha_hat - 1) d)` with d = B/K_0, where K_0 is the unfished level (`gadoids.md` section 3,
  derived from Euler-Lotka). With h 0.73 (alpha_hat 10.8) and T 4.7 yr, L rises with B more slowly than the logistic.
- Compensation is stronger at low B and weaker at high B, matching the "flat" stock-recruit curves in Mackinson &
  Daskalov.
- To give L(B_ref) = 0.39 /yr with T 4.7 yr: ln(1 + 9.8 d) = 1.83, so d = 0.53 and K_0 = 1.3 t/km2. It is steeper near
  B_ref than the `gadoids.md` beta (0.18 /yr per ln B).
- Preferable if the model will ever be run far from B_ref, for example a no-fishing scenario. Equivalent near B_ref.

**Option C3: recruitment-limited r (lower r_max only).**
- Setting r_max to the realised net rate at today's state (≈ F = 0.12, so that net = 0) gives stationarity without a K.
- It has no feedback: any perturbation grows or decays without bound, and it removes the low-density resilience that
  the literature does support (r 0.3-0.8 for cod; Myers et al. 1997).
- Inference: not recommended except as a quick test.

**Option D: restore cannibalism (the unchecked pair or a per-cell 0.016 + 0.0875 B).**
- Literature-supported as a real loss, about 0.08 /yr at B_ref.
- It is learnable per cell, but the group-biomass reward cannot learn the pair itself (resume section 160).
- It closes about 20 % of the surplus, and only if it is left out of the derived g. If it is counted in g, as
  `gadoids.md` item 4 proposes and as a checked pair is, the budget pays it back at B_ref and only its slope remains.
- A good complement to C1, not a replacement.
- If both are used and cannibalism is left out of g, K must be re-derived with cannibalism inside the losses:
  r_eff = r_max - 0.076, so K = 0.69/(1 - 0.12/0.435) = 0.95.

## Confidence

- **High:**
  - Gadoid stocks show compensatory stock-recruitment. Several independent meta-analyses agree, and the Skagerrak cod
    data specifically.
  - An age-less pool with only food limitation lacks it.
- **High that the CMSY r_max implies a K:** it is the same model.
- **Moderate:**
  - K should key on regional biomass.
  - The logistic and B-H forms are adequate near B_ref.
- **Low:**
  - The value of K. It hinges on the assumption that today's state is the fished equilibrium, which `stock_trends_and_mortality.md`
    says holds for 2005-2020 but not exactly for 2021-2025.
  - It also hinges on F (0.12 versus 0.22 historical) and on the 4x model-versus-survey biomass gap.
