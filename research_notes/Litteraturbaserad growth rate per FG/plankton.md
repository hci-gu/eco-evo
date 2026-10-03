# Plankton growth rates for the Mareld model: phytoplankton r and mesozooplankton (copepod) P/B

Conventions used throughout:
- Model tick = 6 h, 4 ticks/day, 1460 ticks/year.
- "lin" per-tick = per-day / 4. "exp" per-tick = exp(mu/4) - 1, which is the discrete per-tick
  increment that reproduces continuous exponential growth at rate mu (d-1). The difference matters
  only for phytoplankton-sized rates: at mu = 1.0 d-1, lin = 0.250, exp = 0.284 (+14%); at
  mu = 1.67 d-1, lin = 0.418, exp = 0.519 (+24%). For copepod rates (0.03-0.3 d-1) lin and exp
  differ by < 4%, so per-day / 4 is fine. Annual P/B -> per tick = P/B / 1460.
- "Gross"/"net" for phytoplankton below: lab mu_max = gross of all losses except none (nutrient-
  and light-replete culture growth); dilution-experiment mu = in situ intrinsic growth, gross of
  microzooplankton grazing; 14C primary production = approximately net of autotroph respiration
  but gross of all grazing, sinking and lysis. The model's r should be gross of mesozooplankton
  grazing but net of microzooplankton grazing, sinking, lysis and respiration.
- For copepods, weight-specific growth rate g (somatic growth of juveniles + weight-specific egg
  production of adults) and Ecopath P/B are both "production P" in the sense used by the model
  (biomass production before predation/natural mortality, after respiration).

## 1. Phytoplankton maximum (nutrient/light-replete) growth rate vs temperature

### Takeaway
Laboratory maximum growth at ~11.5 C is 1.2 d-1 (Eppley 1972) to 1.7 d-1 (Bissinger et al. 2008),
i.e. 0.31-0.42 per tick (lin) or 0.36-0.52 per tick (exp). These are upper envelopes for
replete cultures, not in situ rates, and are 2.5-4x higher than the model's current r = 0.125/tick.

### Cited Findings
- Eppley (1972): maximum expected growth rate of marine phytoplankton (laboratory cultures)
  log10 mu = 0.0275 T - 0.070, i.e. mu = 0.851 (1.066)^T, with mu in **doublings per day**
  (multiply by ln 2 = 0.693 to get d-1). Fitted by eye as the upper envelope of laboratory culture
  data (n = 162 according to Bissinger et al. 2008). — [Eppley 1972, Temperature and phytoplankton growth in the sea, Fishery Bulletin 70:1063-1085](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1972/704/eppley.pdf)
- Eppley (1972) also reports an in situ midsummer case where growth was < 0.5 doublings
  of cell carbon per day, i.e. well below the envelope. — [Eppley 1972](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1972/704/eppley.pdf)
- Bissinger et al. (2008): 99th-quantile regression on n = 1,501 growth rates gives
  mu_max = 0.81 e^(0.0631 T) (d-1, T in C); higher than the Eppley curve at all temperatures (the
  Eppley curve falls below the lower 95% CI of the new curve below 19 C); Q10 = 1.88, identical to
  Eppley's; using the Eppley function in a 1-D temperate shelf model underestimates primary
  production by up to 30%. — [Bissinger, Montagnes, Sharples & Atkinson 2008, Limnol. Oceanogr. 53:487-493, doi:10.4319/lo.2008.53.2.0487](https://doi.org/10.4319/lo.2008.53.2.0487) (abstract via [OpenAlex](https://api.openalex.org/works/doi:10.4319/lo.2008.53.2.0487); also [NORA](http://nora.nerc.ac.uk/5372/))
- Kremer et al. (2017): > 4,200 growth-rate measurements; temperature scaling of maximum growth is
  weaker than the Eppley curve: a 10 C increase raises growth by a factor of 1.53 (not 1.88);
  MTE-form activation energy E = 0.30 eV (95% CI 0.233-0.368), Eppley-form exponent b = 0.04738
  per C; growth decreases only weakly with cell mass (exponent -0.054, vs -0.25 predicted by MTE);
  functional groups differ in intercept, with diatoms and green algae growing faster than the
  Bissinger/Eppley curve predicts at low temperatures and all groups slower at high temperature.
  — [Kremer, Thomas & Litchman 2017, Limnol. Oceanogr. 62:1658-1670, doi:10.1002/lno.10523](https://www.dora.lib4ri.ch/eawag/dload/eawag:14331/PDF/view) (open-access PDF; Table 1, Fig. 2)

### Inferences
- Worked values (computed from the equations above):

| T (C) | Eppley mu (d-1) | lin/tick | exp/tick | Bissinger mu (d-1) | lin/tick | exp/tick |
|---|---|---|---|---|---|---|
| 5 | 0.81 | 0.203 | 0.225 | 1.11 | 0.278 | 0.320 |
| 10 | 1.12 | 0.279 | 0.322 | 1.52 | 0.381 | 0.463 |
| 11.5 | 1.23 | 0.308 | 0.360 | 1.67 | 0.418 | 0.519 |
| 15 | 1.54 | 0.385 | 0.469 | 2.09 | 0.522 | 0.685 |
| 17 | 1.75 | 0.437 | 0.548 | 2.37 | 0.592 | 0.808 |

- The project's current derivation (Eppley at 11.5 C = 1.23 d-1, minus ~60% micrograzing =
  0.49 d-1 -> 0.123/tick) reproduces correctly from the Eppley formula. With Bissinger instead,
  the same recipe gives 1.67 x 0.4 = 0.67 d-1 -> 0.167/tick (lin) or 0.18/tick (exp).
- Using a lab mu_max as the logistic r implicitly assumes that phytoplankton in the model area grows
  at the replete maximum whenever B << K. Real in-situ growth is light- and nutrient-limited for
  most of the year in the Kattegat/Skagerrak (summer nutrient depletion, winter light limitation;
  section 3), so mu_max is an upper bound for r, not a central estimate.

### Gaps
- Kremer et al. give group-specific intercepts in a mass-dependent form; I did not convert them to
  a single community mu_max at 11.5 C (needs a representative cell mass), so only the Q10 = 1.53 is
  reported as a usable number.
- No regional (Kattegat/Skagerrak) lab mu_max compilations were found.

## 2. In situ phytoplankton growth and the partition of grazing (micro- vs mesozooplankton)

### Takeaway
Dilution experiments put in situ intrinsic phytoplankton growth in the southern North Sea in spring
at 0.13-0.67 d-1 (0.03-0.17/tick), with microzooplankton grazing often equal to or larger than
growth. Globally microzooplankton remove ~60% (coastal) / ~59% (temperate) of daily primary
production, while mesozooplankton remove a mode of 6% and mean of 22.6% per day; in the
Kattegat/Skagerrak copepods grazed < 4% of phytoplankton standing stock per day.

### Cited Findings
- Calbet & Landry (2004): 788 paired dilution estimates of phytoplankton growth (mu) and
  microzooplankton grazing (m); microzooplankton consume 67% of phytoplankton daily growth
  globally; 60% for coastal and estuarine environments, 70% open ocean; ~59% for
  temperate-subpolar and polar systems, 75% tropical-subtropical; production from dilution
  experiments is a reasonable proxy for 14C production (r = 0.89). — [Calbet & Landry 2004, Limnol. Oceanogr. 49:51-57, doi:10.4319/lo.2004.49.1.0051](https://doi.org/10.4319/lo.2004.49.1.0051) (abstract via [OpenAlex](https://api.openalex.org/works/doi:10.4319/lo.2004.49.1.0051))
- Calbet (2001): mesozooplankton (200-20,000 um) grazing impact on primary production: mode 6%,
  mean 22.6% of PP consumed per day, decreasing exponentially with increasing productivity; global
  estimate ~12% of oceanic PP (5.5 Gt C/yr). Moderately productive communities defined as
  250-1,000 mg C m-2 d-1. — [Calbet 2001, Limnol. Oceanogr. 46:1824-1830, doi:10.4319/lo.2001.46.7.1824](https://doi.org/10.4319/lo.2001.46.7.1824) (abstract via [OpenAlex](https://api.openalex.org/works/doi:10.4319/lo.2001.46.7.1824))
- Stelfox-Widdicombe et al. (2004), Southern Bight of the North Sea, April 1998, 7.1-9.5 C:
  dilution experiments on the < 200 um fraction; phytoplankton specific growth rates 0.13-0.67 d-1
  (highest offshore); microzooplankton grazing mortality 0.27-1.14 d-1 (highest nearshore, and
  exceeding growth there); microzooplankton grazed 15-54% of daily primary production and 24-68%
  of daily < 200 um chlorophyll standing stock. — [Stelfox-Widdicombe, Archer, Burkill & Stefels 2004, J. Sea Res. 51:37-51](https://www.vliz.be/imisdocs/publications/55283.pdf)
- Tiselius (1988), Skagerrak and Kattegat, May-October: copepod community grazing was < 4% of
  phytoplankton standing stock per day but up to 48% of daily primary production; coastal
  (small-copepod) stations grazed more than offshore (Calanus-dominated) stations. — [Tiselius 1988, Ophelia 28:215-230, doi:10.1080/00785326.1988.10430814](https://doi.org/10.1080/00785326.1988.10430814)
- Mackinson & Daskalov (2007), North Sea Ecopath report, citing older work: copepod grazing in the
  central North Sea ~14, 9 and 3% of primary production in May, June and September (Baars & Fransz
  1984); estimates of the percentage of total PP grazed by zooplankton in different North Sea areas
  vary from 35-100% (average 65%) (Fransz & Gieskes 1984); copepod grazing matches PP in summer but
  does not significantly reduce phytoplankton biomass during spring/autumn blooms; 20-35% of PP
  deposits to the sediment during spring in the northern North Sea (Cadee 1985); 20-60% of PP
  enters the microbial food chain (Linley et al. 1983); extracellular release 0-10% open sea,
  1-16% coastal. — [Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142, sections 7 and 10](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Kaas et al. (1991): a late-stage Chrysochromulina polylepis bloom in the Kattegat (May-June 1988),
  light-limited in the pycnocline, had doubling times of 4-23 d, i.e. specific growth ~0.03-0.17 d-1
  (0.008-0.04/tick). — [Kaas, Larsen, Mohlenberg & Richardson 1991, Mar. Ecol. Prog. Ser. 79:151-161, doi:10.3354/meps079151](https://doi.org/10.3354/meps079151)

### Inferences
- In situ intrinsic growth (dilution) of 0.13-0.67 d-1 = 0.03-0.17/tick (lin); net of
  microzooplankton grazing it was often <= 0 nearshore in the North Sea spring study. That the
  model's r = 0.125/tick (0.5 d-1) sits near the top of the in situ *gross* range and above most
  net-of-micrograzing values suggests the current r is on the high side if it is meant as a
  realised in situ rate.
- The "minus 60%" step in the current derivation is applied to the lab mu_max; Calbet & Landry's
  m:mu ratio is defined relative to in situ growth (mu from dilution), not to mu_max. Applying it to
  mu_max mixes a lab ceiling with a field loss fraction; applying it to in situ mu (0.13-0.67 d-1)
  would give 0.05-0.27 d-1 (0.013-0.067/tick).
- Unresolved losses beyond micrograzing (sinking 20-35% of spring PP in the northern North Sea;
  lysis) would lower the net rate further.
- The low mesozooplankton share (Calbet 2001 mode 6%; Tiselius < 4% of stock/day; Baars & Fransz
  3-14% of PP) is consistent with the model keeping the copepod-grazing term explicit and small
  relative to phytoplankton turnover.

### Gaps
- No dilution-experiment growth/grazing rates from the Kattegat or Skagerrak proper were found
  (searches returned only southern North Sea and non-regional studies).
- Kiorboe & Nielsen (1994) reportedly estimate copepod grazing at ~13% of the annual primary
  production (290 g C m-2) at their southern Kattegat station; this figure comes from a
  search-engine summary of a ResearchGate page and could NOT be verified against the full text
  (Wiley/ResearchGate blocked). [Kiorboe & Nielsen 1994, Limnol. Oceanogr. 39:493-507](https://doi.org/10.4319/lo.1994.39.3.0493)

## 3. Annual primary production, phytoplankton biomass and annual P/B (Kattegat/Skagerrak/North Sea; Ecopath)

### Takeaway
14C annual primary production is 190-290 g C m-2 yr-1 in the Kattegat/Skagerrak and the Gullmar
Fjord mouth (Swedish west coast), ~200-250 in the central and southern North Sea. Ecopath P/B
values for phytoplankton span 87.5-286 yr-1 (0.06-0.20/tick); the best-documented regional
anchor (western Baltic, P/B 120 yr-1 on B 2.16 g C m-2, implying ~260 g C m-2 yr-1, consistent with
Kattegat 14C data) gives 0.33 d-1 = 0.082/tick. Note this is gross of all grazing.

### Cited Findings
- Southern Kattegat, fixed station, 1989 (55 visits): total annual primary production ca. 290
  g C m-2; ~19% during the spring bloom, ~30% in subsurface (pycnocline) populations in summer;
  surface phytoplankton nutrient-limited mid-May to ~1 October; summer surface chl < 2 ug/L.
  — [Richardson & Christoffersen 1991, Mar. Ecol. Prog. Ser. 78:217-227, doi:10.3354/meps078217](https://doi.org/10.3354/meps078217)
- Northern Kattegat / southern Skagerrak frontal region, 15 cruises 1984-1993: total annual
  primary production ~190 g C m-2 yr-1; subsurface chl maximum April-October.
  — [Heilmann, Richardson & Aertebjerg 1994, Mar. Ecol. Prog. Ser. 112:213-223, doi:10.3354/meps112213](https://doi.org/10.3354/meps112213)
- Kattegat: primary production increased from < 100 g C m-2 yr-1 (1950s) to ~200 g C m-2 yr-1
  (1984-1993, recalculated with the 1950s method); the increase is seen in the spring bloom and
  summer but not in winter (light-limited). — [Richardson & Heilmann 1995, Ophelia 41:317-328, doi:10.1080/00785236.1995.10422050](https://doi.org/10.1080/00785236.1995.10422050)
- Gullmar Fjord mouth (Swedish Skagerrak coast), in situ 14C, 1985-1996: mean annual primary
  production 241 g C m-2 yr-1, interannual range 180-339. — [Lindahl et al. 1998, ICES J. Mar. Sci. 55:723-729, doi:10.1006/jmsc.1998.0379](https://doi.org/10.1006/jmsc.1998.0379)
- OSPAR (2017): at the Skagerrak coast (Gullmar Fjord entrance, 14C) primary production declined
  over 1985-2012 (with higher rates 1992-1996); Liverpool Bay mean 223 (167-296), Plymouth L4
  90-130, Celtic Sea shelf break 163-245 g C m-2 yr-1. — [OSPAR Intermediate Assessment 2017, Pilot assessment of production of phytoplankton](https://oap.ospar.org/en/ospar-assessments/intermediate-assessment-2017/biodiversity-status/fish-and-food-webs/phytoplankton-production/)
- North Sea (Fransz & Gieskes 1984, Table 7.1 in Mackinson & Daskalov): annual PP Southern Bight
  coast 200, Southern Bight offshore 250, central North Sea 1981 200-250, northern North Sea (FLEX)
  >>100 (175) g C m-2 yr-1; average of regions 212 g C m-2 yr-1. — [Mackinson & Daskalov 2007, p. 97](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Global estuarine-coastal compilation: median 185, mean 252 g C m-2 yr-1 (131 ecosystems); APPP
  varies up to 10-fold within ecosystems and 5-fold between years; method differences can produce
  up to 3-fold variability. — [Cloern et al. 2014, Biogeosciences 11:2477-2501](https://doi.org/10.5194/bg-11-2477-2014)
- North Sea Ecopath phytoplankton biomass and P/B (Mackinson & Daskalov 2007): average phytoplankton
  standing stock March-June (FLEX 1976) ~750 mg C m-2 (peak > 4000 mg C m-2); southern North Sea
  microplankton biomass 3.7 g C m-2 (Hannon & Joiris 1989); model P/B = 286 yr-1, computed as
  production 2,150 g ww m-2 yr-1 / biomass 7.5 g ww m-2; the report calls the g C -> wet weight
  conversion "an important source of uncertainty" (alternative factors give 400-8000 g ww m-2 yr-1
  from 170 g C m-2 yr-1). — [Mackinson & Daskalov 2007, pp. 97-98](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Western Baltic Sea Ecopath (ICES SD 22+24): phytoplankton B = 2.161 g C m-2 (mean of published
  range 1.01-3.312 g C m-2), P/B = 120 yr-1; input P/B range [87.5, 151.6] yr-1 = average of
  Jarre-Teichmann (1995) and Harvey et al. (2003) Baltic models. — [Scotti et al. 2022, Front. Mar. Sci. 9:879998, Supplementary Materials (GEOMAR repository)](https://oceanrep.geomar.de/id/eprint/57232/2/DataSheet_1_Ecosystem-based%20fisheries%20management%20increases%20catch%20and%20carbon%20sequestration%20through%20recovery%20of%20exploited%20stocks%20The%20western%20Baltic%20Sea.pdf) (Table S7 and P/B section)

### Inferences
- Phytoplankton P/B values converted:

| Source | P/B (yr-1) | d-1 | per tick |
|---|---|---|---|
| Harvey 2003 / Jarre-Teichmann 1995 low (Baltic) | 87.5 | 0.24 | 0.060 |
| Western Baltic Ecopath (Scotti et al.) | 120 | 0.33 | 0.082 |
| Harvey 2003 / Jarre-Teichmann 1995 high | 151.6 | 0.42 | 0.104 |
| Current model r = 0.125/tick (lin) | 182.5 | 0.50 | 0.125 |
| North Sea Ecopath (Mackinson & Daskalov) | 286 | 0.78 | 0.196 |

- The North Sea 286 yr-1 is inflated relative to a mean-annual-biomass P/B: it divides an annual
  production by the March-June mean biomass (0.75 g C m-2) through an uncertain wet-weight
  conversion. Using the same report's southern North Sea biomass of 3.7 g C m-2 with 212 g C m-2
  yr-1 would give ~57 yr-1 instead. Treat it as an upper outlier.
- Scotti et al.: 120 yr-1 x 2.161 g C m-2 = ~259 g C m-2 yr-1, inside the Kattegat/Gullmar 14C range
  (190-290). Kattegat 14C PP 190-290 g C m-2 yr-1 with B ~1-3.3 g C m-2 gives P/B ~58-290 yr-1;
  with B = 2.16 it gives 88-134 yr-1 (0.24-0.37 d-1; 0.06-0.09/tick).
- These P/B values are (a) realised annual means including winter light limitation and summer
  nutrient limitation, and (b) gross of all grazing (micro + meso), sinking and lysis. Two
  corrections act in opposite directions when mapping to a logistic r:
  - Logistic r is the specific rate at B << K; realised production per biomass is r(1 - B/K), so
    at B ~ K/2 the realised rate is r/2. This argues for r > annual P/B (up to ~2x).
  - The model's r should be net of microzooplankton grazing (~60%, Calbet & Landry 2004) and
    sinking. This argues for r ~ 0.4 x P/B.
  - Combined, r ~ 0.8 x P/B: with P/B = 120 yr-1 that is ~0.26 d-1 = 0.066/tick; with the range
    88-152 yr-1, r ~ 0.048-0.083/tick. This is a rough heuristic, not a calibration.
- The current r = 0.125/tick (0.5 d-1, 182 yr-1 lin; 0.47 d-1 / 172 yr-1 if read exponentially)
  is inside the Ecopath P/B span (87.5-286 yr-1), but those P/B values are gross of micrograzing;
  after the micrograzing correction the literature points to roughly half the current value.

### Gaps
- No Kattegat- or Skagerrak-specific Ecopath model with a phytoplankton P/B was found (searches for
  "Ecopath Kattegat" returned nothing relevant). Tomczak et al. (2012, Baltic Proper) and Harvey et
  al. (2003) parameter tables could not be read directly (Elsevier/OUP blocked); their values are
  known only via the Scotti et al. averaged range.
- Henriksen (2009) reportedly shows Kattegat/Belt Sea mean annual phytoplankton biomass of ~200
  ug C/L until the mid-1980s declining to ~100 ug C/L or less from the early 1990s; this comes only
  from a search-engine summary and is NOT verified. [Henriksen 2009, J. Sea Res. 61:114-123, doi:10.1016/j.seares.2008.10.003](https://doi.org/10.1016/j.seares.2008.10.003)
- No depth-integrated Kattegat phytoplankton carbon biomass (g C m-2) from a primary source was
  found, so the Kattegat P/B ratios above borrow the western Baltic biomass.
- Carstensen et al. (2004) (Kattegat summer blooms, bloom frequency 8.7%, large Ceratium and
  Rhizosolenia) gives no rates. [Carstensen et al. 2004, Limnol. Oceanogr. 49:191](https://doi.org/10.4319/lo.2004.49.1.0191)

## 4. Mesozooplankton (copepod) weight-specific growth and P/B

### Takeaway
Global growth models give copepod weight-specific production of ~0.04-0.08 d-1 at 5-11.5 C under
realistic food (Hirst & Bunker 2003) and 0.08-0.16 d-1 if food-unlimited (Huntley & Lopez 1992);
Skagerrak August measurements were 0.10 d-1 (adult females) and 0.27 d-1 (juveniles). Integrated
over a year this is ~18-57 yr-1, while the North Sea Ecopath model uses 9.2 (a May-September
value applied annually). The project's 15-40 yr-1 is plausible but unsourced; mid-range
(~20-30 yr-1, 0.014-0.021/tick) is best supported for a non-seasonal model.

### Cited Findings
- Huntley & Lopez (1992): 181 generation-time estimates, 33 copepod species, -1.7 to 30.7 C;
  temperature alone explains > 90% of variance in growth; weight-specific growth independent of
  body size; hypothesises that food may not limit growth in nature. — [Huntley & Lopez 1992, Am. Nat. 140:201-242, doi:10.1086/285410](https://doi.org/10.1086/285410)
- The Huntley & Lopez model is g = 0.0445 e^(0.111 T) (d-1) per one secondary source and
  g = 0.045 e^(0.111 T) per another; it is reported to overestimate growth at high temperature.
  Applied in Gamak Bay (Korea) it gave copepod P/B 0.08-0.86 d-1 (mean 0.33), vs 0.03-0.33 d-1
  (mean 0.18) with Hirst & Bunker. — [Gamak Bay copepod production, Fish. Aquat. Sci. 24(4):171](https://www.e-fas.org/archive/view_article?pid=fas-24-4-171)
- Hirst & Bunker (2003) model, as quoted in an application paper: log10 g = 0.0186 T - 0.288
  log10 W + 0.417 log10 Chl - 1.209, g in d-1, T in C, W = individual body weight in **ug C
  ind-1**, Chl in ug/L; applies to nauplii, copepodites and adults of both spawning strategies.
  — [Fish. Aquat. Sci. 24(4):171 (Gamak Bay), Eq. 5](https://www.e-fas.org/archive/view_article?pid=fas-24-4-171); another secondary source gives W in mg
  — [ICES J. Mar. Sci. 77:419 (Chile)](https://academic.oup.com/icesjms/article/77/1/419/5612122)
- Hirst & Bunker (2003) abstract: in situ adult weight-specific fecundity has Q10 = 1.59
  (broadcast) and 1.43 (sac spawners), much lower than food-saturated lab Q10 of 2.75 and 3.98;
  in situ rates approximate food-saturated rates at 0-10 C; juveniles grow much faster and closer to
  food saturation than adults of similar size. — [Hirst & Bunker 2003, Limnol. Oceanogr. 48:1988-2010, doi:10.4319/lo.2003.48.5.1988](https://doi.org/10.4319/lo.2003.48.5.1988) (abstract via [OpenAlex](https://api.openalex.org/works/doi:10.4319/lo.2003.48.5.1988))
- Hirst & Lampitt (1998) derived global equations for in situ weight-specific fecundity and
  juvenile growth from body weight and temperature. — [Hirst & Lampitt 1998, Mar. Biol. 132:247-257, doi:10.1007/s002270050390](https://doi.org/10.1007/s002270050390)
- Skagerrak, 8-station transect, August 1988: specific growth rates averaged 0.10 d-1 (adult
  females, egg production) and 0.27 d-1 (juveniles, moulting rates; near lab maximum, i.e. not
  food-limited); egg production food-limited (75% of max for Centropages typicus, 50% Calanus
  finmarchicus, 30% Paracalanus parvus, 15% Acartia longiremis and Temora longicornis); community
  copepod production 3-8 mg C m-3 d-1, mean 4.6; egg production = 25% of total production; chl
  0.2-2.5 ug/L. — [Peterson, Tiselius & Kiorboe 1991, J. Plankton Res. 13:131-154, doi:10.1093/plankt/13.1.131](https://doi.org/10.1093/plankt/13.1.131)
- Southern Kattegat, seasonal study: copepod production episodic, in bursts tied to three net
  phytoplankton blooms; biomass unimodal with peak June-July; biomass declined Aug-Oct during the
  largest production event (Aug-Sep), implying high mortality; egg production correlated with
  chl > 11 um and total microplankton biomass. — [Kiorboe & Nielsen 1994, Limnol. Oceanogr. 39:493-507, doi:10.4319/lo.1994.39.3.0493](https://doi.org/10.4319/lo.1994.39.3.0493) (abstract via [OpenAlex](https://api.openalex.org/works/doi:10.4319/lo.1994.39.3.0493))
- North Sea Ecopath (Mackinson & Daskalov 2007), herbivorous + omnivorous zooplankton (mainly
  copepods): from Fransz et al. (1991b), May-September production and P/B for Temora longicornis
  (P/B 8.667), Acartia clausi (7.667), Pseudocalanus elongatus (11.167), average 9.17; the table
  caption defines "year = 153 days from May to September". The model nevertheless applies P/B 9.2
  to the annual North Sea copepod production of 12.35 g C m-2 yr-1 (147 g ww m-2 yr-1; Fransz &
  Gieskes 1984) to back out biomass 16 g ww m-2. Q/B = 30 yr-1 (also on a 153-day year); P/Q =
  0.30; unassimilated 38%. — [Mackinson & Daskalov 2007, section 10.1, Table 10.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Same report, other North Sea production values: coastal mixed areas 5-20 g C m-2 yr-1 (Fransz et
  al. 1991b); Evans (1977) 3.5 g C m-2 yr-1; Roff et al. (1988) 16.5 g C m-2 yr-1 (considered too
  high); Martens (1980) Wadden Sea 0.4 (considered too low); Fransz & van Arkel (1980) daily
  production 0.02-0.050 g C m-2 d-1 at the end-April phytoplankton peak, when population biomass was
  ~0.4 g C m-2; Calanus biomass rising to 4 g C m-2 by end of May; Fladen summer peak standing stock
  ~12.5 g C m-2 (mostly C. finmarchicus); carnivorous zooplankton (euphausiids) P/B 2.5 yr-1.
  — [Mackinson & Daskalov 2007, pp. 105-108](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Western Baltic Ecopath: a single pooled "zooplankton" group (macro + meso + micro) has
  B = 0.697 g C m-2, P/B = 76.69 yr-1, Q/B = 271.36 yr-1; P/B weighted from macro-, meso- and
  microzooplankton values of earlier Baltic models. For fish, Ecopath P/B is set equal to total
  mortality Z. — [Scotti et al., Supplementary Materials, Table S7](https://oceanrep.geomar.de/id/eprint/57232/2/DataSheet_1_Ecosystem-based%20fisheries%20management%20increases%20catch%20and%20carbon%20sequestration%20through%20recovery%20of%20exploited%20stocks%20The%20western%20Baltic%20Sea.pdf)
- Hirst & Kiorboe (2002): predation accounts for ~2/3 to 3/4 of total adult copepod mortality,
  independent of temperature; mortality increases with temperature. — [Hirst & Kiorboe 2002, Mar. Ecol. Prog. Ser. 230:195-209, doi:10.3354/meps230195](https://doi.org/10.3354/meps230195)

### Inferences
- Unit check for Hirst & Bunker: T = 10 C, W = 5 ug C, Chl = 2 ug/L gives g = 0.080 d-1; with W
  read as 0.005 mg the formula gives 0.58 d-1, which is inconsistent with the 0.10/0.27 d-1
  measured in the Skagerrak. The ug C reading is the plausible one.
- Worked copepod rates (computed):

| Case | g (d-1) | per tick | yr-1 if constant |
|---|---|---|---|
| Huntley & Lopez, 5 C | 0.078 | 0.019 | 28 |
| Huntley & Lopez, 10 C | 0.135 | 0.034 | 49 |
| Huntley & Lopez, 11.5 C | 0.159 | 0.040 | 58 |
| Huntley & Lopez, 17 C | 0.294 | 0.073 | 107 |
| Hirst & Bunker, 5 C, 10 ug C, chl 1 | 0.039 | 0.010 | 14 |
| Hirst & Bunker, 11.5 C, 5 ug C, chl 1 | 0.064 | 0.016 | 23 |
| Hirst & Bunker, 10 C, 5 ug C, chl 2 | 0.080 | 0.020 | 29 |
| Hirst & Bunker, 15 C, 2 ug C, chl 2 | 0.128 | 0.032 | 47 |
| Skagerrak Aug 1988, adult females | 0.10 | 0.025 | (36) |
| Skagerrak Aug 1988, juveniles | 0.27 | 0.068 | (99) |
| Fransz & van Arkel, North Sea late April | 0.05-0.125 | 0.013-0.031 | - |
| North Sea Ecopath P/B 9.17 per 153 d | 0.060 | 0.015 | 9.2 (as used) / 22 (if 0.06 d-1 all year) |
| Project range low (15 yr-1) | 0.041 | 0.0103 | 15 |
| Project range high (40 yr-1) | 0.110 | 0.0274 | 40 |

- Illustrative annual integration (my assumption, not a sourced temperature record): with a
  sinusoidal surface temperature of 3-17 C (mean 10 C), Huntley & Lopez integrates to ~57 yr-1
  (food-unlimited upper bound); Hirst & Bunker with W = 5 ug C and chl 1.5 gives ~26 yr-1, with
  W = 10 ug C and chl 1 ~18 yr-1.
- Verdict on the project's 15-40 yr-1: the lower half (15-30) is supported by Hirst & Bunker
  annual integrations and by the North Sea effective daily rate (0.06 d-1 -> 22 yr-1 if sustained);
  40 yr-1 requires summer-like rates all year. The North Sea Ecopath annual application (9.2 yr-1)
  is below the range, but that value is a 153-day figure. Suggested central value ~20-30 yr-1 =
  0.055-0.082 d-1 = 0.014-0.021 per tick.
- Gross vs net: weight-specific growth (H&L, H&B, Peterson et al.) and Ecopath P/B are
  production P = somatic growth + egg production, after respiration and before mortality, which
  matches the model's DM growth term. Ecopath P/B is constrained to equal total mortality Z at
  steady state, so in Ecopath models it implicitly includes predation + other mortality.
- Copepod P/B > Ecopath zooplankton values that pool microzooplankton (western Baltic 76.7 yr-1)
  must not be used for mesozooplankton; microzooplankton P/B is much higher and inflates the mean.

### Gaps
- Exact Hirst & Bunker coefficients, the body-weight unit, and the separate equations for
  broadcast/sac spawners and juveniles/adults were verified only via secondary sources (original
  paper behind Wiley 403); the Hirst & Lampitt (1998) equation coefficients could not be retrieved.
- Kiorboe & Nielsen (1994) reportedly estimate annual southern Kattegat copepod production at
  ~12 g C m-2 with a seasonal net biomass increase of only ~1 g C m-2 (< 10% of production); this is
  from a search-engine summary only, NOT verified. Their annual mean biomass (needed for a Kattegat
  P/B) was not obtained.
- Annual copepod P/B values of "7.3 yr-1" and "2.7 yr-1" appeared in a search summary without a
  traceable source; not used.
- Fransz et al. (1991b), Fransz & Gieskes (1984), Tomczak et al. (2012) and Harvey et al. (2003)
  mesozooplankton P/B values were not read in the original.
- No regional temperature climatology for the model area was retrieved; the 3-17 C cycle above is
  an assumption for illustration only.

## 5. Seasonality of phytoplankton and copepod production in the region

### Takeaway
Primary production varies ~10-fold through the year (winter ~100 vs spring/summer ~1000 mg C
m-2 d-1 in the North Sea); in the Kattegat ~19% of annual PP falls in the spring bloom and ~30% in
subsurface summer layers. Copepod production is episodic (bloom-linked bursts, largest in
Aug-Sep in the southern Kattegat) while biomass peaks June-July, so annual-mean rates hide strong
seasonal pulses.

### Cited Findings
- North Sea PP by season (Fransz & Gieskes 1984): Jan-Feb ~100, Mar-May 1000-1200, Jun-Sep 700-1000,
  Oct-Dec 100-500 mg C m-2 d-1 (Southern Bight, central North Sea). — [Mackinson & Daskalov 2007, Table 7.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Southern Kattegat: ~19% of 290 g C m-2 yr-1 during the spring bloom; ~30% in subsurface summer
  populations; summer surface layer nutrient-limited. — [Richardson & Christoffersen 1991](https://doi.org/10.3354/meps078217)
- Kattegat PP increase since the 1950s visible in spring bloom and summer, not in light-limited
  winter. — [Richardson & Heilmann 1995](https://doi.org/10.1080/00785236.1995.10422050)
- Gullmar Fjord: interannual PP range 180-339 g C m-2 yr-1, correlated with wind and Kattegat
  run-off. — [Lindahl et al. 1998](https://doi.org/10.1006/jmsc.1998.0379)
- Southern Kattegat copepods: production in bursts with three net phytoplankton blooms; biomass
  unimodal, peak June-July; largest production event Aug-Sep. — [Kiorboe & Nielsen 1994](https://doi.org/10.4319/lo.1994.39.3.0493)
- Southern Kattegat ciliates: spring peak, smaller early-autumn peak; growth Q10 = 2.6 (2.2-3.0),
  rarely food-limited; copepods could control ciliates >= 50 um year-round. — [Nielsen & Kiorboe 1994, Limnol. Oceanogr. 39:508-519, doi:10.4319/lo.1994.39.3.0508](https://doi.org/10.4319/lo.1994.39.3.0508)
- Helgoland (southern North Sea): Temora longicornis and Pseudocalanus spawn year-round but at low
  rates in winter; Acartia clausi does not spawn end-Sep to end-Jan; maximum egg production in
  April/May. — [Halsband & Hirche 2001, Mar. Ecol. Prog. Ser. 209:219, doi:10.3354/meps209219](https://doi.org/10.3354/meps209219)
- North Sea: copepods reach 80-90% of zooplankton biomass by May; copepod grazing ~14/9/3% of PP in
  May/June/September (central North Sea). — [Mackinson & Daskalov 2007, section 10.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- In situ copepod fecundity Q10 1.43-1.59 (adults); juveniles more strongly temperature-dependent.
  — [Hirst & Bunker 2003](https://doi.org/10.4319/lo.2003.48.5.1988)

### Inferences
- With no seasonality in the model, an annual-mean rate is appropriate for both groups, but the
  phytoplankton annual P/B (~90-150 yr-1) averages winter rates ~10x below spring/summer rates;
  a non-seasonal logistic r calibrated to spring rates would overstate annual production.
- For copepods, temperature alone gives roughly a 3-4x seasonal range in g (H&L: 0.078 d-1 at 5 C
  vs 0.29 d-1 at 17 C), and food-pulse effects add episodic variation on top.

### Gaps
- No monthly copepod P/B series for the Kattegat/Skagerrak were obtained (Kiorboe & Nielsen 1994
  full text not accessible).
