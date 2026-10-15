# Predator interference in herring feeding: is `interference: 0.7` for `pelagic_fish` supported?

Model state (supplied by the caller, plus a read-only check of the code and library; nothing was edited):
- Herring (`pelagic_fish`) eat zooplankton with a type III Beddington-DeAngelis response
  f = a Z^2 / (1 + a h Z^2 + w B_herring(c)), Z = visible zooplankton (t/km2 = g WW m-2, whole column, 1-km cell).
  a = 0.04, h = 25, so a h = 1 km2/t in the S = a h Z^2 term. w = 0.7 km2/t (`fg_library.yaml`, `pelagic_fish: interference: 0.7`).
- Herring cell density ~3.4 t/km2, so I = w B = 2.4. Visible zooplankton ~0.45 t/km2 in winter, ~4 in summer (caller).
- Code check (`lib/environments/ecosystem_env/predation.py` lines 33-40; GPU `lib/gpu/ecosystem.py` line 490):
  I = `dm_interference * predator_biomass`, i.e. the **total** herring biomass in the cell, including the
  fraction that chose Rest/hide or Move that tick. Only the demand term is multiplied by `actions.eat`.
- The herring -> zooplankton pair also has `visibility_floor: 0.2`, `dark_ratio: 0.1`, `light_threshold_deg: -8`
  (daylight calendar), so the winter attack rate is further reduced; I is not light-scaled.
- History (git): the commit that introduced interference (5017884, 2026-09-20, resume section 86) already had
  herring `max_intake_rate: 0.04` and `handling_time: 25.0`. Unlike the gadoids (sibling report: handling
  rose 3.5-8.7x after section 86), herring a and h are unchanged since w was set. What changed for herring is
  the seasonal zooplankton cycle, the daylight multiplier and the visibility floor, all of which lower winter S.

Builds on, not redone: `reports/Täthetsberoende och interferens hos torskfiskar.md` and
`research_notes/Täthetsberoende och interferens hos torskfiskar/predator_interference.md` (general interference
syntheses: DeLong & Vasseur 2011 median m ~0.7; Skalski & Gilliam 2001 18/19 predator-dependent; HV/BD/Ecosim
conversion machinery; Beddington mechanism check; Fall et al. 2021 cod null); and
`research_notes/Täthetsberoendets styrka per grupp/pelagic_fish.md` (herring density-dependent growth Q2, spatial
scale Q6). Those are cited as "sibling note".

Conventions: **[F]** fulltext read (or the relevant part of it), **[A]** abstract read, **[S]** search snippet or
machine summary only (NOT verified), **[2nd]** number read in another document that cites the original,
**[D]** computed by me (from data or model parameters). Every section separates *Literature*, *Calculation* and
*Inference*.

---

## How w = 0.7 was chosen (resume section 86, read in full)

- Purpose: remove the 2-cell vertical bands (grid Nyquist mode). Demand was linear in predator biomass, so
  "Packing was therefore free at every density, aggregation was always weakly optimal" (86.1). Any w > 0 gives
  total cell intake a finite ceiling a B_prey / w and aggregation a finite optimum.
- Calibration (86.3): w that makes interference as strong as prey saturation at the observed p90 predator
  density, from a banded 16x16 run: pelagic_fish sat p50 = 15.0, B_pred p90 = 6.5 t, sat/p90 = 2.3. Two strengths
  were trained; "the weaker one won, so the library ships with one third of it" -> 0.7. The criterion for
  "won" was equal band removal at a lower biomass cost (86.5).
- 86.7, verbatim: "Open calibration question: only {0, w/3, w} were swept. Whether an even weaker term still
  clears the bands is untested, and the biomass cost suggests it is worth one more run."
- So 0.7 is an upper bound on what the artefact fix needs, not a measurement, and it was set against a prey
  saturation term S ~15. **Inference:** the herring part of the premise moved: winter visible Z ~0.45 gives
  S ~0.2, about 75x smaller than the S the calibration assumed.

---

## Sources and how they were read

| Source | Read as | Link |
|---|---|---|
| Brunel & Dickey-Collas 2010, MEPS 405:15, 15 Atlantic herring stocks, density vs VB growth | [F] (pdftotext, methods/results/Table 3) | https://doi.org/10.3354/meps08491 (PDF https://www.int-res.com/articles/meps2010/405/m405p015.pdf) |
| Stenevik, Hølleland, Enberg, Høines & Salthaug 2022, ICES JMS 79:1340, NSS herring density-dependent growth | [F] partial (first 100k of 142k chars via WebFetch; Table 2 coefficients and quoted sentences) | https://doi.org/10.1093/icesjms/fsac057 |
| Hunter, Speirs & Heath 2019, PLoS ONE 14:e0212176, North Sea herring/sprat growth vs abundance | [F] (first 100k of 104k chars via WebFetch) | https://doi.org/10.1371/journal.pone.0212176 |
| Casini, Bartolino, Molinero & Kornilovs 2010, MEPS 413:241, central Baltic herring growth | [F] (pdftotext) | https://doi.org/10.3354/meps08592 |
| Casini, Cardinale & Hjelm 2006, Oikos 112:638, Baltic clupeid condition | [A] (OpenAlex) | https://doi.org/10.1111/j.0030-1299.2006.13860.x |
| Casini et al. 2011, Popul Ecol 53:511, spatial density dependence of clupeid condition | [A] (OpenAlex) | https://doi.org/10.1007/s10144-011-0269-2 |
| Cardinale & Arrhenius 2000, ICES JMS 57:882, Baltic herring weight-at-age 1986-1996 | [A] (OpenAlex) | https://doi.org/10.1006/jmsc.2000.0575 |
| Möllmann et al. 2004, J Fish Biol 65:1563, central Baltic herring/sprat feeding 1977-1999 | [A] (Crossref) | https://doi.org/10.1111/j.0022-1112.2004.00566.x |
| Bernreuther et al. 2018, MEPS 592:181, trophic decoupling, Bornholm Basin | [A] (OpenAlex) | https://doi.org/10.3354/meps12512 |
| Eggers 1976, J Fish Res Board Can 33:1964, theoretical effect of schooling on prey consumption | [A] (Crossref abstract) | https://doi.org/10.1139/f76-250 |
| Maszczyk, Bartosiewicz, Jurkowski & Wyszomirski 2014, Limnology 15:155, interference in roach | [A] (OpenAlex) | https://doi.org/10.1007/s10201-014-0426-3 |
| Milne, Shuter & Sprules 2005, CJFAS 62:1210, lake herring (cisco) schooling and feeding | [F] (pdftotext: abstract, Fig. 8, discussion start) | https://doi.org/10.1139/F05-030 |
| Pitcher, Misund, Fernö, Totland & Melle 1996, ICES JMS 53:449, feeding herring schools by sonar | [A] (OpenAlex) | https://doi.org/10.1006/jmsc.1996.0063 |
| Misund 1998, ICES JMS 55:58, tracking herring schools by sonar (school split/join rates) | [A] (OpenAlex) | https://doi.org/10.1006/jmsc.1997.0228 |
| Gregson & Booth 2005, MEPS 299:269, planktivorous reef fish shoal size vs copepod patches | [A] (OpenAlex) | https://doi.org/10.3354/meps299269 |
| Kingsford & MacDiarmid 1988, MEPS 48:103, local zooplankton depletion by reef planktivores | [A] (first page of PDF) | https://www.int-res.com/articles/meps/48/m048p103.pdf |
| van der Lingen 1994, MEPS 109:1, pilchard filter/particulate clearance rates | [A] (OpenAlex) | https://doi.org/10.3354/meps109001 |
| James & Findlay 1989, MEPS 50:275, Cape anchovy feeding mode vs prey | [A] (OpenAlex) | https://doi.org/10.3354/meps050275 |
| Aarflot, Dalpadado & Fiksen 2020, MEPS 644:129, planktivore foraging vs bathymetry | [A] (OpenAlex) | https://doi.org/10.3354/meps13343 |
| Schrandt & Powers 2015, PLoS ONE 10:e0130095, group facilitation in pompano | [A] (OpenAlex) | https://doi.org/10.1371/journal.pone.0130095 |
| Ward, Webster & Hart 2006, Fish Fish 7:231, intraspecific food competition in fishes (review) | [A] (Crossref) | https://doi.org/10.1111/j.1467-2979.2006.00224.x |
| Niwa 2005, arXiv q-bio/0501035, school geometry (quotes Misund 1993a; Pitcher & Partridge 1979) | [F] (intro) | https://arxiv.org/abs/q-bio/0501035 |
| Mackinson & Daskalov 2007, Cefas Tech. Rep. 142, Table 4.5 (fitted 1991 North Sea vulnerabilities) | [D] own column alignment from PDF word coordinates (pages 77-78) | https://www.cefas.co.uk/publications/techrep/tech142.pdf |
| Misund 1993, ICES JMS 50:145, packing density in herring/sprat/saithe schools | [2nd] via Niwa 2005 | https://doi.org/10.1006/jmsc.1993.1016 |
| Beare et al. 2002, ICES JMS 59:469, herring school size vs local school abundance | [2nd] via sibling note `pelagic_fish.md` (read there as fulltext) | https://doi.org/10.1006/jmsc.2002.1201 |
| Dickey-Collas et al. 2010, ICES JMS 67:1875, North Sea herring review | [2nd] via sibling note `pelagic_fish.md` | https://doi.org/10.1093/icesjms/fsq033 |
| Krause 1994, Biol Rev 69:187, fitness returns vs position in groups | [S] (Semantic Scholar machine TLDR only) | https://doi.org/10.1111/j.1469-185x.1994.tb01505.x |
| Krause, Bumann & Todt 1992 (roach, front position and hunger) | [S] (search-engine summary only) | not resolved to a DOI |
| Ryer & Olla 1991, Env Biol Fish 30:317; Magurran & Pitcher 1983, Behav Ecol Sociobiol 12:147; Pitcher, Magurran & Winfield 1982, Behav Ecol Sociobiol 10:149; Morgan 1988, Anim Behav 36:1317; Major 1978, Anim Behav 26:760; Krause 1993, Oecologia 93:356 | citation confirmed (Crossref) only; Ryer & Olla and Magurran & Pitcher also [S] TLDR | https://doi.org/10.1007/bf02028847, https://doi.org/10.1007/bf00343206, https://doi.org/10.1007/bf00300175, https://doi.org/10.1016/s0003-3472(88)80200-8, https://doi.org/10.1016/0003-3472(78)90142-2, https://doi.org/10.1007/bf00317878 |
| Batty, Blaxter & Libby 1986, Mar Biol 91:371, herring filter-feeding in the dark | citation confirmed; content [S] | https://doi.org/10.1007/BF00428631 |

**Failed fetches (recorded, not used for numbers):**
- int-res.com via WebFetch: HTTP 429 (retry-after 3600 s). MEPS PDFs were then downloaded with curl (Brunel 2010,
  Casini 2010, Kingsford & MacDiarmid 1988 worked).
- Casini et al. 2011 fulltext: Springer closed; Unibo CRIS and MPG PuRe repository records HTTP 403; oceanrep.geomar.de
  served a bot challenge. Abstract only.
- Maszczyk et al. 2014 (CC BY, open access): Springer returned a "Client Challenge" page to curl and a login redirect
  to WebFetch. Abstract only; **no effect sizes for the roach interference experiment were obtained.**
- Cardinale & Arrhenius 2000 and Stenevik et al. 2022 PDFs on academic.oup.com: HTTP 403 (Stenevik read as HTML via WebFetch).
- Misund 1993: OUP not tried after repeated 403s; IMR Brage (imr.brage.unit.no) did not resolve (DNS failure).
  **No herring packing density in fish m-3 was verified.** A sardine value (29.5 fish m-3, range 2-233) appeared in a
  search summary only and is not used.
- Eggers 1976 fulltext: ResearchGate 403. Abstract (Crossref) only.
- Krause 1994: closed (Wiley), abstract elided in Semantic Scholar.
- OpenAlex API: one burst of HTTP 429; queries rerouted to Crossref / Semantic Scholar.

---

## Q1. Evidence for interference or intraspecific feeding competition in herring and other schooling planktivores

### Literature
**Within-school effects (school scale, metres to tens of metres)**
- Eggers 1976 [A] extends the Holling disk equation "to account for lateral visual-field overlap and reduction of prey
  density due to removal of prey by forward-occurring members of the school". "This study suggests that schooling of
  predators occurs at the expense of prey consumption. Cost of schooling is reduced if prey densities are high, if
  distance between members is high, or if school size is small. There may be some advantage to schooling when prey
  have a clumped distribution." This is exploitative (front fish deplete the water the rear fish swim through) plus
  visual-field overlap; it is theory, not data.
- Maszczyk et al. 2014 [A], roach (a freshwater shoaling planktivore) on *Daphnia*, prey density held constant by
  minute-by-minute replacement: "fish foraging in a group eat less, due to both exploitation and non-aggressive
  competition for space"; "the effect of interference competition was stronger at higher temperatures, irrespective
  of the prey density". The only experiment found that isolates interference in a planktivorous shoaler; the size of
  the effect was not obtained.
- Krause, Bumann & Todt 1992 [S]: in roach schools of 2 and 4, hungry fish took the front more often and, at equal
  hunger, front fish fed fastest. Krause 1994 [S] (machine TLDR): edge positions give a higher net energy pay-off.
  Not verified; listed only because the brief asks for position effects.
- Gregson & Booth 2005 [A], a planktivorous reef fish: shoals sat on the largest copepod patches, "shoal size
  increasing as prey density increased", but "Available Copepoda per fish, however, decreased with shoal size,
  presenting a possible 'overmatching' foraging situation".
- Kingsford & MacDiarmid 1988 [A], reef planktivores: "Lowest zooplankton densities were usually found in the archway
  where planktivorous fish were abundant"; no such gradient at night or 1 km offshore without fish. Local depletion by
  an aggregation at the 100-m scale (exploitative).

**Facilitation (schooling raising per-capita intake)**
- Milne, Shuter & Sprules 2005 [F], lake herring (cisco, a coregonid, not *Clupea*), gill nets + hydroacoustics:
  "herring stomach fullness increased with school size, suggesting that schooling enhances foraging opportunities for
  individual members" (social facilitation or "many eyes" proposed). Fig. 8: median stomach content Y = 14.7 + 0.1X,
  X = number of fish in the school's net catch, r2 = 0.37, p < 0.001 (daytime). Stomachs were fuller by day (schooled)
  than at night (dispersed).
- Pitcher et al. 1996 [A], 14 actively feeding Norwegian Sea herring schools followed by sonar: "a surprisingly dynamic
  regime of approaching, joining, and splitting that could facilitate rapid adaptive adjustment of herring school size
  to prevailing conditions"; under cod/haddock attacks "most herring were able to continue feeding while retaining the
  risk-dilution advantages of schooling".
- Schrandt & Powers 2015 [A], pompano (benthic prey): groups ate 26 clams vs 14 summed over the same fish alone,
  but "the consumption benefit of group foraging was not shared by all".
- Ryer & Olla 1991 [S], Magurran & Pitcher 1983 [S], Pitcher, Magurran & Winfield 1982 (title: "Fish in larger shoals
  find food faster"): information transfer and patch-finding benefits; only titles/TLDRs seen.

**Field per-capita ration and growth vs conspecific density (stock scale)**
- Brunel & Dickey-Collas 2010 [F], 15 stocks; density = TSB / sea area 0-300 m. Mean stock densities "between 0.23 t
  km-2 in the Irish Sea and 5.6 t km-2 for the Norwegian spring spawning population", single years 0.55-12.1.
  Within populations, winf vs biomass density: "significant and positive in 1 population (Irish Sea) and significant and
  negative in 2"; meta-analytic weighted r = -0.161, p = 0.176 (13 populations); k vs density r = +0.150, p = 0.112.
  Cited there: density-dependent growth in North Sea, Norwegian coast and Gulf of Maine herring; none detected in
  Celtic Sea, West of Scotland and Scotian Shelf "despite major changes in population size".
- Stenevik et al. 2022 [F partial], NSS herring cohorts 1921-2014: l_inf(N) = 36.479 - 0.915 (ln N - ln N_median) cm,
  N = abundance at age 3 + preceding cohort at age 4 (millions); correlation of l_inf with log density -0.66 to -0.69.
  "Modelled asymptotic length varies between 34.5 cm for high densities and 42 cm for the lowest densities."
- Hunter et al. 2019 [F]: "abundance is shown to be negatively correlated to growth rates in herring and sprat"
  (North Sea herring ages 2-5; not in area 6a). No effect sizes reported.
- Dickey-Collas et al. 2010 [2nd]: North Sea herring length-at-age 3 rose ~2 cm during the 1970s collapse, weight-at-age
  fell during recovery.
- Central Baltic: Casini et al. 2010 [F]: "We did not test for intra-specific density-dependence (i.e. effect of herring
  abundance on herring growth) as no evidence has been found for such relationship (Rönkkönen et al. 2004, Casini et al.
  2006)", supported by "the absence of a negative relation between herring abundance and zooplankton in the offshore
  areas"; herring growth is controlled by **sprat** abundance above ~18 x 10^10 individuals. Casini et al. 2006 [A]:
  "total abundance of clupeids ... is by far the most significant predictor of both herring and sprat condition",
  co-varying with zooplankton biomass and with zooplankton weight in stomachs -> "food competition being the main
  mechanism". Casini et al. 2011 [A]: sprat density the main driver; "Herring density was an additional albeit weak
  significant predictor for herring condition"; density dependence also "at the spatial scale within stock units".
  Cardinale & Arrhenius 2000 [A]: herring weight-at-age fell 42-51 % 1986-1996, "well correlated with a drastic increase
  in the number of pelagic fish ... and a consequent reduction in food availability". Möllmann et al. 2004 [A]:
  stomach fullness fell and empty stomachs rose since the late 1970s; diet overlap with the enlarged sprat stock.
- Bernreuther et al. 2018 [A], Bornholm Basin: herring + sprat consumed only 9 % of combined copepod and cladoceran
  production over the year, but more than the production of *T. longicornis* and *P. acuspes* in April-May (C/P > 3.5).

### Calculation
- No source above reports a BD w, a Hassell-Varley m, or per-capita intake against local (cell-scale) herring density.
  The only planktivore interference experiment (Maszczyk 2014) could not be read past the abstract.
- Aggregation scale [D]: 3.4 t/km2 = 3.4 g m-2. Spread over 100 m of water: 0.034 g m-3 = 2-3 x 10^-4 fish m-3 for a
  100-150 g herring; over a 20 m layer, 1.1-1.7 x 10^-3 fish m-3. Tank schools pack at about one fish per body length
  cubed (Pitcher & Partridge 1979, [2nd] via Niwa) = 64 fish m-3 for a 25 cm herring; field herring schools are "an
  order of magnitude lower" (Misund 1993a, [2nd] via Niwa), i.e. ~6 fish m-3 (my reading of "an order of magnitude").
  A herring inside a school therefore experiences ~10^3.5-10^4.5 times the cell-mean conspecific density.
  (Correction to the brief's working estimate of 0.1-0.2 fish m-3 at cell mean, which is ~500x too high.)
- Eggers-type within-school depletion [D]: water crossing a school of packing rho (fish m-3) and length L (m) at speed
  U (m s-1) loses a fraction 1 - exp(-x) of its prey, x = rho Q L / U, Q = clearance rate per fish. Mean intake relative
  to a solitary fish = (1 - e^-x)/x. With pilchard clearance as a herring proxy (van der Lingen 1994 [A]: filtering
  11.78 L fish-1 min-1 = 2.0 x 10^-4 m3 s-1; particulate up to 46.53 L min-1 = 7.8 x 10^-4 m3 s-1), U = 0.25 m s-1
  (assumed, ~1 body length s-1):
  - rho 6, L 20 m, particulate: x = 0.37, relative intake 0.83 (-17 %);
  - rho 6, L 20 m, filtering: x = 0.10, 0.95 (-5 %);
  - rho 30, L 50 m, particulate: x = 4.7, 0.21 (-79 %);
  - rho 1, L 20 m: x = 0.06, 0.97 (-3 %).
  The cost is set by school geometry, which herring change on a ~5-14 min time scale (Pitcher 1996; Misund 1998 [A]:
  inter- and intra-school events "occurred at average rates of about 14 minutes").

### Inference
- **The documented density dependence in herring is exploitative, and stock-scale.** Every field signal that names a
  mechanism names food depletion or competition for zooplankton, usually with sprat (Casini 2006, 2010, 2011; Cardinale &
  Arrhenius; Möllmann). The central Baltic, the best-studied system, finds no or "weak" intraspecific herring effect.
  None of it isolates behavioural interference. The model already has explicit depletion of the shared zooplankton pool,
  so these signals do not call for an additional w term.
- **The school-scale mechanisms point both ways.** Within-school depletion and visual overlap (Eggers, Maszczyk,
  Gregson & Booth) lower per-capita intake; patch-finding, information transfer and "many eyes" raise it (Milne, Pitcher
  1996, Ryer & Olla). In the only field data on a schooling planktivore that relate intake to group size (Milne 2005,
  cisco), the net sign was **positive**.
- **School-scale costs do not scale with cell density.** If extra herring in a cell form more schools rather than
  bigger ones - Beare et al. 2002 [2nd]: school size independent of local school abundance - the within-school cost is
  a constant fraction on the attack rate, independent of B_herring(c). That belongs in a (already a free calibration
  constant), not in a w B term. Only school-size growth with density or overlap of school wakes inside a cell would make
  it density dependent, and no source shows either for herring.

---

## Q2. Quantitative estimates convertible to w, m, or intake reduction per doubling

### Literature
- Stenevik et al. 2022 [F partial]: dl_inf/d ln N = -0.915 cm at l_inf = 36.48 cm (NSS herring).
- Brunel & Dickey-Collas 2010 [F]: within-population winf-density r = -0.16, n.s.; 2/13 significant negative, 1 positive.
- Mackinson & Daskalov 2007, Table 4.5, fitted 1991 North Sea Ecosim [D, own alignment]: vulnerability K on
  "Herbivorous & Omnivorous zooplankton (copepods)" for predator columns 28 juvenile herring = **2.1**, 29 adult herring
  = **4**, 30 sprat = **2.2**, 31 mackerel = 1.5, 33 sandeels = 1.5. Adult herring on carnivorous zooplankton = 4,
  juvenile herring = 2.1. Alignment method: the table is printed rotated, so values were matched to predator columns by
  their PDF y-coordinate. Check: the same procedure gives adult cod, haddock and saithe K = 6 and adult whiting K = 3 on
  herring, as the sibling report read them.
- No herring m or BD w exists in what was retrieved. The general distribution (DeLong & Vasseur median m ~0.7) is in the
  sibling note and is measured at arena densities.

### Calculation [D]
**(a) Model diagnostics at B = 3.4 t/km2, w = 0.7 (I = 2.38).**
Per-capita intake ratio per doubling of B: (1 + S + wB)/(1 + S + 2wB). Local Hassell-Varley exponent
m_local = -d ln f / d ln B = wB/(1 + S + wB). In BD's time-budget reading, the share of time lost to conspecifics
is I/(1 + S + I).

| Season | Z_vis | S = ahZ^2 | Cut vs B -> 0 | Cut per doubling | m_local | Time: search / handling / interference | Z_half |
|---|---|---|---|---|---|---|---|
| Winter | 0.45 | 0.20 | 66 % | **40 %** | 0.66 | 0.28 / 0.06 / **0.66** | 1.00 -> 1.84 |
| Mid | 2.0 | 4.0 | 32 % | 24 % | 0.32 | 0.14 / 0.54 / 0.32 | 1.00 -> 1.84 |
| Summer | 4.0 | 16 | 12 % | 11 % | 0.12 | 0.05 / 0.83 / 0.12 | 1.00 -> 1.84 |

Sensitivity to cell density [D] (occupied-cell spread for pelagic_fish from resume 86.3: p50 0.60, p90 6.5 t/km2,
measured in the banded 16x16 run, so indicative only):

| B (t/km2) | I | Winter cut per doubling (level cut) | Summer cut per doubling (level cut) |
|---|---|---|---|
| 0.6 (p50) | 0.42 | 21 % (26 %) | 2.4 % (2.4 %) |
| 3.4 | 2.38 | 40 % (66 %) | 11 % (12 %) |
| 6.5 (p90) | 4.55 | 44 % (79 %) | 17 % (21 %) |

At median density the summer side of w = 0.7 lies inside the empirical 0-5 % range. The winter side is 4-10x above it at every density.

(For type III, interference moves the half-saturation from sqrt(1/(ah)) to sqrt((1 + I)/(ah)), here from 1.0 to 1.84 t/km2.
The daylight multiplier lowers winter a further, so winter S is if anything below 0.2.)

**(b) Converting the NSS herring growth slope to an intake reduction per doubling.**
- d l_inf per doubling = -0.915 x ln 2 = -0.634 cm = **-1.7 %** of 36.48 cm; W_inf ~ l_inf^3 -> -5.1 %.
  This assumes the paper's log is natural, which the WebFetch reader inferred from one worked example
  ("log 7355.9 = 8.90"; ln 7355.9 = 8.90). The coefficient came through a WebFetch summary and was not checked against
  the authors' code (github.com/holleland/NSSHdensGrowth). If the log were base 10, the slope would be 0.275 cm
  (0.75 %) per doubling, which only weakens the empirical effect.
- In von Bertalanffy, dW/dt = H W^(2/3) - k W and l_inf ~ H/k, so with catabolism fixed d ln H = d ln l_inf:
  the anabolic (intake) term falls **~1.7 % per doubling** of abundance. Reading the W_inf change as an intake change
  gives an upper reading of ~5 %. Caveats: N is stock abundance in numbers, not t/km2; if the occupied area grows with
  abundance, local density rises less than N, so the effect per doubling of *local* density would be larger. The
  observed density range was not extracted (the 34.5-42 cm span would need ln N about +2.2 and -6 around the median).
- North Sea (rough): ~2 cm on a ~25 cm age-3 length (~8 %) across a collapse of at least an order of magnitude (>= 3.3
  doublings; the size of the collapse is my assumption here, not taken from Dickey-Collas) -> <= ~2.4 % length per
  doubling.
- Brunel's within-stock r = -0.16 (n.s.) is consistent with effects of this size or zero.
- **Working empirical range: r = 0-5 % per doubling (most likely 2-3 %), and that includes exploitative depletion,
  so the interference share is smaller still.**

**(c) Inverting to w.** From r = wB/(1 + S + 2wB): wB = r(1 + S)/(1 - 2r), w = that / 3.4.

| r per doubling | w if fitted at winter S = 0.2 | at S = 4 | at summer S = 16 |
|---|---|---|---|
| 1.7 % | 0.006 | 0.026 | 0.088 |
| 3 % | 0.011 | 0.047 | 0.16 |
| 5 % | 0.020 | 0.082 | 0.28 |
| 10 % | 0.044 | 0.18 | 0.63 |

The same w = 0.7 means 11 % per doubling in summer and 40 % in winter; a w matched to 3 % in winter (0.011) is 15x
smaller than one matched to 3 % in summer (0.16).

**(d) Ecosim K to w.** With the model's own form, K = (1 + S + wB0)/(wB0), so wB0 = (1 + S)/(K - 1)
(sibling report's derivation; the a Z^2 numerator cancels, so it holds for type III). With B0 = 3.4:

| K | winter S = 0.2 | S = 4 | summer S = 16 |
|---|---|---|---|
| 4 (adult herring) | w = 0.12 | 0.49 | 1.67 |
| 2.1 (juvenile herring) | 0.32 | 1.34 | 4.5 |

K is defined at the Ecopath base state (annual mean, area mean), has no season, and in Ecosim comes from a prey-side
foraging-arena exchange, not predator behaviour. It encodes zooplankton refuges at least as much as interference.

**(e) Beddington mechanism check** (sibling note Q4 form, w = a_c tau). For type III the per-capita search rate is
a Z = 0.018 km2 t-1 tick-1 in winter and 0.16 in summer. If herring meet conspecific biomass at their prey search rate,
w = 0.7 requires tau = 0.7/0.018 = 39 ticks (~10 days) lost per encounter-equivalent in winter, 4.4 ticks (~1 day) in
summer. With the bare a = 0.04: 17.5 ticks (~4 days).

### Inference
- No literature value converts directly to w. The two indirect routes disagree in level but agree on the seasonal
  problem: growth data put w at 0.006-0.02 if the constraint is applied in winter and 0.09-0.28 in summer; Ecosim K = 4
  gives 0.12 (winter) to 1.7 (summer).
- w = 0.7 reproduces Ecosim's adult-herring K = 4 only at S ~ 7-8 (mid-season), and gives 11 % per doubling in summer,
  roughly 2-5x the growth-based range there. In winter it gives 40 % per doubling, roughly 10-20x above it, and it
  has herring spending two thirds of their time on conspecifics. For a schooling, non-aggressive filter/particulate
  feeder this is not supported.
- The Beddington check gives the same verdict: direct behavioural interference at 3.4 t/km2 cell mean would need days of
  lost foraging per encounter-equivalent. A w of that size can only stand for unresolved sub-cell aggregation or
  depletion (Q1), and Q1 argues that those costs are density independent at cell scale.

---

## Q3. Is interference stronger or weaker when prey is scarce?

### Literature
- Eggers 1976 [A]: the cost of schooling "is reduced if prey densities are high". Mechanism: when fish are
  handling-limited, the front fish clear less water per unit time and leave more for those behind. This matches BD
  qualitatively (relative cost largest at low prey).
- Maszczyk et al. 2014 [A]: interference in roach was temperature dependent "irrespective of the prey density" across
  three prey densities, i.e. no prey-density dependence detected in the abstract's summary.
- Gregson & Booth 2005 [A]: shoals grow where prey is dense (aggregation tracks food), and per-fish prey still fell with
  shoal size.
- Herring winter behaviour: herring disperse into layers at night and switch to filter feeding in the dark (Batty et al.
  1986 [S]); winter rations are near maintenance (winter note Q3: Lohmeyer & Hempel 1977, Bernreuther 2007). No source
  attributes low winter intake to crowding. Casini 2006 [A] links clupeid condition to food via August zooplankton,
  but tests no prey-by-density interaction.
- Generic: the Crowley-Martin form (interference continues during handling) predicts interference whose relative cost
  does not shrink at high prey; Stouffer & Novak (sibling note) found BD often adequate but no robust preference among
  predator-dependent forms. I found no planktivore study that tests BD against Crowley-Martin.

### Calculation [D]
- Ratio of the BD per-doubling cut, winter vs summer, at w = 0.7: 0.40/0.11 = 3.6x. Ratio of the level cut: 66 %/12 % = 5.4x.
- Under a Crowley-Martin form, f = a Z^2 / ((1 + S)(1 + wB)), the per-doubling cut is wB/(1 + 2wB) in every season:
  matching r = 3-5 % needs wB = 0.03-0.06, w = 0.01-0.02. Total cell intake still saturates, at
  a Z^2 / ((1 + S) w), as B -> inf, so the finite-optimum property that removed the bands is kept.

### Inference
- Theory (Eggers, BD) supports a relatively stronger school-scale cost at low prey, and the BD shape is defensible in
  that sense. What is not supported is the size: at w = 0.7 the winter relative effect lands in a range nothing in the
  herring literature approaches. The one planktivore experiment found no prey-density modulation (abstract only).
- A seasonal mismatch in the model adds to this. In winter many herring in a cell are resting/hiding (not eating),
  but I counts the total biomass, so non-feeding fish suppress the feeders. No behavioural reading of interference
  supports that, because interference needs feeding contacts. With I = w x (eating biomass) the winter I would fall in
  proportion to the eat share (not measured here).

---

## Q4. Verdict

### Inference
- **Is interference justified at all for a schooling planktivore?** Weakly at best. The literature supports
  (i) exploitative, stock-scale density dependence through shared zooplankton (often driven by sprat, not herring), and
  (ii) school-scale costs (front-fish depletion, overlap) that are offset or reversed by facilitation (Milne 2005:
  intake rose with school size). Neither is a function of 1-km cell biomass in the way w B assumes, and the first is
  already modelled explicitly.
- **w = 0.7 is unsupported by herring data, and too high as a year-round value.** It sits near Ecosim's adult-herring
  K = 4 only at mid-season S. Its intake cut per doubling is ~2-5x the growth-based range in summer and ~8-20x in winter
  (as a w, 2.5-8x and 35-120x above the growth-based fits), and in winter
  it cuts per-capita intake ~3x at today's cell densities. Its only justification is the band artefact, and section 86.7
  says weaker values were never tested.
- **Plausible range** (two numbers, because one BD w cannot satisfy both seasons):
  - winter-constrained (S ~0.2, r <= 5 %): **w ~0.005-0.02**;
  - summer-constrained (S ~16, r <= 5-10 %): **w ~0.1-0.3** (Ecosim K = 4 would allow up to ~1.7 there, with the
    refuge caveat).
  - If w stays in BD form, a single compromise of **w ~0.02-0.05** gives a winter cut per doubling of 5-11 % and a
    summer cut of 0.4-1 % [D]; w = 0.1 already gives 18 % in winter. About 0.05 is the largest value I would defend from
    the literature. Whether it still clears the bands is the untested question from 86.7.
- **What would match the literature better (inference only, for discussion, no code change made):**
  1. Lower w to <= 0.1 and run the 86.4 band diagnostic (`tools/probes/axis_collapse.py`; aniso, nyq_x) to find the
     smallest w that keeps nyq_x near 0.
  2. Count only the eating biomass in I, so resting winter herring do not suppress feeders.
  3. Alternatively, use a Crowley-Martin form for herring, (1 + S)(1 + wB), with w ~0.01-0.02. That gives a
     season-independent ~3-5 % per doubling and keeps the finite cell-intake ceiling.
  Any of these changes intake and so the herring energy budget; per AGENTS.md they need tests and a training run, and
  w = 0.7 must not be lowered without the band check.
- **Relevance to the winter food problem (winter note):** the model's Dec-Jan herring ration is 1.5-3x below
  the literature. At winter S, removing most of the w = 0.7 interference would raise per-capita intake by up to
  (1 + S + I)/(1 + S) ~3x at 3.4 t/km2. That alone could close the gap the winter note attributed to low intake and
  the 10 % night attack rate. The two levers interact and should not both be pulled at full strength without a probe.

---

## Summary table

| Item | Literature / calculation | Model (w = 0.7, B = 3.4) | Verdict |
|---|---|---|---|
| Behavioural interference measured in herring | none found | - | Unsupported |
| Planktivore interference experiment | roach: groups eat less, exploitation + space (Maszczyk 2014, abstract, no numbers) | - | Exists, size unknown |
| Schooling and intake in a field planktivore | cisco: intake rises with school size (Milne 2005) | - | Facilitation can dominate |
| Herring density-dependent growth | NSS: l_inf -1.7 %/doubling (Stenevik 2022); 2/13 stocks significant (Brunel 2010); Baltic: none / weak, sprat-driven | - | Weak, exploitative |
| Intake cut per doubling, empirical | ~0-5 % (incl. depletion) [D] | winter 40 %, summer 11 % | Too strong, esp. winter |
| Ecosim K, North Sea 1991 | adult herring 4, juvenile 2.1, sprat 2.2 on copepods [D] | equals K = 4 at S ~7-8 | Mid-season match only |
| w from growth data | winter-fit 0.006-0.02; summer-fit 0.09-0.28 [D] | 0.7 | 2.5-8x (summer) to 35-120x (winter) high |
| Prey-density modulation | Eggers: cost lower at high prey (theory); roach: no prey effect (abstract) | BD: winter/summer 3.6x | Direction plausible, size not |
| Aggregation factor school vs cell | ~10^3.5-10^4.5 [D] | cell-mean term | School-scale costs not cell-density dependent |
| What I counts | interference needs feeding contacts | total biomass incl. resting | Mismatch, strongest in winter |

## Open gaps
- Maszczyk et al. 2014 effect sizes (group vs solitary intake at three prey densities) were not obtained; they are the
  closest thing to a planktivore m and should be read if the PDF can be fetched.
- No herring packing density in fish m-3 was verified (Misund 1993 not read); the aggregation factor rests on a
  secondary "order of magnitude below tank density".
- No field test of per-capita herring ration against local (km-scale) herring density was found; Casini et al. 2011
  (rectangle-scale condition vs density) is the closest and was read as abstract only.
- The Stenevik density range, and whether NSS occupied area scales with abundance, were not extracted, so the
  conversion to *local* density per doubling is uncertain by a factor of ~2.
- Ecosim K values come from my own column alignment of a rotated table; worth a visual check of PDF page 78
  (copepod row, predator columns 28-30).
- The band-removal threshold in w is unknown (section 86.7); it is a model experiment, not a literature question.
