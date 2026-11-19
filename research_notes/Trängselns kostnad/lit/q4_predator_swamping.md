# Q4. Does field predation on demersal fish swamp?

Literature note for "Trängselns kostnad" (research only; no code or library change). Written 2026-11-19.

Grades: [F] full text read here (or a cached full text in a sibling note, read again here); [A] abstract read
(verbatim abstract via OpenAlex or the PDF's first page); [S] secondary (a search-engine summary, a citing
paper, or a sibling note, whose own grade is given). "(computed)" = arithmetic done here on the source's numbers.
Model scale for comparison: 1 km cell, 6 h tick.

## 0. Bottom line

- **(a) Aggregative response: present for fish on a sediment-anchored prey, weak relative to prey clumping, and
  not universal.**
  - Fish predators on sandeel are more numerous at stations with more sandeel (~10 km station spacing). Log-log
    coefficients, read here as elasticities (computed, read as elasticity): numbers 0.23 (whiting vs day
    sandeel) and 0.34 (lesser weever vs night sandeel) (Engelhard 2008 Table 3 [F]).
  - Whiting showed no aggregative response to clupeids or juvenile gadoids at five North Sea stations (Rindorf &
    Gislason 2005 [F]). Catches of predator and prey were slightly negatively correlated. Between Engelhard's
    two grids, whiting were less abundant in the high-sandeel grid.
  - Haddock: "the aggregation was less obvious" (Temming 2004 [F]).
  - One strong counter-case: whiting removed a juvenile cod hot spot completely (Temming 2007 [S]).
  - Harbour and grey seals: no numerical-response evidence was found. McConnell 1999 [A] and Iorio-Merlo 2022 [S]
    show patch fidelity and ARS, not convergence. The only pinniped numerical response is Steller sea lions at a
    prespawning eulachon run (Sigler 2004 [A]), i.e. the separate spawning case.
- **(b) Patch-scale saturation: intake plateaus, but usually well below gut capacity.**
  - At the plateau, per-capita intake still rises with local prey density, but weakly. Prey-specific consumption
    (sandeel eaten, not total ration) has elasticity 0.08-0.14 (Engelhard [F], computed, read as elasticity). Cod
    on capelin show ~0.08 above the median capelin density (Fall 2021 via sibling, computed).
  - The plateau is not a full gut. Whiting plateau at 6-17 % of maximum ration (Rindorf & Gislason [F]).
    Piscivores carry 2-3x spare digestive capacity (Armstrong & Schindler 2011 [S]) and often run empty (Arrington
    2002 [A]).
  - Porpoises feed nearly continuously, at rates that follow encounters (0-550 prey/h). Their gut throughput binds
    only at peak rates (Wisniewska 2016 [F], computed).
- **Direct field tests of per-capita risk: both signs occur, and the scale ordering is not clean.**
  - Inverse (safety in numbers): White & Warner 2007 microsites (cm) [A]; Sandin & Pacala on continuous reef [S];
    Rindorf & Gislason's clupeid mortality index at stations (not significant; authors: "not conclusive") [F];
    Engelhard computed at ~10 km (per-capita sandeel mortality elasticity about -0.6).
  - Direct (predators aggregate): White & Warner sites (~2400 m2, km apart) [A]; Anderson 2001 field plots [A];
    Temming 2007 hot spot (18 km2) [S].
  - The cm-to-site ordering holds only inside the reef studies. On which side a 1 km model cell falls is
    unresolved.
- **Implication for the model [J].**
  - Taken at face value, the field does not show that swamping is absent. Extrapolating Engelhard's coefficients,
    a 30-90x clump draws 30^0.23 = 2.2x to 90^0.34 = 4.6x the predators. The model delivers 1.1-3.4x.
    Per-capita risk in such a clump is about 0.06-0.12 (whiting coefficients) to 0.24-0.34 (weever coefficients)
    of that in spread prey (computed).
  - This rests on four things:
    - The coefficients are on log(x+1) with many zero stations, and the paper does not call them elasticities.
    - Measurement error in the sandeel index flattens slopes, so the true response may be stronger.
    - Station contrasts are extrapolated to a 30-90x clump in a 1 km cell.
    - The model's 1.1-3.4x is a per-tonne-weighted statistic, not the same quantity.
  - On this reading, realistic aggregative and functional responses would leave much of the swamping payoff in
    place. The 30-90x vs 3.1 gap then needs costs of clumping other than predator response (Q1-Q3), unless strong
    hot-spot responses (Temming 2007, one [S] case) are the rule for demersal juveniles.

## 1. Key findings

| Study | Predator | Prey | Scale (space / time) | Measured | Magnitude | Grade |
|---|---|---|---|---|---|---|
| Engelhard et al. 2008 | whiting, lesser weever, grey gurnard | sandeel (Dogger Bank) | 2 grids; transects 27 nm long, 6.75 nm (12.5 km) apart; 60 stations; day vs night; May 2004-06 | log-log mixed model: predator number and consumption vs local sandeel | Whiting number +0.23 +- 0.06 vs day sandeel (p 0.0001), -0.42 vs night sandeel; per-capita consumption +0.08 +- 0.03. Weever number +0.34 +- 0.10 vs night sandeel; per-capita +0.14 +- 0.04; total consumption +0.68. Between grids, whiting were less abundant in the high-sandeel grid (-2.60). | [F] Table 3 |
| same (computed) | | | | per-capita sandeel mortality elasticity = total-consumption coefficient - 1 | Whiting 0.38 - 1 = -0.62; weever 0.68 - 1 = -0.32. Risk per sandeel falls with local sandeel density. log(x+1) offsets ignored. | (computed) |
| Temming et al. 2004 | whiting, haddock | sandeel (northern North Sea) | study area ~15 x 20 nm with a 10 x 10 nm box; 5-day survey, July 1996; GOV hauls 30 min | predator density vs sandeel in stomachs; stomach fullness | "pronounced aggregation of whiting" where sandeel were highest; "In haddock the aggregation was less obvious". Whiting density (n/30 min) = 455 + 718 x mean sandeel g/stomach, r2 0.48 (night 0.61); haddock r2 0.01, ns. Whiting stomach content 2.58 g outside the box (the patch) vs 0.50 g inside; empty 10.5 % vs 49.4 %. | [F] abstract, Fig. 6, p. 1358 |
| Rindorf & Gislason 2005 | whiting 20-35 cm | sprat, herring, juvenile gadoids | 5 North Sea stations, 48 h round-the-clock each | functional and aggregative response; mortality index | "No aggregative response of the predators was detected towards any of the prey and catches of prey and predators were slightly negatively correlated." Clupeid intake "saturated around a total intake of 10 and 17% of maximum daily ration", gadoids ~6 %: "never limited by gastric processing rate". Clupeid mortality index "decreased to virtually zero at high densities" (correlations not significant; "not conclusive"). | [F] (cached DTU postprint, lines 86-115, 965-1010) |
| Temming, Floeter & Ehrich 2007 | whiting | juvenile cod (0-group) | 18 km2, 5 d | removal of a local aggregation | ">50 million juvenile cod within an area of only 18 km2 was entirely wiped out in 5 d by predatory whiting". That is 2.8e6 cod/km2 (computed). No swamping at the hot-spot scale. | [S] (Fauchald 2009 [F]; abstract not obtainable) |
| Rose & Leggett 1990 | cod | capelin | acoustic transects, Newfoundland | spatial correlation vs scale | In phase at > 4-10 km. Out of phase below the aggregation size (3.5 km); most negative at 2-3 km. | [S] (sibling "Vitlingens uteblivna predation" [A]) |
| Fall et al. 2021 | cod | capelin (Barents Sea) | 2 km segments, ~2000 stomachs | consumption vs capelin density; cod-density effect | Consumption rises sharply below the median and then levels off. Cod density had no effect (no interference). Per-capita elasticity ~0.17 below the median and ~0.08 from the median to 300x the median (computed from the sibling's by-eye digitisation). | [S] (sibling "Födobegränsning/fish_krill" [F]); doi 10.1111/ecog.05473 |
| Sveegaard et al. 2012 (MEPS) | harbour porpoise, mackerel | herring (Kattegat-Skagerrak) | ~870 km2 cells, Jun-Aug pooled, 1998-2010 | partial Mantel; hurdle model | Porpoise-herring partial Mantel statistic 0.254 (p < 0.0001). Porpoise distribution "best explained solely by the distribution of herring". Box scale, not patch scale; no slope. | [F] Mantel table p. 249, Table 2 |
| Sigler, Womble & Vollenweider 2004 | Steller sea lion | prespawning eulachon | one bay; 22 km to the source haul-out; weeks | sea lion counts vs prey | "Sea lion abundance increased as eulachon gathered ..., peaked as eulachon abundance peaked, and decreased as the eulachon moved up-river"; animals left a haul-out 22 km away. Also: "a large surplus of prey was available per sea lion", i.e. the predators were swamped at the spawning aggregation. | [A] |
| McConnell et al. 1999 | grey seal | (sandeel habitat) | 14 seals, North Sea | trip structure | "local, repeated trips ... to discrete offshore areas"; 88 % return to the same haul-out; trips mean 2.33 d and 39.8 km. Patch fidelity, no prey field. | [A] |
| Iorio-Merlo et al. 2022 | harbour seal | (prey encounters from accelerometry) | 1 km cells, previous month; Moray Firth | HMM switch to area-restricted search (ARS) | Both memory and prey encounters raise P(start ARS). ARS steps 165-207 m per 5-dive batch. Trips mean 38.65 h. | [S] (sibling "Minne och beständighet" [F]) |
| Sharples et al. 2012 | harbour seal | none measured | 7 UK regions | trip distance and duration | Mean trip distance 11-21 km (Shetland, Orkney, Thames) to 100.6 km (Moray Firth); mean duration < 1 d to 4.5 d; site fidelity. No prey field. | [F] |
| Bowen et al. 2002 | harbour seal (video) | sand lance schools, flounder | single dives, minutes | tactics | "Males made 3.2 +- 0.4 capture attempts before re-establishing contact with the school" (n = 52 schools): the seal stays with the school. Eight males regurgitated meals while diving (read as gorging at a patch [J]). | [F] p. 240 |
| Wisniewska et al. 2016 | harbour porpoise (DTAG) | 3-10 cm fish | 15-23 h tags | buzz rates | "forage nearly continuously day and night", up to 550 prey/h; 0-200/h by day, 50-550/h after dusk. Stomach 1.9 kg, passage ~140 min. "starve to death in less than a week". | [F] (cached accepted manuscript) |
| same (computed) | | | | gut throughput vs intake | Throughput ~1.9 kg / 2.33 h = 0.8 kg/h. At 550/h x ~2 g (prey mass from sibling "Daggdjurens attackrater" [S]) = 1.1 kg/h the gut binds; at 100/h x 2 g = 0.2 kg/h (about 0.25 of throughput) it does not. Binding only at peak rates. | (computed) |
| Benoit-Bird et al. 2013 | fur seals, murres, kittiwakes | pollock, krill | 10 km transects, 200 km radius | predator use vs prey metrics | No relation to areal biomass, density or numerical abundance. Use was predicted by "prey patch characteristics such as depth and local density within spatial aggregations". | [A] |
| Fauchald 2009 (review) | seabirds (also cod and seals on capelin) | schooling prey | multi-scale | synthesis | Predator-prey match is weak at small scales and stronger at large scales. Theory (Abrams 2007, as summarised there): "Predator satiation made predator dilution possible", so prey aggregate and predators chase them from patch to patch. Theory, not data. | [F] |
| Armstrong & Schindler 2011 | 38 fish species | - | daily | capacity vs realised feeding | Predators keep the capacity to feed at 2-3 times the rate they realise on average. | [S] (search summary; not obtained) |
| Arrington et al. 2002 | 254 fish species | - | - | % empty stomachs | "Piscivorous fishes seem to be the only trophic group that regularly experience long periods of empty stomachs"; mean empty 16.2 +- 1.2 %. | [A] |
| ICES CRR 219 (1991 stomach programme) | saithe ages 3-9+ | - | quarters | % empty | Empty 25-50 % in Q1, 0-3 % in Q2, 8-20 % in Q3. | [S] (sibling "Sejens diet på unga torskfiskar" [S]) |
| White & Warner 2007 | reef piscivores | juvenile bluehead wrasse | microsites cm apart; sites ~2400 m2, km apart | per-capita mortality vs group size and density | "per capita mortality decreased with group size but was not related to density" at the microsite scale; "At the larger spatial scale, however, per capita mortality increased with settler density". | [A] |
| Anderson 2001 | kelp bass | kelp perch | lab tanks vs field plots | per-capita mortality vs density | Lab (fixed predators): "inversely density dependent to density independent". Field: "strongly density dependent, with evidence for an aggregative response". | [A] |
| Sandin & Pacala 2005 | reef piscivores | Chromis cyanea aggregations | continuous reef | strikes per capita | Inversely density dependent on continuous reefs. | [S] |
| Laurel et al. 2003 | juvenile-cod predators (Greenland and Atlantic cod, cunner, sculpin) | tethered age-0 cod | eelgrass patches 0.3-22 m2 | predation vs habitat patch size | Predator density 2.3x higher in the largest (22 m2) patch (0.086 pred m-2). Habitat-patch study, not a prey-density test. | [F] |
| Laurel et al. 2004 | - | age-0 cod | seine sites | prey aggregation | Juvenile cod "aggregated more in sand than eelgrass habitat", more so at high density, i.e. they aggregate where cover is lacking. No mortality data. | [A] |

## 2. Notes by sub-question

### (a) Aggregative numerical response

- **Fish predators on sandeel** are the only quantified slopes.
  - Engelhard's log-log coefficients are elasticities: whiting numbers 0.23, weever 0.34.
  - The response is tied to the prey's diel phase. Whiting follow the day (pelagic) sandeel distribution and are
    negatively related to the night distribution (-0.42); weever do the reverse.
  - Temming 2004 shows the same pattern over a 5-day survey on a patch a few nm across. The authors infer
    area-restricted search ("changed their swimming behaviour"), not choice by sight, at > 80 m depth.
  - Distance and time: day and night sandeel centres "up to 15 km apart" (Engelhard). Whether predators re-sort
    within a diel cycle was not observed [J]. Temming 2004 found stomachs filling overnight on the patch, but
    "no direct evidence that the predators actually stay".
- **The same predator (whiting) gives no aggregative response on clupeids and juvenile gadoids** (Rindorf &
  Gislason). The response depends on the prey type.
  - Sandeel are anchored to sediment, a predictable "spatial handicap" of the prey in Fauchald's terms.
  - Pelagic schools and mobile juveniles are not anchored. Rose & Leggett: cod and capelin are out of phase below
    3.5 km.
  - The Skagerrak benthivores are mobile demersal fish, closer to the second case. This is an inference [J].
- **Hot spots.** Temming 2007 is the one documented case of complete removal of a gadoid aggregation, by whiting,
  within 5 d over 18 km2 [S]. Its abstract was not obtainable, so neither the number of whiting nor the time over
  which they arrived is known here.
- **Mammals.**
  - Pinnipeds track predictable aggregations over days to weeks within 20-40 km of haul-outs (Sigler [A], McConnell
    [A]).
  - Harbour seals return to the same 1 km ARS areas month to month and switch to ARS on prey encounters
    (Iorio-Merlo [S]).
  - Porpoises match herring at the ~870 km2, 3-month scale (Sveegaard [F]).
  - No study gives a mammal's numerical slope against prey density at patch scale [gap].
  - Mammal search in the field is still patch tracking: Bowen's seals stay with a school.

### (b) Saturation at patch scale

- **The field "plateau" is a capture or time window, not a full gut.**
  - Whiting fish intake saturates at 6-17 % of maximum ration (Rindorf & Gislason). They take fish "almost
    exclusively during dawn and dusk".
  - Cod on capelin level off (Fall 2021 [S]). That pair is described there as digestion-limited at observed
    densities. It is the one case of near-gut saturation.
- **The two readings act differently on swamping.**
  - A plateau below the gut ceiling still fixes per-predator intake per day, so it swamps the same way as a full
    gut. "Predators are hungry" (Armstrong & Schindler; Arrington; saithe Q1 empties) is therefore not evidence
    against swamping. This matches the caller's test, where hungrier predators cut the clumped/spread payoff only
    ~20 %.
  - What would break swamping is a per-capita intake elasticity near 1 at patch scale. Measured values are
    0.08-0.17, and the sandeel values are prey-specific consumption, not total ration.
- **Mammals.**
  - Porpoises: intake follows encounter rate across 0-550/h and binds on gut throughput only at the top
    (computed).
  - Seals: they appear to gorge on patches (regurgitation; Bowen [F], reading [J]), and the eulachon case gave "a large surplus of prey
    ... per sea lion" [A]. Both point to satiation at rich patches.

### Direct field tests of swamping (per-capita risk vs group size or density)

- **Inverse (safety in numbers).**
  - White & Warner: microsites cm apart.
  - Sandin & Pacala: continuous reef.
  - Rindorf & Gislason: clupeid mortality index at stations (not significant).
  - Engelhard computed: -0.62 / -0.32 at ~10 km stations, even with an aggregative response present.
- **Direct (driven by predator aggregation).**
  - White & Warner: ~2400 m2 sites.
  - Anderson 2001: field plots.
  - Temming 2007: 18 km2 hot spot.
- **Scale.** Within the reef studies, inverse at the group and direct at the site scale is the reading of White &
  Warner. Across studies the scales overlap: inverse cases reach ~10 km, and direct cases start at ~50 m sites. So
  scale alone does not decide the sign, and on which side a 1 km cell falls is unresolved. No two-scale test
  exists for North Sea demersal fish [gap].
- **Spawning aggregations and eggs** (brief): predator satiation at spawning aggregations is the classic
  hypothesis (Johannes 1978 [S, via theses]). Egg predators also converge on spawning sites [S]. The Sigler 2004
  surplus is a pinniped example of predators swamped at a spawning aggregation [A]. This is a separate case from
  standing demersal clumps.

## 3. Gaps and access

- Not obtained in full:
  - Temming et al. 2007 (Ecosystems, doi 10.1007/s10021-007-9066-3): paywalled, no abstract in OpenAlex or
    Crossref.
  - Armstrong & Schindler 2011 (doi 10.1038/nature10240): paywalled, no abstract record.
  - Arrington 2002, Anderson 2001, White & Warner 2007, McConnell 1999, Sigler 2004: abstract only.
  - Rose & Leggett 1990 (doi 10.2307/1940245): via sibling note only.
  - Womble et al. 2005 (doi 10.3354/meps294271): not read.
- No field slope of seal or porpoise numbers against prey density at 1-10 km.
- No North Sea demersal-fish test of per-capita mortality against local density.
- No saithe aggregative-response study found (budget).
- Engelhard's elasticities are for sandeel, a sediment-anchored prey. Transferring them to mobile benthivores is a
  judgement [J], and it likely overstates the response.

## 4. References (DOI)

- Anderson TW 2001. Ecology 82:245-257. doi 10.1890/0012-9658(2001)082[0245:PRPRAD]2.0.CO;2
- Armstrong JB, Schindler DE 2011. Nature 476:84-87. doi 10.1038/nature10240
- Arrington DA, Winemiller KO, Loftus WF, Akin S 2002. Ecology 83:2145-2151. doi 10.1890/0012-9658(2002)083[2145:HODFRO]2.0.CO;2
- Benoit-Bird KJ et al. 2013. PLoS ONE 8:e53348. doi 10.1371/journal.pone.0053348
- Bowen WD, Tully D, Boness DJ, Bulheier BM, Marshall GJ 2002. MEPS 244:235-245. doi 10.3354/meps244235
- Engelhard GH et al. 2008. MEPS 369:213-227. doi 10.3354/meps07575
- Fall J et al. 2021. Ecography 44:802-815. doi 10.1111/ecog.05473
- Fauchald P 2009. MEPS 391:139-151. doi 10.3354/meps07818
- Iorio-Merlo V et al. 2022. Proc R Soc B 289:20212261. doi 10.1098/rspb.2021.2261
- Laurel BJ, Gregory RS, Brown JA 2003. MEPS 251:245-254. doi 10.3354/meps251245
- Laurel BJ et al. 2004. MEPS 272:257-270. doi 10.3354/meps272257
- McConnell BJ, Fedak MA, Lovell P, Hammond PS 1999. J Appl Ecol 36:573-590. doi 10.1046/j.1365-2664.1999.00429.x
- Rindorf A, Gislason H 2005. JEMBE 324:1-19. doi 10.1016/j.jembe.2005.04.013
- Rose GA, Leggett WC 1990. Ecology 71:33-43. doi 10.2307/1940245
- Sandin SA, Pacala SW 2005. Ecology (volume, pages and DOI not verified)
- Sharples RJ, Moss SE, Patterson TA, Hammond PS 2012. PLoS ONE 7:e37216. doi 10.1371/journal.pone.0037216
- Sigler MF, Womble JN, Vollenweider JJ 2004. CJFAS 61:1475-1484. doi 10.1139/f04-086
- Sveegaard S et al. 2012. MEPS 468:245-253. doi 10.3354/meps09959
- Temming A, Götz S, Mergardt N, Ehrich S 2004. J Fish Biol 64:1351-1372. doi 10.1111/j.0022-1112.2004.00400.x
- Temming A, Floeter J, Ehrich S 2007. Ecosystems 10:865-876. doi 10.1007/s10021-007-9066-3
- White JW, Warner RR 2007. Ecology 88:3044-3054. doi 10.1890/06-1949.1
- Wisniewska DM et al. 2016. Curr Biol 26:1441-1446. doi 10.1016/j.cub.2016.03.069

Files: the PDFs downloaded for this question are in `../papers/q4/` (licences in `LICENCES_q4.md`). Temming 2004 was
read from `../../Attackrater ur mötesteori/data/benth_validation/`; Engelhard 2008 from the download here. Rindorf & Gislason
2005 was read from `../../Attackrater ur mötesteori/data/gap_validation/rindorf2005/postprint.txt`. Wisniewska
2016 was read from `../../Tumlarens byten/data/lit/`. A PDF fetched under a mis-guessed DOI (10.3354/meps08003,
humpback whales) was deleted.
